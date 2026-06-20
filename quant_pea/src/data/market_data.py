"""Market data ingestion — Yahoo Finance + FRED + ECB. 100% gratuit."""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from loguru import logger
from tqdm import tqdm

from ..utils.config import cfg
from .database import get_conn, upsert_prices, load_prices

ROOT = Path(__file__).resolve().parents[3]


# ─────────────────────────────────────────────
# Yahoo Finance
# ─────────────────────────────────────────────

def download_yahoo(tickers: list[str], period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance, returns long-format DataFrame."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    logger.info(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                logger.warning(f"No data for {ticker}")
                continue
            sub = raw[ticker].copy().dropna(how="all")
            sub.index = pd.to_datetime(sub.index).tz_localize(None)
            for dt, row in sub.iterrows():
                rows.append({
                    "date": dt.date(),
                    "ticker": ticker,
                    "open": row.get("Open"),
                    "high": row.get("High"),
                    "low": row.get("Low"),
                    "close": row.get("Close"),
                    "volume": row.get("Volume"),
                    "adj_close": row.get("Close"),
                })
    else:
        sub = raw.dropna(how="all")
        sub.index = pd.to_datetime(sub.index).tz_localize(None)
        ticker = tickers[0]
        for dt, row in sub.iterrows():
            rows.append({
                "date": dt.date(),
                "ticker": ticker,
                "open": row.get("Open"),
                "high": row.get("High"),
                "low": row.get("Low"),
                "close": row.get("Close"),
                "volume": row.get("Volume"),
                "adj_close": row.get("Close"),
            })

    df = pd.DataFrame(rows)
    logger.info(f"Downloaded {len(df)} rows")
    return df


def get_latest_close_prices(tickers: list[str]) -> pd.DataFrame:
    """Get the most recent close price per ticker. Used for snapshot."""
    df = load_prices(tickers=tickers)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "close", "date"])
    idx = df.groupby("ticker")["date"].idxmax()
    return df.loc[idx, ["ticker", "close", "date"]].reset_index(drop=True)


def close_pivot(tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    """Return wide-format close prices (date × ticker)."""
    df = load_prices(tickers=tickers, start=start)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return pivot.ffill()


# ─────────────────────────────────────────────
# FRED — macro data (gratuit, clé API gratuite)
# ─────────────────────────────────────────────

FRED_SERIES = {
    "DGS10":    "US 10Y yield",
    "T10YIE":   "US 10Y inflation breakeven",
    "VIXCLS":   "VIX",
    "DEXUSEU":  "EUR/USD",
    "BAMLH0A0HYM2": "US HY spread",
    "T10Y2Y":   "US 10Y-2Y spread (recession indicator)",
}

def download_fred(series_ids: list[str] | None = None) -> pd.DataFrame:
    """Download macro series from FRED. Needs FRED_API_KEY env var (free at fred.stlouisfed.org)."""
    c = cfg()
    api_key = c["data_sources"]["fred"].get("api_key", "")
    if not api_key:
        logger.warning("No FRED_API_KEY — skipping macro data. Get one free at fred.stlouisfed.org")
        return pd.DataFrame()

    ids = series_ids or list(FRED_SERIES.keys())
    rows = []
    for sid in tqdm(ids, desc="FRED"):
        try:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={sid}&api_key={api_key}&file_type=json"
                f"&observation_start=2010-01-01"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("observations", [])
            for obs in data:
                v = obs["value"]
                if v == ".":
                    continue
                rows.append({"date": obs["date"], "series_id": sid, "value": float(v)})
            time.sleep(0.3)
        except Exception as e:
            logger.error(f"FRED {sid}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ─────────────────────────────────────────────
# ECB — données gratuites
# ─────────────────────────────────────────────

ECB_SERIES = {
    "FM.B.U2.EUR.FR2.BB.U2_2Y.YLD":  "ECB 2Y OIS rate",
    "FM.B.U2.EUR.FR2.BB.U2_10Y.YLD": "ECB 10Y OIS rate",
}

def download_ecb(series_key: str) -> pd.DataFrame:
    """Download a single ECB data series."""
    url = f"https://data-api.ecb.europa.eu/service/data/{series_key}?format=csvdata"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        rows = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                dt = parts[0].strip()
                val = float(parts[1].strip())
                rows.append({"date": dt, "value": val})
            except (ValueError, IndexError):
                continue
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
    except Exception as e:
        logger.warning(f"ECB {series_key}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Returns computation
# ─────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """Compute returns from price matrix. method: 'simple' or 'log'."""
    prices = prices.ffill().dropna(how="all")
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def annualize_returns(returns: pd.DataFrame, freq: int = 252) -> pd.Series:
    return (1 + returns.mean()) ** freq - 1


def annualize_vol(returns: pd.DataFrame, freq: int = 252) -> pd.Series:
    return returns.std(ddof=1) * np.sqrt(freq)


def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta of asset vs benchmark."""
    def _beta(x, y):
        cov = np.cov(x, y, ddof=1)
        return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan

    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    betas = aligned.iloc[:, 0].rolling(window).apply(
        lambda x: _beta(x.values, aligned.iloc[x.index, 1].values) if len(x) == window else np.nan,
        raw=False,
    )
    return betas
