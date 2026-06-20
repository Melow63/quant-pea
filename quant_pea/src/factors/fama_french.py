"""
Factor models — Fama-French 5 facteurs + Momentum + Low-Vol + Quality.
Téléchargement Kenneth French Data Library (gratuit).
Attribution factorielle du portefeuille.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from loguru import logger
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "processed" / "factors"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRENCH_DATA_URLS = {
    "ff3_us": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5_us": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_us": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "ff3_eu": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_3_Factors_daily_CSV.zip",
    "ff5_eu": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_5_Factors_daily_CSV.zip",
    "mom_eu": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_Mom_Factor_daily_CSV.zip",
}


def _download_french(url: str) -> pd.DataFrame:
    """Download and parse Kenneth French CSV zip."""
    cache_key = url.split("/")[-1].replace(".zip", ".parquet")
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        logger.debug(f"Loading factor data from cache: {cache_key}")
        return pd.read_parquet(cache_path)

    logger.info(f"Downloading factor data: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        content = z.read(csv_name).decode("utf-8", errors="ignore")

        lines = content.split("\n")
        # Find the header line (contains comma-separated factor names)
        header_idx = 0
        for i, line in enumerate(lines):
            if "Mkt" in line or "SMB" in line or "Mom" in line:
                header_idx = i
                break

        data_lines = []
        for line in lines[header_idx + 1:]:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            # Date column is yyyymmdd
            try:
                int(parts[0].strip())
                data_lines.append(line)
            except ValueError:
                break

        header = [h.strip() for h in lines[header_idx].split(",")]
        df = pd.read_csv(io.StringIO("\n".join([lines[header_idx]] + data_lines)))
        df.columns = [c.strip() for c in df.columns]

        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)
        df = df.apply(pd.to_numeric, errors="coerce") / 100  # percentages → decimals

        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} factor rows → {cache_key}")
        return df

    except Exception as e:
        logger.error(f"Failed to download factor data from {url}: {e}")
        return pd.DataFrame()


def load_ff5_europe() -> pd.DataFrame:
    """Load Fama-French 5 factors + Momentum for Europe."""
    ff5 = _download_french(FRENCH_DATA_URLS["ff5_eu"])
    mom = _download_french(FRENCH_DATA_URLS["mom_eu"])

    if ff5.empty:
        return pd.DataFrame()

    factors = ff5.copy()
    if not mom.empty:
        mom_col = [c for c in mom.columns if "Mom" in c or "WML" in c]
        if mom_col:
            factors["MOM"] = mom[mom_col[0]]

    # Rename to standard names
    rename = {}
    for col in factors.columns:
        col_u = col.upper().strip()
        if "MKT" in col_u or "RM-RF" in col_u:
            rename[col] = "MKT_RF"
        elif col_u == "SMB":
            rename[col] = "SMB"
        elif col_u == "HML":
            rename[col] = "HML"
        elif col_u == "RMW":
            rename[col] = "RMW"
        elif col_u == "CMA":
            rename[col] = "CMA"
        elif col_u == "RF":
            rename[col] = "RF"
        elif "MOM" in col_u or "WML" in col_u:
            rename[col] = "MOM"

    factors = factors.rename(columns=rename)
    return factors


# ─────────────────────────────────────────────
# Factor attribution via OLS regression
# ─────────────────────────────────────────────

def factor_attribution(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    factor_cols: Optional[list[str]] = None,
    rolling_window: Optional[int] = None,
) -> dict:
    """
    OLS factor regression: R_p = alpha + Σ β_i * F_i + ε
    Returns: loadings, t-stats, R², alpha (Jensen), factor contributions.
    """
    if factor_cols is None:
        factor_cols = [c for c in ["MKT_RF", "SMB", "HML", "MOM", "RMW", "CMA"] if c in factors.columns]

    rf = factors["RF"] if "RF" in factors.columns else pd.Series(0.0, index=factors.index)

    # Excess returns
    excess_port = portfolio_returns - rf.reindex(portfolio_returns.index, fill_value=0.0)

    X = factors[factor_cols].reindex(excess_port.index).dropna()
    y = excess_port.reindex(X.index).dropna()
    X = X.reindex(y.index)

    if len(y) < 30:
        return {"error": "Insufficient data for factor regression"}

    X_const = np.column_stack([np.ones(len(X)), X.values])
    result = np.linalg.lstsq(X_const, y.values, rcond=None)
    coeffs = result[0]
    alpha = coeffs[0]
    betas = coeffs[1:]

    # OLS statistics
    y_hat = X_const @ coeffs
    residuals = y.values - y_hat
    n, k = len(y), len(betas)
    sse = float(residuals @ residuals)
    sst = float(((y.values - y.values.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2
    se = np.sqrt(sse / (n - k - 1)) * np.sqrt(np.diag(np.linalg.pinv(X_const.T @ X_const)))
    t_stats = coeffs / se

    # Factor contributions to return
    factor_mean_returns = X.mean()
    contributions = {col: float(betas[i] * factor_mean_returns[col]) for i, col in enumerate(factor_cols)}
    contributions["alpha"] = float(alpha)
    contributions["specific"] = float(residuals.mean())

    # Annualized
    ann = 252
    out = {
        "alpha_daily": float(alpha),
        "alpha_annualized": float(alpha * ann),
        "alpha_tstat": float(t_stats[0]),
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "residual_vol_annualized": float(residuals.std(ddof=1) * np.sqrt(ann)),
        "betas": {col: float(betas[i]) for i, col in enumerate(factor_cols)},
        "t_stats": {col: float(t_stats[i + 1]) for i, col in enumerate(factor_cols)},
        "factor_return_contributions": contributions,
        "n_obs": n,
    }

    # Rolling if requested
    if rolling_window and len(y) >= rolling_window:
        roll_betas = {col: [] for col in factor_cols}
        roll_alpha = []
        roll_dates = []
        for end in range(rolling_window, len(y) + 1):
            start = end - rolling_window
            X_roll = X_const[start:end]
            y_roll = y.values[start:end]
            c_roll, _, _, _ = np.linalg.lstsq(X_roll, y_roll, rcond=None)
            roll_alpha.append(c_roll[0] * ann)
            for i, col in enumerate(factor_cols):
                roll_betas[col].append(c_roll[i + 1])
            roll_dates.append(y.index[end - 1])

        out["rolling"] = pd.DataFrame(
            {"alpha": roll_alpha, **roll_betas},
            index=roll_dates,
        )

    return out


# ─────────────────────────────────────────────
# Custom factor construction from price data
# ─────────────────────────────────────────────

def compute_momentum_score(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.Series:
    """
    12-1 momentum: return from t-252 to t-21.
    Standard Jegadeesh-Titman momentum factor.
    """
    if len(prices) < lookback + skip:
        return pd.Series(dtype=float)
    past_return = (prices.iloc[-skip] / prices.iloc[-(lookback + skip)]) - 1
    return past_return.sort_values(ascending=False)


def compute_low_vol_score(returns: pd.DataFrame, window: int = 126) -> pd.Series:
    """Realized volatility over past 6 months — low vol anomaly."""
    if len(returns) < window:
        return pd.Series(dtype=float)
    recent = returns.iloc[-window:]
    return recent.std(ddof=1) * np.sqrt(252)  # annualized vol


def compute_quality_score(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Quality proxy from price data: SR (Sharpe ratio) as quality signal.
    Ideally would use ROE/debt, but from prices only — use Sharpe as quality proxy.
    """
    if len(returns) < window:
        return pd.Series(dtype=float)
    r = returns.iloc[-window:]
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(252)
    return sharpe
