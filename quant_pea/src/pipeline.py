"""
Pipeline orchestrator — assemble tous les modules.
Point d'entrée pour l'app Streamlit et les scripts CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from .data.database import init_schema, load_prices, load_transactions, get_conn
from .data.market_data import close_pivot, get_latest_close_prices, compute_returns
from .accounting.fifo import build_fifo_ledger, positions_from_lots, compute_twr, compute_mwr, brinson_attribution
from .risk.models import (
    compute_risk_metrics, var_monte_carlo, risk_contribution,
    concentration_metrics, all_stress_tests, fit_egarch,
)
from .factors.fama_french import load_ff5_europe, factor_attribution
from .optimization.portfolio import run_all_optimizers, ledoit_wolf_cov
from .utils.config import cfg

ROOT = Path(__file__).resolve().parents[1]


def load_security_master() -> pd.DataFrame:
    sm_path = ROOT / "data" / "security_master.csv"
    if sm_path.exists():
        return pd.read_csv(sm_path)
    with get_conn(read_only=True) as con:
        return con.execute("SELECT * FROM security_master").df()


def get_positions_snapshot(use_live: bool = True) -> pd.DataFrame:
    """Build current positions from transactions + live prices."""
    tx = load_transactions()
    if tx.empty:
        return _load_positions_from_csv()

    tx["trade_datetime"] = pd.to_datetime(tx["trade_datetime"])
    tx["quantity"] = tx["quantity"].abs()
    open_lots, _ = build_fifo_ledger(tx)

    if open_lots.empty:
        return pd.DataFrame()

    sm = load_security_master()
    positions = positions_from_lots(open_lots, sm)

    # Attach current prices
    tickers = [t for t in positions["yahoo_ticker"].dropna().tolist() if t]
    if use_live and tickers:
        latest = get_latest_close_prices(tickers)
        price_map = dict(zip(latest["ticker"], latest["close"]))
        positions["current_price_eur"] = positions["yahoo_ticker"].map(price_map)
    else:
        positions["current_price_eur"] = positions["avg_price_gross"]

    positions["market_value_eur"] = positions["quantity"] * positions["current_price_eur"]
    positions["unrealized_pnl_eur"] = positions["market_value_eur"] - positions["total_cost_net"]
    total = positions["market_value_eur"].sum()
    positions["weight"] = positions["market_value_eur"] / total if total > 0 else 0.0
    positions["unrealized_return"] = positions["unrealized_pnl_eur"] / positions["total_cost_net"]
    positions = positions.sort_values("market_value_eur", ascending=False).reset_index(drop=True)
    return positions


def _load_positions_from_csv() -> pd.DataFrame:
    """Fallback: load from original CSV transactions (legacy compatibility)."""
    tx_path = ROOT / "data" / "transactions_actions.csv"
    funds_path = ROOT / "data" / "transactions_funds.csv"
    snap_path = ROOT / "data" / "market_snapshot_live.csv"
    sm_path = ROOT / "data" / "security_master.csv"

    if not tx_path.exists():
        return pd.DataFrame()

    tx = pd.read_csv(tx_path)
    tx["trade_datetime"] = pd.to_datetime(tx["trade_date"])
    tx["side"] = tx["side"].str.upper()
    tx["signed_quantity"] = tx["quantity"].where(tx["side"] == "BUY", -tx["quantity"])
    tx["signed_net_cash"] = tx["net_cash_eur"].where(tx["side"] == "BUY", -tx["net_cash_eur"])

    grp = tx.groupby(["security_id", "asset_name", "yahoo_ticker", "sector", "asset_class"]).agg(
        quantity=("signed_quantity", "sum"),
        net_invested_eur=("signed_net_cash", "sum"),
        gross_cost_eur=("gross_amount_eur", "sum"),
    ).reset_index()

    grp["avg_price_net_eur"] = grp["net_invested_eur"] / grp["quantity"]
    grp = grp[grp["quantity"] > 1e-6]

    if snap_path.exists():
        snap = pd.read_csv(snap_path)
        price_map = dict(zip(snap["yahoo_ticker"], snap["price_eur"]))
        grp["current_price_eur"] = grp["yahoo_ticker"].map(price_map).fillna(grp["avg_price_net_eur"])
    else:
        grp["current_price_eur"] = grp["avg_price_net_eur"]

    grp["market_value_eur"] = grp["quantity"] * grp["current_price_eur"]
    grp["unrealized_pnl_eur"] = grp["market_value_eur"] - grp["net_invested_eur"]
    total = grp["market_value_eur"].sum()
    grp["weight"] = grp["market_value_eur"] / total if total > 0 else 0.0
    grp["unrealized_return"] = grp["unrealized_pnl_eur"] / grp["net_invested_eur"]
    return grp.sort_values("market_value_eur", ascending=False).reset_index(drop=True)


def full_analytics(run_ml: bool = False, run_nlp: bool = False) -> dict:
    """
    Master analytics function — runs everything.
    run_ml: run ML signal model (slow first time)
    run_nlp: run sentiment scraping (requires internet)
    """
    c = cfg()
    out = {}

    # ── 1. Positions ────────────────────────────────────────────
    logger.info("Computing positions...")
    positions = get_positions_snapshot()
    out["positions"] = positions

    # ── 2. Price history ────────────────────────────────────────
    logger.info("Loading price history...")
    sm = load_security_master()
    tickers = [t for t in sm["yahoo_ticker"].dropna().tolist() if t]
    prices = close_pivot(tickers=tickers)

    if prices.empty:
        logger.warning("No price history — run scripts/update_data.py first")
        out["has_history"] = False
        return out

    out["has_history"] = True
    out["price_history"] = prices

    benchmark_ticker = c["portfolio"]["benchmark_ticker"]
    returns = compute_returns(prices)

    # ── 3. Portfolio history & TWR ──────────────────────────────
    logger.info("Computing portfolio history...")
    try:
        ticker_map = dict(zip(sm["security_id"], sm["yahoo_ticker"]))
        port_nav, flows = _compute_portfolio_nav(positions, prices, sm)
        if not port_nav.empty:
            twr = compute_twr(port_nav, flows)
            mwr = compute_mwr(port_nav, flows)
            out["portfolio_nav"] = port_nav
            out["portfolio_twr"] = twr
            out["portfolio_mwr"] = mwr
            out["external_flows"] = flows

            port_returns = port_nav.pct_change().dropna()
            bench_returns = (
                returns[benchmark_ticker].reindex(port_returns.index).dropna()
                if benchmark_ticker in returns.columns else None
            )

            # ── 4. Risk metrics ─────────────────────────────────
            logger.info("Computing risk metrics...")
            out["risk_metrics"] = compute_risk_metrics(port_returns, bench_returns)

            # VaR Monte Carlo
            if len(positions) >= 2:
                port_tickers = [t for t in positions["yahoo_ticker"].dropna() if t in returns.columns]
                if len(port_tickers) >= 2:
                    port_weights = (
                        positions.set_index("yahoo_ticker")["weight"]
                        .reindex(port_tickers).fillna(0.0)
                    )
                    mc_result = var_monte_carlo(
                        returns[port_tickers].dropna(how="all"),
                        port_weights,
                        n_simulations=c["risk"]["monte_carlo_simulations"],
                    )
                    out["var_monte_carlo"] = mc_result

            # EGARCH vol forecast
            try:
                out["garch"] = fit_egarch(port_returns)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Portfolio history computation failed: {e}")

    # ── 5. Risk contribution & concentration ────────────────────
    logger.info("Computing risk contribution...")
    try:
        port_tickers = [t for t in (positions["yahoo_ticker"].dropna().tolist() if not positions.empty else [])
                        if t in returns.columns]
        if len(port_tickers) >= 2:
            rets_port = returns[port_tickers].dropna(how="all")
            cov = ledoit_wolf_cov(rets_port)
            w = positions.set_index("yahoo_ticker")["weight"].reindex(port_tickers).fillna(0.0)
            # Reindex to security_id for display
            sec_map = dict(zip(sm["yahoo_ticker"], sm["security_id"]))
            w.index = [sec_map.get(t, t) for t in w.index]
            cov.index = [sec_map.get(t, t) for t in cov.index]
            cov.columns = [sec_map.get(t, t) for t in cov.columns]
            out["risk_contribution"] = risk_contribution(w, cov)
            out["concentration"] = concentration_metrics(w)
    except Exception as e:
        logger.warning(f"Risk contribution: {e}")

    # ── 6. Factor attribution ────────────────────────────────────
    logger.info("Computing factor attribution...")
    try:
        factors = load_ff5_europe()
        if not factors.empty and "port_returns" in dir():
            fa = factor_attribution(port_returns, factors, rolling_window=126)
            out["factor_attribution"] = fa
    except Exception as e:
        logger.warning(f"Factor attribution: {e}")

    # ── 7. Stress tests ──────────────────────────────────────────
    logger.info("Running stress tests...")
    try:
        if not positions.empty:
            out["stress_tests"] = all_stress_tests(positions)
    except Exception as e:
        logger.warning(f"Stress tests: {e}")

    # ── 8. Optimizer ─────────────────────────────────────────────
    logger.info("Running portfolio optimizer...")
    try:
        port_tickers = [t for t in (positions["yahoo_ticker"].dropna().tolist() if not positions.empty else [])
                        if t in returns.columns and t != benchmark_ticker]
        if len(port_tickers) >= 3:
            rets_opt = returns[port_tickers].dropna(how="all")
            max_w = c["optimization"]["max_weight"]
            min_w = c["optimization"]["min_weight"]

            # Market weights = equal weight (no index data available free)
            mw = pd.Series(1 / len(port_tickers), index=port_tickers)
            opt_results = run_all_optimizers(rets_opt, market_weights=mw, max_weight=max_w, min_weight=min_w)
            out["optimizer"] = opt_results
    except Exception as e:
        logger.warning(f"Optimizer: {e}")

    # ── 9. ML Signals (optional, slow) ──────────────────────────
    if run_ml:
        logger.info("Computing ML signals...")
        try:
            from .signals.ml_signals import compute_features, compute_targets, train_signal_model, predict_signals
            features = compute_features(prices)
            targets = compute_targets(prices)
            model_dict = train_signal_model(features, targets)
            signals = predict_signals(model_dict, features)
            out["ml_signals"] = signals
            out["ml_model"] = model_dict
        except Exception as e:
            logger.warning(f"ML signals: {e}")

    # ── 10. NLP Sentiment (optional) ────────────────────────────
    if run_nlp:
        logger.info("Computing sentiment...")
        try:
            from .nlp.sentiment import compute_portfolio_sentiment
            sm_active = sm[sm.get("include_analytics", "Y") == "Y"] if "include_analytics" in sm.columns else sm
            out["sentiment"] = compute_portfolio_sentiment(sm_active)
        except Exception as e:
            logger.warning(f"NLP sentiment: {e}")

    logger.info("Full analytics complete ✓")
    return out


def _compute_portfolio_nav(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    sm: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Compute daily portfolio NAV from positions and price history."""
    ticker_map = dict(zip(sm["security_id"], sm["yahoo_ticker"]))
    if positions.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Reconstruct daily holdings from transactions
    tx = load_transactions()
    if tx.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tx["trade_date"] = pd.to_datetime(tx["trade_date"])
    dates = prices.index

    # Build daily quantity matrix
    holdings = pd.DataFrame(0.0, index=dates, columns=sm["security_id"].tolist())
    flows = pd.Series(0.0, index=dates)

    qty_changes = tx.groupby(["trade_date", "security_id"])["quantity"].sum()
    cash_changes = tx.groupby("trade_date")["net_cash_eur"].sum()

    for (dt, sec), qty in qty_changes.items():
        dt = pd.Timestamp(dt)
        if dt in holdings.index and sec in holdings.columns:
            holdings.loc[dt:, sec] += qty

    for dt, cf in cash_changes.items():
        dt = pd.Timestamp(dt)
        if dt in flows.index:
            flows.loc[dt] = cf

    # Map to prices
    nav = pd.Series(0.0, index=dates)
    for sec in holdings.columns:
        ticker = ticker_map.get(sec)
        if ticker and ticker in prices.columns:
            nav += holdings[sec] * prices[ticker]

    nav = nav[nav > 0]
    return nav, flows.reindex(nav.index, fill_value=0.0)
