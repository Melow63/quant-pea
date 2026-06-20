"""
Risk engine — institutional grade.
VaR: Historical / Parametric / Monte Carlo full simulation
CVaR / Expected Shortfall
GARCH volatility forecasting (EGARCH)
Stress tests: historical scenarios + user-defined shocks
DCC correlation dynamics
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Value at Risk
# ─────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95, horizon: int = 1) -> float:
    """Historical simulation VaR. No distributional assumption."""
    clean = returns.dropna()
    if len(clean) < 30:
        return np.nan
    q = np.quantile(clean, 1 - confidence)
    return abs(q) * np.sqrt(horizon)


def var_parametric(returns: pd.Series, confidence: float = 0.95, horizon: int = 1,
                   distribution: str = "normal") -> float:
    """
    Parametric VaR. distribution: 'normal' or 't' (Student — fatter tails, more realistic).
    """
    clean = returns.dropna()
    mu = clean.mean()
    sigma = clean.std(ddof=1)

    if distribution == "t":
        nu, _, _ = stats.t.fit(clean)
        z = stats.t.ppf(1 - confidence, df=nu)
    else:
        z = stats.norm.ppf(1 - confidence)

    var_1d = -(mu + z * sigma)
    return max(var_1d * np.sqrt(horizon), 0.0)


def var_monte_carlo(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    horizon: int = 10,
    n_simulations: int = 10_000,
    use_t: bool = True,
) -> dict:
    """
    Full Monte Carlo VaR & CVaR using multivariate t-distribution.
    Captures fat tails and cross-asset correlations.

    Returns dict with var, cvar, simulated_pnl distribution.
    """
    rets = returns.dropna(how="all").fillna(0.0)
    w = weights.reindex(rets.columns).fillna(0.0).values
    w = w / w.sum() if w.sum() > 0 else w

    mu = rets.mean().values
    cov = rets.cov().values
    n_assets = len(mu)

    if use_t:
        # Fit degrees of freedom from portfolio return distribution
        port_ret = rets @ w
        df_t, _, _ = stats.t.fit(port_ret.dropna())
        df_t = max(df_t, 3.0)  # minimum 3 for finite variance
    else:
        df_t = np.inf

    # Cholesky decomposition for correlated draws
    try:
        L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-8)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov)))

    rng = np.random.default_rng(42)
    daily_pnls = np.zeros(n_simulations)

    for _ in range(horizon):
        if np.isinf(df_t):
            z = rng.standard_normal((n_simulations, n_assets))
        else:
            z = rng.standard_t(df=df_t, size=(n_simulations, n_assets))
        daily_rets = mu + z @ L.T
        daily_pnls += daily_rets @ w

    var_mc = float(-np.quantile(daily_pnls, 1 - confidence))
    cvar_mc = float(-daily_pnls[daily_pnls <= -var_mc].mean()) if (daily_pnls <= -var_mc).any() else var_mc

    return {
        "var": var_mc,
        "cvar": cvar_mc,
        "confidence": confidence,
        "horizon_days": horizon,
        "n_simulations": n_simulations,
        "simulated_pnl": daily_pnls,
        "pnl_percentiles": {
            "p1": np.percentile(daily_pnls, 1),
            "p5": np.percentile(daily_pnls, 5),
            "p10": np.percentile(daily_pnls, 10),
            "p25": np.percentile(daily_pnls, 25),
            "p50": np.percentile(daily_pnls, 50),
        },
    }


# ─────────────────────────────────────────────
# Full risk metrics suite
# ─────────────────────────────────────────────

def compute_risk_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf: float = 0.035 / 252,  # daily risk-free
) -> dict:
    """Complete risk metrics — Sharpe, Sortino, Calmar, VaR, CVaR, beta, alpha, IR, etc."""
    r = returns.dropna()
    if len(r) < 20:
        return {}

    n = len(r)
    ann_factor = 252

    ann_return = float((1 + r).prod() ** (ann_factor / n) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(ann_factor))
    rf_ann = float(rf * ann_factor)

    sharpe = (ann_return - rf_ann) / ann_vol if ann_vol > 0 else np.nan

    # Sortino — downside only
    downside = r[r < rf]
    sortino_vol = float(downside.std(ddof=1) * np.sqrt(ann_factor)) if len(downside) > 1 else np.nan
    sortino = (ann_return - rf_ann) / sortino_vol if sortino_vol and sortino_vol > 0 else np.nan

    # Max drawdown
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # VaR & CVaR
    var_95 = float(var_historical(r, 0.95))
    var_99 = float(var_historical(r, 0.99))
    cvar_95 = float(-r[r <= -var_95].mean()) if (r <= -var_95).any() else var_95
    cvar_99 = float(-r[r <= -var_99].mean()) if (r <= -var_99).any() else var_99

    # Skewness & Kurtosis
    skew = float(r.skew())
    kurt = float(r.kurtosis())  # excess kurtosis

    out = {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "var_95_1d": var_95,
        "var_99_1d": var_99,
        "cvar_95_1d": cvar_95,
        "cvar_99_1d": cvar_99,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "n_trading_days": n,
        "win_rate": float((r > 0).mean()),
        "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss": float(r[r < 0].mean()) if (r < 0).any() else 0.0,
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
    }

    if benchmark_returns is not None:
        bench = benchmark_returns.dropna()
        aligned = pd.concat([r.rename("p"), bench.rename("b")], axis=1).dropna()
        if len(aligned) > 20 and aligned["b"].var(ddof=1) > 0:
            cov_mat = np.cov(aligned["p"], aligned["b"], ddof=1)
            beta = cov_mat[0, 1] / cov_mat[1, 1]
            alpha_daily = aligned["p"].mean() - beta * aligned["b"].mean()
            active = aligned["p"] - aligned["b"]
            te = float(active.std(ddof=1) * np.sqrt(ann_factor))
            ir = float(active.mean() * ann_factor / te) if te > 0 else np.nan
            corr = float(aligned.corr().iloc[0, 1])

            out.update({
                "beta": beta,
                "alpha_annualized": alpha_daily * ann_factor,
                "tracking_error": te,
                "information_ratio": ir,
                "correlation_benchmark": corr,
                "r_squared": corr ** 2,
                "active_return": float(active.mean() * ann_factor),
            })

    return out


# ─────────────────────────────────────────────
# Risk contribution per asset
# ─────────────────────────────────────────────

def risk_contribution(weights: pd.Series, cov_annual: pd.DataFrame) -> pd.DataFrame:
    """Marginal risk contribution — used for risk parity diagnostics."""
    w = weights.reindex(cov_annual.index).fillna(0.0)
    w_arr = w.values
    sigma = cov_annual.values
    port_vol = float(np.sqrt(w_arr @ sigma @ w_arr))
    if port_vol <= 0:
        return pd.DataFrame()

    marginal = (sigma @ w_arr) / port_vol
    contribution = w_arr * marginal
    pct = contribution / port_vol

    return pd.DataFrame({
        "security_id": cov_annual.index,
        "weight": w_arr,
        "marginal_risk": marginal,
        "risk_contribution_abs": contribution,
        "risk_contribution_pct": pct,
        "diversification_ratio": port_vol / np.sqrt(w_arr @ np.diag(np.diag(sigma)) @ w_arr),
    })


# ─────────────────────────────────────────────
# GARCH volatility forecasting
# ─────────────────────────────────────────────

def fit_egarch(returns: pd.Series, horizon: int = 21) -> dict:
    """
    Fit EGARCH(1,1) model and forecast volatility.
    EGARCH captures asymmetry (leverage effect) — more realistic than GARCH.
    """
    try:
        from arch import arch_model
        r_pct = returns.dropna() * 100  # arch works in percentage returns
        model = arch_model(r_pct, vol="EGARCH", p=1, q=1, dist="t")
        result = model.fit(disp="off", show_warning=False)
        forecast = result.forecast(horizon=horizon, reindex=False)
        vol_forecast = np.sqrt(forecast.variance.values[-1]) / 100  # back to decimal

        return {
            "model": "EGARCH(1,1)-t",
            "current_vol_annualized": float(result.conditional_volatility.iloc[-1] / 100 * np.sqrt(252)),
            "forecast_vol_1d": float(vol_forecast[0]),
            "forecast_vol_annualized": float(np.mean(vol_forecast) * np.sqrt(252)),
            "aic": float(result.aic),
            "log_likelihood": float(result.loglikelihood),
        }
    except Exception as e:
        logger.warning(f"EGARCH failed: {e}")
        vol = float(returns.dropna().std(ddof=1) * np.sqrt(252))
        return {"model": "historical_vol", "forecast_vol_annualized": vol}


# ─────────────────────────────────────────────
# Historical stress scenarios
# ─────────────────────────────────────────────

STRESS_SCENARIOS = {
    "GFC_2008": {
        "description": "Grande crise financière 2008 (Lehman Brothers)",
        "shocks": {
            "Financials": -0.55, "Technology": -0.45, "Industrials": -0.40,
            "Communication Services": -0.35, "Healthcare": -0.20,
            "Consumer Staples": -0.15, "Energy": -0.50,
        },
        "duration_months": 12,
    },
    "COVID_2020": {
        "description": "Crash COVID mars 2020",
        "shocks": {
            "Financials": -0.40, "Technology": -0.25, "Industrials": -0.45,
            "Communication Services": -0.30, "Healthcare": -0.10,
            "Consumer Staples": -0.15, "Energy": -0.60,
            "Global Equity ETF": -0.35,
        },
        "duration_months": 1,
    },
    "RATES_SHOCK_2022": {
        "description": "Choc taux 2022 — hausse 400bps",
        "shocks": {
            "Technology": -0.40, "Financials": +0.05,
            "Real Estate": -0.35, "Utilities": -0.20,
            "Consumer Discretionary": -0.30, "Healthcare": -0.15,
            "Global Equity ETF": -0.25,
        },
        "duration_months": 12,
    },
    "EURO_CRISIS_2011": {
        "description": "Crise dettes souveraines eurozone 2011",
        "shocks": {
            "Financials": -0.35, "Industrials": -0.25,
            "Technology": -0.20, "Global Equity ETF": -0.20,
        },
        "duration_months": 8,
    },
    "CUSTOM_TECH_MELTDOWN": {
        "description": "Scénario bulle tech (-40% tech, flat autres)",
        "shocks": {"Technology": -0.40},
        "duration_months": 6,
    },
}


def run_stress_test(
    positions: pd.DataFrame,
    scenario_name: str | None = None,
    custom_shocks: dict[str, float] | None = None,
    shock_level: str = "sector",  # 'sector' or 'security'
) -> pd.DataFrame:
    """
    Apply stress scenario to portfolio.
    shock_level: 'sector' applies shocks by sector, 'security' by security_id.
    """
    out = positions.copy()

    if scenario_name and scenario_name in STRESS_SCENARIOS:
        scenario = STRESS_SCENARIOS[scenario_name]
        shocks = scenario["shocks"]
    elif custom_shocks:
        shocks = custom_shocks
    else:
        raise ValueError("Provide scenario_name or custom_shocks")

    key_col = "sector" if shock_level == "sector" else "security_id"
    out["shock_pct"] = out[key_col].map(shocks).fillna(0.0)
    out["shocked_price"] = out["current_price_eur"] * (1 + out["shock_pct"])
    out["shocked_value"] = out["quantity"] * out["shocked_price"]
    out["value_change"] = out["shocked_value"] - out["market_value_eur"]
    out["pct_change_portfolio"] = out["value_change"] / out["market_value_eur"].sum()

    return out


def all_stress_tests(positions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all predefined stress scenarios."""
    results = {}
    for name in STRESS_SCENARIOS:
        try:
            results[name] = run_stress_test(positions, scenario_name=name)
        except Exception as e:
            logger.warning(f"Stress {name}: {e}")
    return results


# ─────────────────────────────────────────────
# Concentration metrics
# ─────────────────────────────────────────────

def concentration_metrics(weights: pd.Series) -> dict:
    w = weights.fillna(0.0)
    hhi = float((w ** 2).sum())
    eff_n = 1 / hhi if hhi > 0 else np.nan
    top3 = float(w.nlargest(3).sum())
    top5 = float(w.nlargest(5).sum())
    gini = _gini(w.values)
    return {
        "hhi": hhi,
        "effective_n": eff_n,
        "top3_weight": top3,
        "top5_weight": top5,
        "gini_coefficient": gini,
    }


def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr)
    n = len(arr)
    if n == 0:
        return 0.0
    cumsum = np.cumsum(arr)
    return float((2 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * cumsum[-1]) / (n * cumsum[-1]))
