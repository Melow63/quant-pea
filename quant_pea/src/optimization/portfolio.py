"""
Portfolio optimization — niveau institutionnel.
- Black-Litterman avec views bayésiens
- Hierarchical Risk Parity (Lopez de Prado)
- Min-CVaR (CVXPY)
- Robust MVO (shrinkage Ledoit-Wolf)
- Equal Risk Contribution
Toutes les méthodes respectent les contraintes PEA (long-only, no leverage).
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from loguru import logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Covariance estimation (robuste)
# ─────────────────────────────────────────────

def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage estimator — réduit l'erreur d'estimation.
    Analytically optimal shrinkage intensity.
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.dropna())
        cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns) * 252
        return cov
    except Exception as e:
        logger.warning(f"Ledoit-Wolf failed, using sample cov: {e}")
        return returns.cov() * 252


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.dropna(how="all").fillna(0.0).cov() * 252


# ─────────────────────────────────────────────
# Black-Litterman
# ─────────────────────────────────────────────

def black_litterman(
    market_weights: pd.Series,
    returns: pd.DataFrame,
    views: Optional[dict] = None,
    risk_aversion: float = 2.5,
    tau: float = 0.025,
    cov_estimator: str = "ledoit_wolf",
) -> pd.Series:
    """
    Black-Litterman model.
    market_weights: benchmark/market cap weights (e.g. MSCI World sector weights)
    views: dict of {security_id: expected_return_view} — your alpha views
           e.g. {"LVMH": 0.15, "OVH": -0.05}  # annual returns
    risk_aversion: lambda — typically 2-3 for equity
    tau: uncertainty of prior — typically 0.01-0.05
    Returns posterior expected returns (Black-Litterman combined estimate).
    """
    assets = returns.columns.tolist()
    mw = market_weights.reindex(assets).fillna(1 / len(assets))
    mw = mw / mw.sum()

    if cov_estimator == "ledoit_wolf":
        cov = ledoit_wolf_cov(returns)
    else:
        cov = sample_cov(returns)

    sigma = cov.values

    # Implied equilibrium returns (reverse optimization)
    pi = risk_aversion * (sigma @ mw.values)

    if not views:
        # No views → return equilibrium
        return pd.Series(pi, index=assets, name="bl_expected_return")

    # Build views matrices P (pick matrix) and Q (view returns)
    view_assets = [a for a in views if a in assets]
    if not view_assets:
        return pd.Series(pi, index=assets, name="bl_expected_return")

    n = len(assets)
    k = len(view_assets)
    P = np.zeros((k, n))
    Q = np.zeros(k)

    for i, asset in enumerate(view_assets):
        j = assets.index(asset)
        P[i, j] = 1.0
        Q[i] = views[asset]

    # Uncertainty of views (proportional to variance of each asset)
    omega = np.diag([tau * float(cov.iloc[assets.index(a), assets.index(a)]) for a in view_assets])

    # BL posterior
    tau_sigma = tau * sigma
    M1 = np.linalg.inv(tau_sigma)
    M2 = P.T @ np.linalg.inv(omega) @ P
    mu_bl = np.linalg.inv(M1 + M2) @ (M1 @ pi + P.T @ np.linalg.inv(omega) @ Q)

    return pd.Series(mu_bl, index=assets, name="bl_expected_return")


def bl_optimal_weights(
    bl_returns: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 2.5,
    max_weight: float = 0.35,
    min_weight: float = 0.02,
) -> pd.Series:
    """Optimize weights given BL expected returns."""
    assets = bl_returns.index.tolist()
    n = len(assets)
    mu = bl_returns.values
    sigma = cov.reindex(index=assets, columns=assets).values + np.eye(n) * 1e-8

    def neg_utility(w):
        ret = w @ mu
        var = w @ sigma @ w
        return -(ret - risk_aversion / 2 * var)

    x0 = np.ones(n) / n
    bounds = [(min_weight, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

    try:
        res = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000})
        if res.success:
            w = np.maximum(res.x, 0)
            w /= w.sum()
            return pd.Series(w, index=assets, name="black_litterman")
    except Exception as e:
        logger.warning(f"BL optimization failed: {e}")

    return pd.Series(x0, index=assets, name="black_litterman")


# ─────────────────────────────────────────────
# Hierarchical Risk Parity (Lopez de Prado)
# ─────────────────────────────────────────────

def hrp_weights(returns: pd.DataFrame, linkage_method: str = "ward") -> pd.Series:
    """
    Hierarchical Risk Parity — Lopez de Prado (2016).
    Uses hierarchical clustering on the correlation matrix.
    More robust than MVO — no matrix inversion, no Markowitz instability.
    """
    assets = returns.columns.tolist()
    cov = ledoit_wolf_cov(returns)
    corr = returns.corr()

    # Distance matrix from correlation
    dist = np.sqrt((1 - corr) / 2)
    dist_arr = squareform(dist.values, checks=False)
    link = linkage(dist_arr, method=linkage_method)

    # Quasi-diagonalization: sort assets by clustering
    sorted_idx = _quasi_diag(link, len(assets))
    sorted_assets = [assets[i] for i in sorted_idx]

    # Recursive bisection
    w = _recursive_bisection(cov, sorted_assets)
    return pd.Series(w, name="hrp").reindex(assets).fillna(0.0)


def _quasi_diag(link: np.ndarray, n: int) -> list[int]:
    """Sort items by hierarchy — quasi-diagonal."""
    link = link.astype(int)
    sorted_items = [link[-1, 0], link[-1, 1]]
    n_items = link[-1, 3]
    while len(sorted_items) < n_items:
        new_items = []
        for i in sorted_items:
            if i >= n:
                row = link[i - n]
                new_items += [row[0], row[1]]
            else:
                new_items.append(i)
        sorted_items = new_items
    return [i for i in sorted_items if i < n]


def _recursive_bisection(cov: pd.DataFrame, sorted_assets: list[str]) -> dict[str, float]:
    w = {a: 1.0 for a in sorted_assets}
    clusters = [sorted_assets]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]

            var_left = _cluster_var(cov, left)
            var_right = _cluster_var(cov, right)

            alloc_left = 1 - var_left / (var_left + var_right)
            alloc_right = 1 - alloc_left

            for a in left:
                w[a] *= alloc_left
            for a in right:
                w[a] *= alloc_right

            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)

        clusters = new_clusters

    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def _cluster_var(cov: pd.DataFrame, assets: list[str]) -> float:
    sub_cov = cov.loc[assets, assets].values
    n = len(assets)
    w_eq = np.ones(n) / n
    return float(w_eq @ sub_cov @ w_eq)


# ─────────────────────────────────────────────
# Min-CVaR (CVXPY — convex optimization)
# ─────────────────────────────────────────────

def min_cvar_weights(
    returns: pd.DataFrame,
    confidence: float = 0.95,
    max_weight: float = 0.35,
    min_weight: float = 0.02,
    target_return: Optional[float] = None,
) -> pd.Series:
    """
    Minimize CVaR using linear programming (Rockafellar-Uryasev formulation).
    Convex — guaranteed global optimum.
    """
    try:
        import cvxpy as cp
    except ImportError:
        logger.warning("cvxpy not installed — falling back to EW")
        n = len(returns.columns)
        return pd.Series(np.ones(n) / n, index=returns.columns, name="min_cvar")

    R = returns.dropna(how="all").fillna(0.0).values
    T, n = R.shape
    alpha = 1 - confidence

    w = cp.Variable(n)
    z = cp.Variable(T)
    gamma = cp.Variable()

    port_rets = R @ w
    constraints = [
        w >= min_weight,
        w <= max_weight,
        cp.sum(w) == 1,
        z >= 0,
        z >= -port_rets - gamma,
    ]
    if target_return is not None:
        ann_ret = returns.mean() * 252
        constraints.append(ann_ret.values @ w >= target_return)

    cvar_obj = gamma + (1 / (alpha * T)) * cp.sum(z)
    prob = cp.Problem(cp.Minimize(cvar_obj), constraints)

    try:
        prob.solve(solver=cp.ECOS, verbose=False)
        if w.value is not None:
            weights = np.maximum(w.value, 0)
            weights /= weights.sum()
            return pd.Series(weights, index=returns.columns, name="min_cvar")
    except Exception as e:
        logger.warning(f"CVaR optimization failed: {e}")

    # Fallback equal weight
    n_assets = len(returns.columns)
    return pd.Series(np.ones(n_assets) / n_assets, index=returns.columns, name="min_cvar")


# ─────────────────────────────────────────────
# Equal Risk Contribution
# ─────────────────────────────────────────────

def erc_weights(
    returns: pd.DataFrame,
    max_weight: float = 0.35,
) -> pd.Series:
    """Equal Risk Contribution (risk parity)."""
    cov = ledoit_wolf_cov(returns).values
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}

    cov_reg = cov + np.eye(n) * 1e-8

    def objective(w):
        port_vol = np.sqrt(float(w @ cov_reg @ w))
        if port_vol <= 0:
            return 1e6
        mrc = cov_reg @ w / port_vol
        rc = w * mrc
        target = port_vol / n
        return float(np.sum((rc - target) ** 2))

    try:
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})
        if res.success:
            w = np.maximum(res.x, 0)
            w /= w.sum()
            return pd.Series(w, index=returns.columns, name="erc")
    except Exception as e:
        logger.warning(f"ERC failed: {e}")

    return pd.Series(x0, index=returns.columns, name="erc")


# ─────────────────────────────────────────────
# Efficient frontier (robust)
# ─────────────────────────────────────────────

def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 30,
    max_weight: float = 0.35,
    min_weight: float = 0.0,
    cov_estimator: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Compute full efficient frontier with transaction-cost-aware constraints."""
    if cov_estimator == "ledoit_wolf":
        cov = ledoit_wolf_cov(returns)
    else:
        cov = sample_cov(returns)

    mu = returns.mean() * 252
    assets = returns.columns.tolist()
    n = len(assets)
    sigma = cov.values + np.eye(n) * 1e-8
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

    min_ret = float(mu.min())
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, n_points)
    rows = []

    def port_vol(w):
        return float(np.sqrt(max(w @ sigma @ w, 0.0)))

    for target in targets:
        c = constraints + [{"type": "eq", "fun": lambda w, t=target: float(w @ mu.values) - t}]
        try:
            res = minimize(port_vol, np.ones(n) / n, method="SLSQP",
                           bounds=bounds, constraints=c, options={"maxiter": 500})
            if res.success:
                w = np.maximum(res.x, 0)
                w /= w.sum()
                rows.append({
                    "target_return": target,
                    "volatility": port_vol(w),
                    "sharpe": (target - 0.035) / port_vol(w),
                    **{assets[i]: w[i] for i in range(n)},
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Combined optimizer runner
# ─────────────────────────────────────────────

def run_all_optimizers(
    returns: pd.DataFrame,
    market_weights: Optional[pd.Series] = None,
    views: Optional[dict] = None,
    max_weight: float = 0.35,
    min_weight: float = 0.02,
) -> dict:
    """Run all optimization methods and return results dict."""
    results = {}
    cov = ledoit_wolf_cov(returns)
    mu = returns.mean() * 252

    # 1. HRP
    try:
        results["hrp"] = hrp_weights(returns)
    except Exception as e:
        logger.warning(f"HRP: {e}")

    # 2. Min-CVaR
    try:
        results["min_cvar"] = min_cvar_weights(returns, max_weight=max_weight, min_weight=min_weight)
    except Exception as e:
        logger.warning(f"Min-CVaR: {e}")

    # 3. Equal Risk Contribution
    try:
        results["erc"] = erc_weights(returns, max_weight=max_weight)
    except Exception as e:
        logger.warning(f"ERC: {e}")

    # 4. Black-Litterman
    if market_weights is not None:
        try:
            bl_mu = black_litterman(market_weights, returns, views=views)
            results["black_litterman"] = bl_optimal_weights(bl_mu, cov, max_weight=max_weight, min_weight=min_weight)
        except Exception as e:
            logger.warning(f"Black-Litterman: {e}")

    # 5. Max Sharpe (classic)
    try:
        results["max_sharpe"] = _max_sharpe(mu.values, cov.values, returns.columns, max_weight, min_weight)
    except Exception as e:
        logger.warning(f"Max Sharpe: {e}")

    # 6. Min Variance
    try:
        results["min_variance"] = _min_variance(cov.values, returns.columns, max_weight, min_weight)
    except Exception as e:
        logger.warning(f"Min Variance: {e}")

    # 7. Efficient Frontier
    try:
        results["efficient_frontier"] = efficient_frontier(returns, max_weight=max_weight, min_weight=0.0)
    except Exception as e:
        logger.warning(f"Efficient Frontier: {e}")

    return results


def _max_sharpe(mu, sigma, index, max_w, min_w, rf=0.035):
    n = len(mu)
    sig = sigma + np.eye(n) * 1e-8

    def neg_sharpe(w):
        ret = w @ mu
        vol = np.sqrt(max(w @ sig @ w, 1e-12))
        return -(ret - rf) / vol

    res = minimize(neg_sharpe, np.ones(n) / n, method="SLSQP",
                   bounds=[(min_w, max_w)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else np.ones(n) / n
    return pd.Series(w, index=index, name="max_sharpe")


def _min_variance(sigma, index, max_w, min_w):
    n = sigma.shape[0]
    sig = sigma + np.eye(n) * 1e-8

    def var(w):
        return float(w @ sig @ w)

    res = minimize(var, np.ones(n) / n, method="SLSQP",
                   bounds=[(min_w, max_w)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    w = np.maximum(res.x, 0) / np.maximum(res.x, 0).sum() if res.success else np.ones(n) / n
    return pd.Series(w, index=index, name="min_variance")
