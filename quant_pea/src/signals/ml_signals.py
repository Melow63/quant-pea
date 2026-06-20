"""
ML signal generation — feature engineering + XGBoost + walk-forward CV.
Features: technical (momentum, vol, reversal), cross-sectional ranks, macro.
Target: forward 21-day return sign (classification) ou return (regression).
"""
from __future__ import annotations

from typing import Optional
import warnings

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def compute_features(
    prices: pd.DataFrame,
    macro: Optional[pd.DataFrame] = None,
    lag_days: int = 1,
) -> pd.DataFrame:
    """
    Build feature matrix from price history.
    Returns DataFrame indexed by (date, ticker).
    """
    rets = prices.pct_change()
    log_rets = np.log(prices / prices.shift(1))

    features_list = []

    for ticker in prices.columns:
        p = prices[ticker]
        r = rets[ticker]
        lr = log_rets[ticker]

        feat = pd.DataFrame(index=prices.index)
        feat["ticker"] = ticker

        # ── Momentum features ──────────────────────
        for window in [5, 10, 21, 63, 126, 252]:
            feat[f"ret_{window}d"] = r.rolling(window).sum()
            feat[f"logret_{window}d"] = lr.rolling(window).sum()

        # 12-1 momentum (skip 1 month)
        feat["mom_12_1"] = (p.shift(21) / p.shift(252)) - 1

        # ── Volatility features ────────────────────
        for window in [10, 21, 63]:
            feat[f"vol_{window}d"] = r.rolling(window).std() * np.sqrt(252)

        feat["vol_ratio"] = feat["vol_10d"] / feat["vol_63d"]
        feat["vol_trend"] = feat["vol_21d"] / feat["vol_63d"]

        # ── Reversal ──────────────────────────────
        feat["reversal_1w"] = -r.rolling(5).sum()
        feat["reversal_1m"] = -r.rolling(21).sum()

        # ── Technical ─────────────────────────────
        sma20 = p.rolling(20).mean()
        sma50 = p.rolling(50).mean()
        sma200 = p.rolling(200).mean()
        feat["sma_ratio_20_50"] = sma20 / sma50 - 1
        feat["sma_ratio_50_200"] = sma50 / sma200 - 1
        feat["price_to_52w_high"] = p / p.rolling(252).max()
        feat["price_to_52w_low"] = p / p.rolling(252).min()

        # RSI
        feat["rsi_14"] = _compute_rsi(r, 14)

        # Bollinger band position
        bb_mean = p.rolling(20).mean()
        bb_std = p.rolling(20).std()
        feat["bb_position"] = (p - bb_mean) / (2 * bb_std + 1e-8)

        # ── Drawdown ───────────────────────────────
        rolling_max = p.rolling(252).max()
        feat["drawdown_from_peak"] = p / rolling_max - 1

        features_list.append(feat)

    all_features = pd.concat(features_list)
    all_features = all_features.set_index("ticker", append=True)
    all_features.index.names = ["date", "ticker"]

    # ── Cross-sectional ranks (important for factor investing) ──
    date_groups = all_features.groupby(level="date")
    for col in [c for c in all_features.columns if c.startswith("ret_") or c.startswith("vol_")]:
        all_features[f"rank_{col}"] = date_groups[col].rank(pct=True)

    # ── Macro features (if available) ─────────────────────────
    if macro is not None and not macro.empty:
        macro_wide = macro.pivot(columns="series_id", values="value")
        macro_wide.index = pd.to_datetime(macro_wide.index)
        macro_wide = macro_wide.ffill()

        for series in macro_wide.columns:
            col = f"macro_{series.lower()}"
            macro_series = macro_wide[series]
            macro_ret = macro_series.pct_change(21)  # monthly change
            all_features[col] = macro_ret.reindex(all_features.index.get_level_values("date")).values

    return all_features.shift(lag_days)  # lag to avoid lookahead


def _compute_rsi(returns: pd.Series, window: int = 14) -> pd.Series:
    delta = returns
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ema_up = up.ewm(span=window, adjust=False).mean()
    ema_down = down.ewm(span=window, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-8)
    return 100 - 100 / (1 + rs)


# ─────────────────────────────────────────────
# Target construction
# ─────────────────────────────────────────────

def compute_targets(prices: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """
    Forward returns (t to t+horizon).
    Returns long-format DataFrame indexed by (date, ticker).
    """
    fwd_returns = prices.pct_change(horizon).shift(-horizon)
    rows = []
    for ticker in fwd_returns.columns:
        df = fwd_returns[[ticker]].copy()
        df.columns = ["fwd_return"]
        df["ticker"] = ticker
        df["fwd_sign"] = (df["fwd_return"] > 0).astype(int)  # classification target
        rows.append(df)
    combined = pd.concat(rows).set_index("ticker", append=True)
    combined.index.names = ["date", "ticker"]
    return combined


# ─────────────────────────────────────────────
# XGBoost model training
# ─────────────────────────────────────────────

def train_signal_model(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    model_type: str = "classification",  # 'classification' or 'regression'
    train_end: Optional[str] = None,
) -> dict:
    """
    Train XGBoost signal model with walk-forward cross-validation.
    Returns: model, feature importances, CV score.
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, mean_squared_error
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("xgboost/sklearn not installed")
        return {}

    # Align features and targets
    combined = features.join(targets, how="inner").dropna()
    if len(combined) < 100:
        logger.warning("Insufficient data for ML training")
        return {}

    feature_cols = [c for c in features.columns if c != "ticker"]
    target_col = "fwd_sign" if model_type == "classification" else "fwd_return"

    X = combined[feature_cols].values
    y = combined[target_col].values
    dates = combined.index.get_level_values("date")

    # Walk-forward splits (no random — time series!)
    n = len(X)
    min_train = max(n // 3, 200)
    train_idx = np.arange(min_train)
    test_idx = np.arange(min_train, n)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train = y[train_idx]
    y_test = y[test_idx]

    if model_type == "classification":
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=10,   # regularization — crucial for finance
            gamma=0.1,
            reg_lambda=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred)
        metric_name = "AUC-ROC"
    else:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        score = float(np.corrcoef(y_test, y_pred)[0, 1])
        metric_name = "IC (Pearson corr)"

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    logger.info(f"ML model trained — {metric_name}: {score:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "score": score,
        "metric": metric_name,
        "feature_importance": importance,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


def predict_signals(
    model_dict: dict,
    features: pd.DataFrame,
    top_date: Optional[str] = None,
) -> pd.Series:
    """
    Generate current signals for all tickers.
    Returns Series indexed by ticker with signal score.
    """
    if not model_dict or "model" not in model_dict:
        return pd.Series(dtype=float)

    model = model_dict["model"]
    scaler = model_dict["scaler"]
    feature_cols = model_dict["feature_cols"]

    # Get latest features
    if top_date:
        latest = features.xs(top_date, level="date", drop_level=False)
    else:
        latest_date = features.index.get_level_values("date").max()
        latest = features.xs(latest_date, level="date", drop_level=False)

    available_cols = [c for c in feature_cols if c in latest.columns]
    if not available_cols:
        return pd.Series(dtype=float)

    X = scaler.transform(latest[available_cols].fillna(0.0))

    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1]
        else:
            scores = model.predict(X)
    except Exception as e:
        logger.warning(f"Signal prediction failed: {e}")
        return pd.Series(dtype=float)

    tickers = latest.index.get_level_values("ticker")
    return pd.Series(scores, index=tickers, name="ml_signal").sort_values(ascending=False)
