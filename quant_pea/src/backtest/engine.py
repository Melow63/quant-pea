"""
Backtesting engine — event-driven, walk-forward, realistic costs.
Niveau: comparable à QuantConnect/Zipline mais simplifié et transparent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import warnings

import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Transaction cost model
# ─────────────────────────────────────────────

@dataclass
class CostModel:
    """Realistic transaction costs for French PEA (Boursobank / Saxo)."""
    commission_bps: float = 15.0     # 0.15% — réaliste Boursobank
    min_commission: float = 0.50     # minimum €0.50 par trade
    slippage_bps: float = 5.0        # 0.05% bid-ask spread
    market_impact_bps: float = 2.0   # impact de marché (petite taille → faible)
    stamp_duty: float = 0.0          # PEA: pas de TTF sur ETF/fonds

    def total_cost_bps(self) -> float:
        return self.commission_bps + self.slippage_bps + self.market_impact_bps

    def compute_cost(self, trade_value: float) -> float:
        commission = max(trade_value * self.commission_bps / 10_000, self.min_commission)
        slippage = trade_value * self.slippage_bps / 10_000
        impact = trade_value * self.market_impact_bps / 10_000
        return commission + slippage + impact


# ─────────────────────────────────────────────
# Portfolio state
# ─────────────────────────────────────────────

@dataclass
class Portfolio:
    cash: float = 0.0
    holdings: dict[str, float] = field(default_factory=dict)  # ticker → quantity
    trade_history: list[dict] = field(default_factory=list)
    cost_model: CostModel = field(default_factory=CostModel)

    def nav(self, prices: pd.Series) -> float:
        equity = sum(self.holdings.get(t, 0) * prices.get(t, 0) for t in self.holdings)
        return self.cash + equity

    def weights(self, prices: pd.Series) -> pd.Series:
        total = self.nav(prices)
        if total <= 0:
            return pd.Series(dtype=float)
        w = {t: (self.holdings.get(t, 0) * prices.get(t, 0)) / total
             for t in self.holdings}
        return pd.Series(w)

    def rebalance(self, target_weights: pd.Series, prices: pd.Series, date) -> float:
        """Rebalance to target weights. Returns total transaction costs."""
        total_nav = self.nav(prices)
        if total_nav <= 0:
            return 0.0

        total_cost = 0.0
        current_weights = self.weights(prices)

        for ticker in target_weights.index:
            target_w = float(target_weights.get(ticker, 0.0))
            current_w = float(current_weights.get(ticker, 0.0))
            delta_w = target_w - current_w
            if abs(delta_w) < 1e-4:  # skip tiny rebalances
                continue

            trade_value = abs(delta_w) * total_nav
            cost = self.cost_model.compute_cost(trade_value)
            total_cost += cost

            price = float(prices.get(ticker, 0))
            if price <= 0:
                continue

            qty_delta = (delta_w * total_nav) / price
            self.holdings[ticker] = self.holdings.get(ticker, 0.0) + qty_delta
            self.cash -= (qty_delta * price + cost * np.sign(delta_w))

            self.trade_history.append({
                "date": date,
                "ticker": ticker,
                "quantity": qty_delta,
                "price": price,
                "cost": cost,
                "delta_weight": delta_w,
            })

        return total_cost


# ─────────────────────────────────────────────
# Backtest engine
# ─────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, int], pd.Series],
    initial_capital: float = 100_000.0,
    rebalance_freq: str = "ME",       # 'ME'=month-end, 'QE'=quarter-end, 'W-FRI'=weekly
    cost_model: Optional[CostModel] = None,
    benchmark_ticker: Optional[str] = None,
    warmup_periods: int = 252,
) -> dict:
    """
    Event-driven backtest.
    strategy_fn(prices_up_to_t, t_index) → target_weights Series (index=tickers, sum=1)
    Returns dict with nav_series, trades, metrics, etc.
    """
    if cost_model is None:
        cost_model = CostModel()

    prices = prices.ffill().dropna(how="all")
    dates = prices.index
    rebal_dates = pd.date_range(dates[warmup_periods], dates[-1], freq=rebalance_freq)

    portfolio = Portfolio(cash=initial_capital, cost_model=cost_model)
    nav_series = pd.Series(index=dates[warmup_periods:], dtype=float)
    turnover_series = pd.Series(index=dates[warmup_periods:], dtype=float)
    weights_history = []

    logger.info(f"Starting backtest: {dates[warmup_periods].date()} → {dates[-1].date()}")

    for i, date in enumerate(dates[warmup_periods:], start=warmup_periods):
        current_prices = prices.iloc[i]
        nav_series.loc[date] = portfolio.nav(current_prices)

        if date in rebal_dates or i == warmup_periods:
            prices_so_far = prices.iloc[: i + 1]
            try:
                target_w = strategy_fn(prices_so_far, i)
                target_w = target_w.dropna().clip(lower=0)
                if target_w.sum() > 0:
                    target_w /= target_w.sum()

                if portfolio.nav(current_prices) > 0:
                    cost = portfolio.rebalance(target_w, current_prices, date)
                    turnover = sum(abs(target_w.get(t, 0) - float(portfolio.weights(current_prices).get(t, 0)))
                                   for t in set(list(target_w.index) + list(portfolio.holdings)))
                    turnover_series.loc[date] = turnover / 2
                else:
                    # First allocation
                    for ticker, w in target_w.items():
                        price = float(current_prices.get(ticker, 0))
                        if price > 0:
                            qty = (w * initial_capital) / price
                            portfolio.holdings[ticker] = qty
                            portfolio.cash -= w * initial_capital

                weights_history.append({"date": date, **target_w.to_dict()})

            except Exception as e:
                logger.warning(f"Strategy error at {date}: {e}")

    nav = nav_series.dropna()
    returns = nav.pct_change().dropna()

    result = {
        "nav": nav,
        "returns": returns,
        "initial_capital": initial_capital,
        "final_nav": float(nav.iloc[-1]) if len(nav) > 0 else initial_capital,
        "total_return": float(nav.iloc[-1] / initial_capital - 1) if len(nav) > 0 else 0.0,
        "trades": pd.DataFrame(portfolio.trade_history),
        "weights_history": pd.DataFrame(weights_history).set_index("date") if weights_history else pd.DataFrame(),
        "turnover_series": turnover_series.dropna(),
        "avg_annual_turnover": float(turnover_series.dropna().mean() * (252 / _freq_to_periods(rebalance_freq))),
    }

    # Benchmark comparison
    if benchmark_ticker and benchmark_ticker in prices.columns:
        bench_prices = prices[benchmark_ticker].iloc[warmup_periods:]
        bench_returns = bench_prices.pct_change().dropna()
        bench_nav = initial_capital * (1 + bench_returns).cumprod()
        result["benchmark_nav"] = bench_nav
        result["benchmark_returns"] = bench_returns

    return result


def _freq_to_periods(freq: str) -> int:
    mapping = {"ME": 21, "QE": 63, "W-FRI": 5, "W": 5, "D": 1}
    for k, v in mapping.items():
        if k in freq:
            return v
    return 21


# ─────────────────────────────────────────────
# Walk-forward validation
# ─────────────────────────────────────────────

def walk_forward_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable,
    train_months: int = 36,
    test_months: int = 6,
    cost_model: Optional[CostModel] = None,
    benchmark_ticker: Optional[str] = None,
) -> dict:
    """
    Walk-forward out-of-sample validation.
    Trains on rolling window, tests on next period.
    Prevents lookahead bias.
    """
    from dateutil.relativedelta import relativedelta

    dates = prices.index
    start = dates[0]
    end = dates[-1]

    all_nav = []
    all_returns = []
    periods = []

    train_start = start
    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_end = train_end + relativedelta(months=test_months)

        if test_end > end:
            break

        train_prices = prices[prices.index <= train_end]
        test_prices = prices[(prices.index > train_end) & (prices.index <= test_end)]

        if len(train_prices) < 60 or len(test_prices) < 5:
            train_start += relativedelta(months=test_months)
            continue

        try:
            # Get strategy weights from training data
            target_w = strategy_fn(train_prices, len(train_prices) - 1)
            target_w = target_w.dropna().clip(lower=0)
            if target_w.sum() > 0:
                target_w /= target_w.sum()

            # Apply weights to test period (buy-and-hold between rebalances)
            available = [t for t in target_w.index if t in test_prices.columns]
            if not available:
                train_start += relativedelta(months=test_months)
                continue

            w_test = target_w.reindex(available).fillna(0.0)
            w_test /= w_test.sum()

            test_rets = test_prices[available].pct_change().dropna()
            port_rets = (test_rets * w_test.values).sum(axis=1)

            # Apply one-time rebalancing cost
            if cost_model:
                cost_drag = cost_model.total_cost_bps() / 10_000
                port_rets.iloc[0] -= cost_drag

            all_returns.append(port_rets)
            periods.append({"train_start": train_start, "train_end": train_end,
                            "test_start": train_end, "test_end": test_end,
                            "weights": w_test.to_dict()})

        except Exception as e:
            logger.warning(f"Walk-forward period {train_start}→{test_end}: {e}")

        train_start += relativedelta(months=test_months)

    if not all_returns:
        return {"error": "No valid walk-forward periods"}

    combined_returns = pd.concat(all_returns).sort_index()
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep="last")]
    combined_nav = (1 + combined_returns).cumprod() * 100

    return {
        "returns": combined_returns,
        "nav": combined_nav,
        "periods": pd.DataFrame(periods),
        "n_periods": len(periods),
    }


# ─────────────────────────────────────────────
# Pre-built strategies for PEA
# ─────────────────────────────────────────────

def strategy_momentum(prices: pd.DataFrame, t: int, top_n: int = 5,
                      lookback: int = 252, skip: int = 21) -> pd.Series:
    """Momentum strategy: buy top-N by 12-1 month return."""
    if t < lookback + skip:
        return pd.Series(dtype=float)
    past = prices.iloc[t - lookback - skip]
    recent = prices.iloc[t - skip]
    mom = (recent / past - 1).dropna()
    top = mom.nlargest(top_n).index
    w = pd.Series(1.0 / len(top), index=top)
    return w


def strategy_min_vol(prices: pd.DataFrame, t: int, window: int = 126,
                     max_weight: float = 0.35) -> pd.Series:
    """Minimum volatility strategy."""
    from ..optimization.portfolio import min_cvar_weights, sample_cov
    if t < window + 20:
        return pd.Series(dtype=float)
    rets = prices.iloc[max(0, t - window): t + 1].pct_change().dropna()
    if len(rets) < 30:
        return pd.Series(dtype=float)
    return min_cvar_weights(rets, max_weight=max_weight, min_weight=0.0)


def strategy_equal_weight(prices: pd.DataFrame, t: int) -> pd.Series:
    """Naive 1/N equal weight baseline."""
    n = len(prices.columns)
    return pd.Series(1 / n, index=prices.columns)
