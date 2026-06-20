"""
scripts/run_backtest.py
Lance les stratégies de backtest sur l'univers PEA.
Usage: python scripts/run_backtest.py --strategy momentum
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import click
from loguru import logger

from src.data.market_data import close_pivot
from src.backtest.engine import (
    run_backtest, walk_forward_backtest, CostModel,
    strategy_momentum, strategy_min_vol, strategy_equal_weight,
)
from src.risk.models import compute_risk_metrics


STRATEGIES = {
    "momentum": strategy_momentum,
    "min_vol": strategy_min_vol,
    "equal_weight": strategy_equal_weight,
}


@click.command()
@click.option("--strategy", default="momentum", type=click.Choice(list(STRATEGIES.keys())))
@click.option("--walk-forward/--simple", default=True, help="Walk-forward ou backtest simple")
@click.option("--initial-capital", default=10_000.0, help="Capital initial (€)")
@click.option("--rebalance", default="ME", help="Fréquence rebalancement (ME, QE, W-FRI)")
def main(strategy: str, walk_forward: bool, initial_capital: float, rebalance: str):
    """Lancer un backtest sur l'univers PEA."""
    logger.info(f"=== Backtest {strategy} ===")

    # Load prices
    prices = close_pivot()
    if prices.empty:
        logger.error("Pas de données de prix. Lancez scripts/update_data.py d'abord.")
        sys.exit(1)

    sm_path = Path(__file__).resolve().parents[1] / "data" / "security_master.csv"
    sm = pd.read_csv(sm_path)
    invest = sm[sm.get("include_analytics", "Y") == "Y"]["yahoo_ticker"].dropna().tolist()
    invest = [t for t in invest if t in prices.columns and t != "CW8.PA"]
    prices_invest = prices[invest].dropna(how="all")

    cost_model = CostModel(commission_bps=15, slippage_bps=5)
    strat_fn = STRATEGIES[strategy]

    if walk_forward:
        logger.info("Mode walk-forward (out-of-sample)")
        result = walk_forward_backtest(
            prices_invest, strat_fn,
            train_months=36, test_months=6,
            cost_model=cost_model,
        )
        if "error" in result:
            logger.error(result["error"])
            sys.exit(1)
        rets = result["returns"]
    else:
        logger.info("Mode backtest simple")
        result = run_backtest(
            prices_invest, strat_fn,
            initial_capital=initial_capital,
            rebalance_freq=rebalance,
            cost_model=cost_model,
            benchmark_ticker="CW8.PA" if "CW8.PA" in prices.columns else None,
        )
        rets = result["returns"]

    # Print tearsheet
    metrics = compute_risk_metrics(rets)
    logger.info("\n" + "=" * 50)
    logger.info(f"STRATÉGIE: {strategy.upper()}")
    logger.info("=" * 50)
    logger.info(f"Rendement annualisé : {metrics.get('annual_return', 0):.2%}")
    logger.info(f"Volatilité          : {metrics.get('annual_volatility', 0):.2%}")
    logger.info(f"Sharpe Ratio        : {metrics.get('sharpe', 0):.3f}")
    logger.info(f"Sortino Ratio       : {metrics.get('sortino', 0):.3f}")
    logger.info(f"Max Drawdown        : {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"Calmar Ratio        : {metrics.get('calmar', 0):.3f}")
    logger.info(f"VaR 95% (1j)        : {metrics.get('var_95_1d', 0):.2%}")
    logger.info(f"Win Rate            : {metrics.get('win_rate', 0):.1%}")
    logger.info("=" * 50)

    # Save results
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / f"backtest_{strategy}.parquet"
    if not result.get("nav", pd.Series()).empty:
        result["nav"].to_frame("nav").to_parquet(out_path)
        logger.success(f"Résultats sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
