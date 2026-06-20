"""
Accounting engine — FIFO lot tracking, TWR, MWR, fiscal français PEA.
Niveau: prime brokerage internal systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid

import numpy as np
import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────
# Lot tracking
# ─────────────────────────────────────────────

@dataclass
class Lot:
    lot_id: str
    trade_datetime: pd.Timestamp
    security_id: str
    quantity_original: float
    quantity_remaining: float
    unit_cost_net: float        # coût net par unité (cash sorti / qty)
    unit_cost_gross: float      # prix d'exécution brut
    commission_unit: float      # commission par unité
    source_index: int

    @property
    def cost_basis_remaining(self) -> float:
        return self.quantity_remaining * self.unit_cost_net

    @property
    def holding_days(self) -> Optional[int]:
        return None  # set at close time


@dataclass
class ClosedTrade:
    security_id: str
    lot_id: str
    buy_datetime: pd.Timestamp
    sell_datetime: pd.Timestamp
    quantity: float
    cost_basis_net: float
    proceeds_net: float
    realized_pnl: float
    holding_days: int
    pea_abattement_eligible: bool = True  # PEA > 5 ans = exonéré IR

    @property
    def realized_pnl_after_tax(self) -> float:
        """Prélèvements sociaux 17.2% sur PEA (IR exonéré après 5 ans)."""
        if self.realized_pnl > 0:
            return self.realized_pnl * (1 - 0.172)
        return self.realized_pnl  # moins-values pas taxées


def build_fifo_ledger(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full FIFO lot book.
    Returns: (open_lots_df, closed_trades_df)
    """
    tx = (
        transactions
        .copy()
        .sort_values(["trade_datetime", "security_id"])
        .reset_index(drop=True)
    )

    lot_book: Dict[str, List[Lot]] = {}
    closed_trades: List[ClosedTrade] = []

    for idx, row in tx.iterrows():
        sec = str(row["security_id"])
        qty = abs(float(row["quantity"]))
        side = str(row["side"]).upper()
        net_cash = float(row["net_cash_eur"])
        gross = float(row.get("gross_amount_eur", net_cash))
        commission = float(row.get("commission_eur", 0.0)) + float(row.get("fees_eur", 0.0))

        if sec not in lot_book:
            lot_book[sec] = []

        if side == "BUY":
            lot = Lot(
                lot_id=str(uuid.uuid4())[:8],
                trade_datetime=pd.to_datetime(row["trade_datetime"]),
                security_id=sec,
                quantity_original=qty,
                quantity_remaining=qty,
                unit_cost_net=net_cash / qty if qty else 0.0,
                unit_cost_gross=gross / qty if qty else 0.0,
                commission_unit=commission / qty if qty else 0.0,
                source_index=idx,
            )
            lot_book[sec].append(lot)

        elif side == "SELL":
            qty_to_close = qty
            sell_unit_proceeds = net_cash / qty if qty else 0.0  # net proceeds per unit

            while qty_to_close > 1e-9:
                if not lot_book.get(sec):
                    logger.error(f"FIFO: Sell without open lot for {sec} at {row['trade_datetime']}")
                    break

                lot = lot_book[sec][0]
                close_qty = min(qty_to_close, lot.quantity_remaining)
                hold_days = (
                    pd.to_datetime(row["trade_datetime"]) - lot.trade_datetime
                ).days

                pnl = close_qty * (sell_unit_proceeds - lot.unit_cost_net)

                closed_trades.append(ClosedTrade(
                    security_id=sec,
                    lot_id=lot.lot_id,
                    buy_datetime=lot.trade_datetime,
                    sell_datetime=pd.to_datetime(row["trade_datetime"]),
                    quantity=close_qty,
                    cost_basis_net=close_qty * lot.unit_cost_net,
                    proceeds_net=close_qty * sell_unit_proceeds,
                    realized_pnl=pnl,
                    holding_days=hold_days,
                    pea_abattement_eligible=hold_days >= 0,  # PEA toujours exonéré IR
                ))

                lot.quantity_remaining -= close_qty
                qty_to_close -= close_qty
                if lot.quantity_remaining <= 1e-9:
                    lot_book[sec].pop(0)
        else:
            logger.warning(f"Unknown side: {side} for {sec}")

    # Open lots summary
    open_rows = []
    for sec, lots in lot_book.items():
        for lot in lots:
            if lot.quantity_remaining > 1e-9:
                open_rows.append({
                    "lot_id": lot.lot_id,
                    "security_id": sec,
                    "open_date": lot.trade_datetime,
                    "quantity_remaining": lot.quantity_remaining,
                    "unit_cost_net": lot.unit_cost_net,
                    "unit_cost_gross": lot.unit_cost_gross,
                    "cost_basis_net": lot.cost_basis_remaining,
                })

    open_df = pd.DataFrame(open_rows)
    closed_df = pd.DataFrame([
        {
            "security_id": t.security_id,
            "lot_id": t.lot_id,
            "buy_datetime": t.buy_datetime,
            "sell_datetime": t.sell_datetime,
            "quantity": t.quantity,
            "cost_basis_net": t.cost_basis_net,
            "proceeds_net": t.proceeds_net,
            "realized_pnl": t.realized_pnl,
            "realized_pnl_after_tax": t.realized_pnl_after_tax,
            "holding_days": t.holding_days,
        }
        for t in closed_trades
    ])

    return open_df, closed_df


# ─────────────────────────────────────────────
# Position aggregation from open lots
# ─────────────────────────────────────────────

def positions_from_lots(open_lots: pd.DataFrame, security_master: pd.DataFrame) -> pd.DataFrame:
    if open_lots.empty:
        return pd.DataFrame()

    agg = (
        open_lots.groupby("security_id")
        .apply(lambda g: pd.Series({
            "quantity": g["quantity_remaining"].sum(),
            "avg_price_net": (g["quantity_remaining"] * g["unit_cost_net"]).sum() / g["quantity_remaining"].sum(),
            "avg_price_gross": (g["quantity_remaining"] * g["unit_cost_gross"]).sum() / g["quantity_remaining"].sum(),
            "total_cost_net": g["cost_basis_net"].sum(),
            "n_lots": len(g),
            "oldest_lot_date": g["open_date"].min(),
        }))
        .reset_index()
    )
    agg = agg.merge(
        security_master[["security_id", "asset_name", "yahoo_ticker", "sector", "asset_class"]],
        on="security_id", how="left",
    )
    return agg


# ─────────────────────────────────────────────
# Time-Weighted Return (GIPS compliant)
# ─────────────────────────────────────────────

def compute_twr(portfolio_values: pd.Series, external_flows: pd.Series) -> pd.Series:
    """
    Modified Dietz / sub-period linking — GIPS compliant TWR.
    portfolio_values: NAV daily series
    external_flows: cash in (+) / out (-) per day
    Returns cumulative TWR series.
    """
    nav = portfolio_values.copy().fillna(method="ffill")
    flows = external_flows.reindex(nav.index, fill_value=0.0)
    prev_nav = nav.shift(1)

    # Sub-period return: (V_end - V_start - CF) / V_start
    sub_returns = pd.Series(index=nav.index, dtype=float)
    for dt in nav.index[1:]:
        v_end = nav.loc[dt]
        v_start = prev_nav.loc[dt]
        cf = flows.loc[dt]
        if v_start > 0:
            sub_returns.loc[dt] = (v_end - v_start - cf) / v_start
        else:
            sub_returns.loc[dt] = 0.0

    sub_returns.iloc[0] = 0.0
    cum_twr = (1 + sub_returns).cumprod() - 1
    return cum_twr


def compute_mwr(portfolio_values: pd.Series, external_flows: pd.Series) -> float:
    """
    Money-Weighted Return (= IRR) — mesure la vraie performance investisseur.
    Uses Newton-Raphson on cash flow NPV.
    """
    from scipy.optimize import brentq

    flows = external_flows.copy()
    nav = portfolio_values.copy()

    dates = flows.index.union(nav.index).sort_values()
    all_flows = flows.reindex(dates, fill_value=0.0)

    # First flow = initial investment (positive = cash out from investor POV)
    # Last value = terminal NAV (negative = cash in from investor POV)
    first_date = dates[0]
    last_date = dates[-1]
    terminal_value = -nav.iloc[-1]

    cash_flows = []
    for dt in dates:
        cf = all_flows.loc[dt]
        if cf != 0:
            t = (dt - first_date).days / 365.25
            cash_flows.append((t, cf))
    t_terminal = (last_date - first_date).days / 365.25
    cash_flows.append((t_terminal, terminal_value))

    def npv(r):
        return sum(cf / (1 + r) ** t for t, cf in cash_flows)

    try:
        mwr = brentq(npv, -0.99, 10.0, maxiter=500)
    except Exception:
        mwr = np.nan

    return mwr


# ─────────────────────────────────────────────
# Brinson-Hood-Beebower Attribution
# ─────────────────────────────────────────────

def brinson_attribution(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    Single-period Brinson-Hood-Beebower attribution by sector.
    Returns DataFrame with allocation, selection, interaction effects.
    """
    sectors = sector_map.unique()
    rows = []

    for sector in sectors:
        mask = sector_map == sector
        secs = sector_map[mask].index

        wp = portfolio_weights.reindex(secs).fillna(0.0).sum()
        wb = benchmark_weights.reindex(secs).fillna(0.0).sum()
        rp = (
            (portfolio_weights.reindex(secs).fillna(0.0) * portfolio_returns.reindex(secs).fillna(0.0)).sum()
            / wp if wp > 0 else 0.0
        )
        rb = (
            (benchmark_weights.reindex(secs).fillna(0.0) * benchmark_returns.reindex(secs).fillna(0.0)).sum()
            / wb if wb > 0 else 0.0
        )
        rb_total = (benchmark_weights * benchmark_returns.reindex(benchmark_weights.index, fill_value=0.0)).sum()

        allocation_effect = (wp - wb) * (rb - rb_total)
        selection_effect = wb * (rp - rb)
        interaction_effect = (wp - wb) * (rp - rb)
        total_effect = allocation_effect + selection_effect + interaction_effect

        rows.append({
            "sector": sector,
            "portfolio_weight": wp,
            "benchmark_weight": wb,
            "portfolio_return": rp,
            "benchmark_return": rb,
            "allocation_effect": allocation_effect,
            "selection_effect": selection_effect,
            "interaction_effect": interaction_effect,
            "total_effect": total_effect,
        })

    return pd.DataFrame(rows).sort_values("total_effect", ascending=False)
