"""
scripts/update_data.py
Télécharge toutes les données de marché et met à jour la DB.
Usage: python scripts/update_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from loguru import logger
import click

from src.data.database import init_schema, upsert_prices, upsert_transactions
from src.data.market_data import download_yahoo, download_fred
from src.utils.config import cfg


@click.command()
@click.option("--period", default="5y", help="Période historique Yahoo (ex: 5y, 10y)")
@click.option("--fred/--no-fred", default=True, help="Télécharger données macro FRED")
@click.option("--init-db/--no-init-db", default=True, help="Initialiser le schéma DB")
def main(period: str, fred: bool, init_db: bool):
    """Mise à jour complète des données marché."""
    logger.info("=== Mise à jour données ===")

    c = cfg()

    # Init DB
    if init_db:
        init_schema()
        logger.info("DB initialisée")

    # Load security master
    sm_path = Path(__file__).resolve().parents[1] / "data" / "security_master.csv"
    if not sm_path.exists():
        logger.error(f"security_master.csv introuvable: {sm_path}")
        sys.exit(1)

    sm = pd.read_csv(sm_path)
    tickers = [t for t in sm["yahoo_ticker"].dropna().tolist() if str(t).strip()]
    logger.info(f"Tickers à télécharger: {tickers}")

    # Download Yahoo Finance
    prices = download_yahoo(tickers, period=period)
    if not prices.empty:
        n = upsert_prices(prices)
        logger.success(f"✓ {n} prix sauvegardés (Yahoo Finance)")
    else:
        logger.warning("Aucune donnée Yahoo Finance reçue")

    # Download FRED macro (si clé API dispo)
    if fred:
        macro_df = download_fred()
        if not macro_df.empty:
            from src.data.database import get_conn
            with get_conn() as con:
                con.register("_macro_tmp", macro_df)
                con.execute("INSERT OR REPLACE INTO macro_data SELECT * FROM _macro_tmp")
            logger.success(f"✓ {len(macro_df)} observations macro FRED")

    # Import transactions CSV → DB
    tx_path = Path(__file__).resolve().parents[1] / "data" / "transactions_actions.csv"
    if tx_path.exists():
        tx = pd.read_csv(tx_path)
        tx["id"] = (tx["trade_date"].astype(str) + "_" + tx["security_id"] + "_" +
                    tx.index.astype(str))
        tx["trade_datetime"] = pd.to_datetime(tx["trade_date"])
        tx["account_type"] = "PEA"
        tx["price_eur"] = tx["price_executed_eur"] if "price_executed_eur" in tx.columns else tx.get("nav_eur", 0.0)

        # Colonnes exactes du schéma DB — on ignore le reste
        DB_COLS = [
            "id", "trade_date", "trade_datetime", "security_id", "asset_name",
            "isin", "yahoo_ticker", "asset_class", "sector", "side", "quantity",
            "price_eur", "gross_amount_eur", "commission_eur", "fees_eur",
            "net_cash_eur", "source_file", "account_type",
        ]
        for col in DB_COLS:
            if col not in tx.columns:
                tx[col] = None
        tx_clean = tx[DB_COLS].copy()
        upsert_transactions(tx_clean)
        logger.success(f"✓ {len(tx_clean)} transactions importées")

    logger.success("=== Mise à jour terminée ===")


if __name__ == "__main__":
    main()
