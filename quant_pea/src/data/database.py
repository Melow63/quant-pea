"""DuckDB persistence layer — single connection, all tables managed here."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from loguru import logger

from ..utils.config import cfg

ROOT = Path(__file__).resolve().parents[3]


def _db_path() -> Path:
    p = ROOT / cfg()["database"]["path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(_db_path()), read_only=read_only)


def init_schema() -> None:
    """Create all tables if they don't exist."""
    with get_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            date        DATE NOT NULL,
            ticker      VARCHAR NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      BIGINT,
            adj_close   DOUBLE,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS security_master (
            security_id     VARCHAR PRIMARY KEY,
            asset_name      VARCHAR,
            isin            VARCHAR,
            yahoo_ticker    VARCHAR,
            asset_class     VARCHAR,
            sector          VARCHAR,
            country         VARCHAR,
            currency        VARCHAR DEFAULT 'EUR',
            pea_eligible    BOOLEAN DEFAULT true,
            include_analytics BOOLEAN DEFAULT true,
            price_source    VARCHAR
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id              VARCHAR PRIMARY KEY,
            trade_date      DATE NOT NULL,
            trade_datetime  TIMESTAMP,
            security_id     VARCHAR NOT NULL,
            asset_name      VARCHAR,
            isin            VARCHAR,
            yahoo_ticker    VARCHAR,
            asset_class     VARCHAR,
            sector          VARCHAR,
            side            VARCHAR NOT NULL,   -- BUY / SELL
            quantity        DOUBLE NOT NULL,
            price_eur       DOUBLE,
            gross_amount_eur DOUBLE,
            commission_eur  DOUBLE DEFAULT 0,
            fees_eur        DOUBLE DEFAULT 0,
            net_cash_eur    DOUBLE NOT NULL,
            source_file     VARCHAR,
            account_type    VARCHAR DEFAULT 'PEA'
        );

        CREATE TABLE IF NOT EXISTS macro_data (
            date        DATE NOT NULL,
            series_id   VARCHAR NOT NULL,
            value       DOUBLE,
            PRIMARY KEY (date, series_id)
        );

        CREATE TABLE IF NOT EXISTS factor_returns (
            date        DATE NOT NULL,
            factor      VARCHAR NOT NULL,
            value       DOUBLE,
            universe    VARCHAR DEFAULT 'europe',
            PRIMARY KEY (date, factor, universe)
        );

        CREATE TABLE IF NOT EXISTS signals (
            date            DATE NOT NULL,
            security_id     VARCHAR NOT NULL,
            signal_type     VARCHAR NOT NULL,
            signal_value    DOUBLE,
            confidence      DOUBLE,
            PRIMARY KEY (date, security_id, signal_type)
        );

        CREATE TABLE IF NOT EXISTS news_sentiment (
            id              VARCHAR PRIMARY KEY,
            date            DATE NOT NULL,
            security_id     VARCHAR,
            source          VARCHAR,
            headline        VARCHAR,
            sentiment_score DOUBLE,
            sentiment_label VARCHAR,
            url             VARCHAR
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            snapshot_date   DATE NOT NULL,
            security_id     VARCHAR NOT NULL,
            quantity        DOUBLE,
            avg_price_net   DOUBLE,
            current_price   DOUBLE,
            market_value_eur DOUBLE,
            unrealized_pnl  DOUBLE,
            weight          DOUBLE,
            PRIMARY KEY (snapshot_date, security_id)
        );
        """)
    logger.info("DuckDB schema initialized")


def upsert_prices(df: pd.DataFrame) -> int:
    """Insert or replace price rows. df must have: date, ticker, close (+ optionally others)."""
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in ["open", "high", "low", "volume", "adj_close"]:
        if col not in df.columns:
            df[col] = None
    df = df[["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]]
    with get_conn() as con:
        con.register("_prices_tmp", df)
        con.execute("""
            INSERT OR REPLACE INTO price_history
            SELECT * FROM _prices_tmp
        """)
    logger.debug(f"Upserted {len(df)} price rows")
    return len(df)


def load_prices(tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM price_history"
    clauses = []
    if tickers:
        t_list = ", ".join(f"'{t}'" for t in tickers)
        clauses.append(f"ticker IN ({t_list})")
    if start:
        clauses.append(f"date >= '{start}'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY date, ticker"
    with get_conn(read_only=True) as con:
        return con.execute(q).df()


def load_transactions() -> pd.DataFrame:
    with get_conn(read_only=True) as con:
        return con.execute("SELECT * FROM transactions ORDER BY trade_datetime").df()


def upsert_transactions(df: pd.DataFrame) -> int:
    with get_conn() as con:
        con.register("_tx_tmp", df)
        con.execute("INSERT OR REPLACE INTO transactions SELECT * FROM _tx_tmp")
    return len(df)


def load_macro(series: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM macro_data"
    clauses = []
    if series:
        s_list = ", ".join(f"'{s}'" for s in series)
        clauses.append(f"series_id IN ({s_list})")
    if start:
        clauses.append(f"date >= '{start}'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    with get_conn(read_only=True) as con:
        return con.execute(q).df()
