# Quant PEA — Institutional-Grade Portfolio Management System

A desk-style portfolio management system built for a French PEA (Plan d'Épargne en Actions), combining institutional-level analytics with a fully automated Python engine and interactive Streamlit dashboard.

**Stack:** Python · DuckDB · Streamlit · Plotly · CVXPY · XGBoost · FinBERT

---

## Overview

Most retail portfolio tools show you a pie chart. This system gives you what an institutional analyst actually uses:

- FIFO lot accounting with TWR/MWR performance measurement
- Factor attribution (Fama-French 5 + Momentum)
- Portfolio optimization (Black-Litterman, HRP, Min-CVaR)
- Risk analytics (VaR, CVaR, EGARCH, stress testing)
- ML alpha signals (XGBoost)
- NLP sentiment analysis (FinBERT)
- 10-tab Streamlit dashboard

All built from scratch. One source of truth: `portfolio_system_master.xlsx`.

---

## Architecture

```
portfolio_system/
├── src/
│   ├── accounting.py      FIFO lots, TWR/MWR, Brinson-Hood-Beebower attribution
│   ├── risk.py            VaR/CVaR, EGARCH, Sharpe/Sortino/Calmar, drawdown
│   ├── optimization.py    Black-Litterman, HRP (Lopez de Prado), Min-CVaR, ERC
│   ├── performance.py     NAV reconstruction, daily position matrix, alpha/beta
│   ├── stress.py          5 historical stress scenarios + custom shock
│   ├── market_data.py     Yahoo Finance + FRED + ECB live data
│   └── pipeline.py        Central orchestrator
├── dashboard/
│   └── app.py             Streamlit — 10 tabs
├── scripts/
│   ├── update_market_data.py
│   └── refresh_workbook.py
├── data/
│   ├── transactions_actions.csv
│   ├── transactions_funds.csv
│   └── security_master.csv
└── portfolio_system_master.xlsx
```

---

## Features

### Analytics

| Module | Techniques |
|--------|-----------|
| **Accounting** | FIFO, TWR (GIPS-compliant), MWR/IRR, Brinson-Hood-Beebower |
| **Risk** | Historical/Parametric/Monte Carlo VaR, CVaR, EGARCH(1,1), stress tests |
| **Factors** | Fama-French 5, Momentum (12-1), alpha/beta/tracking error/IR |
| **Optimizer** | Black-Litterman, HRP, Min-CVaR (CVXPY), ERC, Max-Sharpe |
| **Backtest** | Event-driven, walk-forward OOS, realistic transaction costs |
| **ML Signals** | XGBoost, cross-sectional features, walk-forward CV |
| **NLP** | FinBERT (HuggingFace), French/English financial news sentiment |
| **Stress** | GFC 2008, COVID 2020, 2022 rate shock, Euro crisis, custom |

### Dashboard (10 tabs)

1. **Overview** — NAV, key metrics, treemap, sector donut, HHI concentration
2. **Holdings** — detailed position table, P&L bars, returns
3. **Performance** — NAV vs benchmark, TWR/MWR comparison, drawdown chart
4. **Risk** — Sharpe/Sortino/Calmar + Monte Carlo VaR distribution
5. **Optimizer** — efficient frontier, 6 methods comparison, optimal weights
6. **Factors** — FF5 attribution, rolling betas, factor contribution
7. **Signals** — ML scores, XGBoost feature importance
8. **Stress** — 5 scenarios + interactive custom shock
9. **Sentiment** — FinBERT by security, news scraping
10. **Reporting** — CSV export, executive summary

---

## Setup

```bash
git clone https://github.com/eliottbertin/quant-pea
cd quant-pea
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Add your FRED API key (free at fred.stlouisfed.org)
cp .env.example .env
# Edit .env → FRED_API_KEY=your_key

# Update market data
python scripts/update_market_data.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## References

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal.
- Lopez de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample.* Journal of Portfolio Management.
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model.* Journal of Financial Economics.
- Brinson, G., Hood, L. & Beebower, G. (1986). *Determinants of Portfolio Performance.* Financial Analysts Journal.
- Rockafellar, R. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.* Journal of Risk.
- Nelson, D. (1991). *Conditional Heteroskedasticity in Asset Returns: A New Approach.* Econometrica.
