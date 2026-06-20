"""
Dashboard Streamlit — Portfolio System v2
Interface institutionnelle complète.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import streamlit as st
except ImportError:
    raise SystemExit("pip install streamlit")

from src.pipeline import full_analytics, get_positions_snapshot
from src.risk.models import STRESS_SCENARIOS, run_stress_test
from src.utils.config import cfg

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Quant PEA — Portfolio System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetric { background: #0e1117; border: 1px solid #262730; border-radius: 8px; padding: 12px; }
    .stMetric label { font-size: 0.75rem; color: #9b9b9b; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
    .positive { color: #00d26a !important; }
    .negative { color: #ff4b4b !important; }
    section[data-testid="stSidebar"] { background: #0a0c10; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()

    run_ml = st.toggle("🤖 ML Signals", value=False, help="XGBoost alpha model (lent)")
    run_nlp = st.toggle("📰 NLP Sentiment", value=False, help="FinBERT scraping news")

    st.divider()
    st.subheader("Stress test")
    scenario = st.selectbox(
        "Scénario",
        options=list(STRESS_SCENARIOS.keys()),
        format_func=lambda x: STRESS_SCENARIOS[x]["description"],
    )
    custom_shock = st.slider("Choc tech custom (%)", -70, 0, -20, 1) / 100

    st.divider()
    st.subheader("Optimizer")
    max_weight = st.slider("Poids max par ligne (%)", 5, 50, 35, 5) / 100
    rf_rate = st.slider("Taux sans risque (%)", 0.0, 8.0, 3.5, 0.1) / 100


def auto_h(df: pd.DataFrame, base: int = 35) -> int:
    return min(max(base * (len(df) + 1), 200), 800)


def color_pnl(val: float) -> str:
    return "🟢" if val > 0 else "🔴" if val < 0 else "⚪"


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Chargement des analytics...")
def load_analytics(ml: bool, nlp: bool):
    return full_analytics(run_ml=ml, run_nlp=nlp)


data = load_analytics(run_ml, run_nlp)
positions = data.get("positions", pd.DataFrame())

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tabs = st.tabs([
    "📊 Overview",
    "📋 Holdings",
    "📈 Performance",
    "⚠️ Risk",
    "🧮 Optimizer",
    "🔬 Factors",
    "🤖 Signals",
    "💥 Stress",
    "📰 Sentiment",
    "🗂️ Reporting",
])

tab_overview, tab_holdings, tab_perf, tab_risk, tab_opt, tab_factors, tab_signals, tab_stress, tab_sentiment, tab_report = tabs


# ────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ────────────────────────────────────────────────────────────

with tab_overview:
    if positions.empty:
        st.info("Aucune position trouvée. Vérifiez vos données de transactions.")
        st.stop()

    total_value = positions["market_value_eur"].sum()
    total_pnl = positions["unrealized_pnl_eur"].sum()
    total_invested = positions.get("total_cost_net", positions.get("net_invested_eur", pd.Series([0]))).sum()
    total_return = total_pnl / total_invested if total_invested > 0 else 0.0

    risk = data.get("risk_metrics", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💼 Valeur totale", f"{total_value:,.0f} €")
    c2.metric("📈 P&L non-réalisé", f"{total_pnl:+,.0f} €", f"{total_return:+.1%}")
    c3.metric("📉 Max Drawdown", f"{risk.get('max_drawdown', 0):.1%}" if risk else "—")
    c4.metric("⚡ Sharpe Ratio", f"{risk.get('sharpe', 0):.2f}" if risk else "—")
    c5.metric("📊 Vol annualisée", f"{risk.get('annual_volatility', 0):.1%}" if risk else "—")
    c6.metric("🏆 Lignes", f"{(positions['quantity'] > 0).sum()}")

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_tree = px.treemap(
            positions,
            path=[px.Constant("Portefeuille"), "sector", "asset_name"],
            values="market_value_eur",
            color="unrealized_return",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Allocation par secteur et ligne",
        )
        fig_tree.update_traces(textinfo="label+percent root+value")
        st.plotly_chart(fig_tree, use_container_width=True)

    with col2:
        # Sector donut
        sector_agg = positions.groupby("sector")["market_value_eur"].sum().sort_values(ascending=False)
        fig_donut = px.pie(
            values=sector_agg.values,
            names=sector_agg.index,
            hole=0.5,
            title="Répartition sectorielle",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Concentration metrics
    conc = data.get("concentration", {})
    if conc:
        st.subheader("Métriques de concentration")
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("HHI", f"{conc.get('hhi', 0):.3f}", help="Herfindahl-Hirschman Index (0=diversifié, 1=concentré)")
        cc2.metric("N effectif", f"{conc.get('effective_n', 0):.1f}", help="Nombre effectif de positions")
        cc3.metric("Top 3 poids", f"{conc.get('top3_weight', 0):.1%}")
        cc4.metric("Gini", f"{conc.get('gini_coefficient', 0):.3f}", help="Coefficient de Gini (0=égal, 1=concentré)")


# ────────────────────────────────────────────────────────────
# TAB 2: HOLDINGS
# ────────────────────────────────────────────────────────────

with tab_holdings:
    if positions.empty:
        st.info("Aucune position.")
    else:
        cols_show = [c for c in [
            "security_id", "asset_name", "sector", "quantity",
            "avg_price_net_eur", "avg_price_gross", "current_price_eur",
            "market_value_eur", "unrealized_pnl_eur", "unrealized_return", "weight",
        ] if c in positions.columns]

        fmt = {
            "avg_price_net_eur": "{:.2f} €",
            "avg_price_gross": "{:.2f} €",
            "current_price_eur": "{:.2f} €",
            "market_value_eur": "{:,.0f} €",
            "unrealized_pnl_eur": "{:+,.0f} €",
            "unrealized_return": "{:+.2%}",
            "weight": "{:.1%}",
        }

        st.dataframe(
            positions[cols_show].style.format(fmt, na_rep="—"),
            use_container_width=True,
            height=auto_h(positions),
        )

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            fig_bar = px.bar(
                positions.sort_values("unrealized_pnl_eur"),
                x="asset_name", y="unrealized_pnl_eur",
                color="unrealized_pnl_eur",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="P&L non-réalisé par ligne",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_ret = px.bar(
                positions.sort_values("unrealized_return"),
                x="asset_name", y="unrealized_return",
                color="unrealized_return",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                title="Rendement non-réalisé (%)",
            )
            fig_ret.update_layout(yaxis_tickformat=".1%")
            st.plotly_chart(fig_ret, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 3: PERFORMANCE
# ────────────────────────────────────────────────────────────

with tab_perf:
    port_nav = data.get("portfolio_nav")
    if port_nav is None or port_nav.empty:
        st.info("Pas d'historique de prix. Lancez `scripts/update_data.py` d'abord.")
    else:
        twr = data.get("portfolio_twr", pd.Series(dtype=float))
        mwr = data.get("portfolio_mwr", np.nan)
        bench_ticker = cfg()["portfolio"]["benchmark_ticker"]
        prices = data.get("price_history", pd.DataFrame())

        rm = data.get("risk_metrics", {})
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Rendement annualisé", f"{rm.get('annual_return', 0):.1%}")
        r2.metric("TWR cumulé", f"{float(twr.iloc[-1]):.1%}" if not twr.empty else "—")
        r3.metric("MWR (IRR investisseur)", f"{mwr:.1%}" if not np.isnan(mwr) else "—")
        r4.metric("Meilleur jour", f"{rm.get('best_day', 0):.1%}")

        # NAV chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_nav.index, y=port_nav, name="Portefeuille (€)", line=dict(color="#00d26a", width=2)))
        if bench_ticker in prices.columns:
            bench_prices = prices[bench_ticker].reindex(port_nav.index).ffill()
            bench_scaled = bench_prices / bench_prices.iloc[0] * port_nav.iloc[0]
            fig.add_trace(go.Scatter(x=bench_scaled.index, y=bench_scaled, name=f"Benchmark ({bench_ticker})",
                                     line=dict(color="#7a7a7a", dash="dash")))
        fig.update_layout(title="Valeur du portefeuille vs benchmark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # TWR chart
        fig2 = go.Figure()
        if not twr.empty:
            fig2.add_trace(go.Scatter(x=twr.index, y=twr * 100, name="TWR portefeuille (%)",
                                      line=dict(color="#00d26a")))
        if bench_ticker in prices.columns:
            bench_rets = prices[bench_ticker].pct_change()
            bench_twr = (1 + bench_rets).cumprod() - 1
            bench_twr = bench_twr.reindex(twr.index if not twr.empty else prices.index)
            fig2.add_trace(go.Scatter(x=bench_twr.index, y=bench_twr * 100,
                                      name=f"Benchmark TWR (%)", line=dict(color="#7a7a7a", dash="dash")))
        fig2.update_layout(title="Time-Weighted Return cumulé (%)", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

        # Drawdown
        if not twr.empty:
            wealth = (1 + twr.pct_change().fillna(0))
            cum = wealth.cumprod()
            dd = cum / cum.cummax() - 1
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy",
                                      fillcolor="rgba(255,75,75,0.2)", line=dict(color="#ff4b4b"),
                                      name="Drawdown (%)"))
            fig3.update_layout(title="Drawdown historique (%)")
            st.plotly_chart(fig3, use_container_width=True)

        # Return contribution
        if "risk_contribution" in data and not data["risk_contribution"].empty:
            st.subheader("Contribution au risque par actif")
            rc = data["risk_contribution"].copy()
            st.dataframe(
                rc.style.format({
                    "weight": "{:.1%}",
                    "risk_contribution_abs": "{:.4f}",
                    "risk_contribution_pct": "{:.1%}",
                }),
                use_container_width=True,
                height=auto_h(rc),
            )


# ────────────────────────────────────────────────────────────
# TAB 4: RISK
# ────────────────────────────────────────────────────────────

with tab_risk:
    rm = data.get("risk_metrics", {})
    if not rm:
        st.info("Lancez update_data.py pour les analytics de risque.")
    else:
        st.subheader("Métriques de risque complètes")
        rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
        rc1.metric("VaR 95% (1j)", f"{rm.get('var_95_1d', 0):.2%}")
        rc2.metric("VaR 99% (1j)", f"{rm.get('var_99_1d', 0):.2%}")
        rc3.metric("CVaR 95% (1j)", f"{rm.get('cvar_95_1d', 0):.2%}")
        rc4.metric("Beta", f"{rm.get('beta', 0):.2f}" if rm.get("beta") else "—")
        rc5.metric("Alpha annualisé", f"{rm.get('alpha_annualized', 0):.2%}" if rm.get("alpha_annualized") else "—")
        rc6.metric("Info Ratio", f"{rm.get('information_ratio', 0):.2f}" if rm.get("information_ratio") else "—")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Métriques étendues")
            extended = {
                "Rendement annualisé": f"{rm.get('annual_return', 0):.2%}",
                "Volatilité annualisée": f"{rm.get('annual_volatility', 0):.2%}",
                "Sharpe Ratio": f"{rm.get('sharpe', 0):.3f}",
                "Sortino Ratio": f"{rm.get('sortino', 0):.3f}",
                "Calmar Ratio": f"{rm.get('calmar', 0):.3f}",
                "Max Drawdown": f"{rm.get('max_drawdown', 0):.2%}",
                "Skewness": f"{rm.get('skewness', 0):.3f}",
                "Kurtosis excess": f"{rm.get('excess_kurtosis', 0):.3f}",
                "Win Rate": f"{rm.get('win_rate', 0):.1%}",
                "Avg Win / Avg Loss": f"{rm.get('avg_win', 0) / abs(rm.get('avg_loss', -1)):.2f}",
                "Tracking Error": f"{rm.get('tracking_error', 0):.2%}" if rm.get("tracking_error") else "—",
                "R²": f"{rm.get('r_squared', 0):.3f}" if rm.get("r_squared") else "—",
            }
            st.dataframe(pd.DataFrame.from_dict(extended, orient="index", columns=["Valeur"]))

        with col2:
            # Monte Carlo VaR distribution
            mc = data.get("var_monte_carlo")
            if mc and "simulated_pnl" in mc:
                pnl = mc["simulated_pnl"]
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(x=pnl * 100, nbinsx=80, name="Simulations",
                                              marker_color="#4a9eff", opacity=0.7))
                fig_mc.add_vline(x=-mc["var"] * 100, line_dash="dash", line_color="orange",
                                 annotation_text=f"VaR 95%: {mc['var']:.1%}")
                fig_mc.add_vline(x=-mc["cvar"] * 100, line_dash="dash", line_color="red",
                                 annotation_text=f"CVaR 95%: {mc['cvar']:.1%}")
                fig_mc.update_layout(title=f"Distribution P&L Monte Carlo ({mc['n_simulations']:,} simulations)",
                                     xaxis_title="Return (%)", yaxis_title="Fréquence")
                st.plotly_chart(fig_mc, use_container_width=True)

        # GARCH vol forecast
        garch = data.get("garch", {})
        if garch:
            st.subheader("Prévision de volatilité EGARCH")
            g1, g2, g3 = st.columns(3)
            g1.metric("Vol actuelle (annualisée)", f"{garch.get('current_vol_annualized', 0):.1%}")
            g2.metric("Vol prévue 1j", f"{garch.get('forecast_vol_1d', 0):.3%}")
            g3.metric("Vol prévue (annualisée)", f"{garch.get('forecast_vol_annualized', 0):.1%}")


# ────────────────────────────────────────────────────────────
# TAB 5: OPTIMIZER
# ────────────────────────────────────────────────────────────

with tab_opt:
    opt = data.get("optimizer")
    if not opt:
        st.info("Pas assez de données pour l'optimiseur. Lancez update_data.py.")
    else:
        method_names = {
            "hrp": "HRP (Hierarchical Risk Parity)",
            "min_cvar": "Min-CVaR",
            "erc": "Equal Risk Contribution",
            "black_litterman": "Black-Litterman",
            "max_sharpe": "Max Sharpe",
            "min_variance": "Min Variance",
        }

        # Efficient frontier
        if "efficient_frontier" in opt and not opt["efficient_frontier"].empty:
            ef = opt["efficient_frontier"]
            fig_ef = go.Figure()
            fig_ef.add_trace(go.Scatter(x=ef["volatility"] * 100, y=ef["target_return"] * 100,
                                        mode="lines+markers", name="Frontière efficiente",
                                        line=dict(color="#4a9eff")))
            # Mark current portfolio
            rm = data.get("risk_metrics", {})
            if rm:
                fig_ef.add_trace(go.Scatter(
                    x=[rm.get("annual_volatility", 0) * 100],
                    y=[rm.get("annual_return", 0) * 100],
                    mode="markers", name="Portefeuille actuel",
                    marker=dict(size=14, color="#ff4b4b", symbol="star"),
                ))
            fig_ef.update_layout(title="Frontière efficiente (Ledoit-Wolf cov)",
                                 xaxis_title="Volatilité (%)", yaxis_title="Rendement (%)")
            st.plotly_chart(fig_ef, use_container_width=True)

        # Weight comparison table
        weight_cols = {}
        for key, name in method_names.items():
            if key in opt and isinstance(opt[key], pd.Series):
                weight_cols[name] = opt[key]

        if weight_cols:
            weight_df = pd.DataFrame(weight_cols).fillna(0.0)
            st.subheader("Comparaison des poids optimaux")
            fig_comp = px.bar(
                weight_df.reset_index().rename(columns={"index": "Actif"}),
                x="Actif",
                y=list(weight_cols.keys()),
                barmode="group",
                title="Poids par méthode d'optimisation",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(weight_df.style.format("{:.1%}"), use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 6: FACTOR ATTRIBUTION
# ────────────────────────────────────────────────────────────

with tab_factors:
    fa = data.get("factor_attribution")
    if not fa or isinstance(fa, dict) and "error" in fa:
        st.info("Attribution factorielle non disponible. Données Fama-French téléchargées au premier lancement.")
    elif isinstance(fa, dict):
        st.subheader("Attribution factorielle — Fama-French 5 + Momentum")
        f1, f2, f3 = st.columns(3)
        f1.metric("Alpha annualisé", f"{fa.get('alpha_annualized', 0):.2%}",
                  help=f"t-stat: {fa.get('alpha_tstat', 0):.2f}")
        f2.metric("R²", f"{fa.get('r_squared', 0):.1%}")
        f3.metric("Vol résiduelle", f"{fa.get('residual_vol_annualized', 0):.2%}")

        betas = fa.get("betas", {})
        t_stats = fa.get("t_stats", {})
        contribs = fa.get("factor_return_contributions", {})

        if betas:
            beta_df = pd.DataFrame({
                "Beta": betas,
                "t-stat": t_stats,
                "Contribution return": contribs,
            }).round(4)
            st.dataframe(beta_df, use_container_width=True)

            # Factor contribution chart
            contrib_series = pd.Series(contribs)
            fig_fa = px.bar(
                x=contrib_series.index, y=contrib_series.values * 100,
                title="Décomposition du rendement par facteur (%)",
                color=contrib_series.values,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
            )
            st.plotly_chart(fig_fa, use_container_width=True)

        if "rolling" in fa and isinstance(fa["rolling"], pd.DataFrame):
            st.subheader("Betas glissants (fenêtre 6 mois)")
            roll = fa["rolling"]
            fig_roll = px.line(roll, title="Évolution des factor loadings")
            st.plotly_chart(fig_roll, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 7: ML SIGNALS
# ────────────────────────────────────────────────────────────

with tab_signals:
    if not run_ml:
        st.info("Activez '🤖 ML Signals' dans la sidebar pour lancer le modèle XGBoost.")
    else:
        ml_sig = data.get("ml_signals")
        ml_model = data.get("ml_model", {})

        if ml_sig is None or ml_sig.empty:
            st.warning("Pas assez de données pour le modèle ML.")
        else:
            st.subheader("Signaux alpha XGBoost")
            m1, m2, m3 = st.columns(3)
            m1.metric("Score OOS", f"{ml_model.get('score', 0):.3f}", help=ml_model.get("metric", ""))
            m2.metric("N train", f"{ml_model.get('n_train', 0):,}")
            m3.metric("N test", f"{ml_model.get('n_test', 0):,}")

            sig_df = ml_sig.reset_index()
            sig_df.columns = ["Ticker", "Signal"]
            sig_df["Signal label"] = sig_df["Signal"].apply(lambda x: "🟢 Long" if x > 0.6 else "🔴 Short" if x < 0.4 else "⚪ Neutre")
            st.dataframe(sig_df, use_container_width=True)

            imp = ml_model.get("feature_importance")
            if imp is not None:
                fig_imp = px.bar(
                    x=imp.head(20).values, y=imp.head(20).index,
                    orientation="h", title="Top 20 features XGBoost",
                )
                st.plotly_chart(fig_imp, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 8: STRESS TESTS
# ────────────────────────────────────────────────────────────

with tab_stress:
    if positions.empty:
        st.info("Pas de positions.")
    else:
        all_stress = data.get("stress_tests", {})

        st.subheader(f"Scénario sélectionné: {STRESS_SCENARIOS[scenario]['description']}")
        stressed = run_stress_test(positions, scenario_name=scenario)
        total_shock = stressed["value_change"].sum() if "value_change" in stressed.columns else 0

        s1, s2, s3 = st.columns(3)
        s1.metric("Impact total", f"{total_shock:+,.0f} €")
        s2.metric("Impact % portefeuille", f"{total_shock / positions['market_value_eur'].sum():.1%}")
        s3.metric("Valeur post-choc", f"{positions['market_value_eur'].sum() + total_shock:,.0f} €")

        display_cols = [c for c in ["asset_name", "sector", "market_value_eur", "shock_pct", "shocked_value", "value_change"]
                        if c in stressed.columns]
        st.dataframe(
            stressed[display_cols].sort_values("value_change").style.format({
                "market_value_eur": "{:,.0f} €",
                "shock_pct": "{:.0%}",
                "shocked_value": "{:,.0f} €",
                "value_change": "{:+,.0f} €",
            }),
            use_container_width=True,
        )

        # Custom tech shock
        st.divider()
        st.subheader(f"Choc tech custom: {custom_shock:.0%}")
        custom_result = run_stress_test(positions, custom_shocks={"Technology": custom_shock})
        custom_total = custom_result["value_change"].sum() if "value_change" in custom_result.columns else 0
        st.metric("Impact choc custom", f"{custom_total:+,.0f} €",
                  f"{custom_total / positions['market_value_eur'].sum():.1%}")

        # All scenarios summary
        if all_stress:
            st.divider()
            st.subheader("Résumé tous scénarios")
            summary_rows = []
            for name, df in all_stress.items():
                if "value_change" in df.columns:
                    total = df["value_change"].sum()
                    summary_rows.append({
                        "Scénario": STRESS_SCENARIOS[name]["description"],
                        "Impact (€)": total,
                        "Impact (%)": total / positions["market_value_eur"].sum(),
                    })
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows).sort_values("Impact (€)")
                st.dataframe(summary_df.style.format({"Impact (€)": "{:+,.0f}", "Impact (%)": "{:.1%}"}))


# ────────────────────────────────────────────────────────────
# TAB 9: SENTIMENT
# ────────────────────────────────────────────────────────────

with tab_sentiment:
    if not run_nlp:
        st.info("Activez '📰 NLP Sentiment' dans la sidebar pour lancer l'analyse FinBERT.")
    else:
        sent = data.get("sentiment")
        if sent is None or sent.empty:
            st.warning("Aucune donnée de sentiment.")
        else:
            st.subheader("Sentiment FinBERT par sécurité")
            active = sent[sent["n_articles"] > 0]
            if active.empty:
                st.warning("Aucune news récupérée. Vérifiez votre connexion internet.")
            else:
                fig_sent = px.bar(
                    active.sort_values("sentiment_score"),
                    x="security_id", y="sentiment_score",
                    color="sentiment_label",
                    color_discrete_map={"positive": "#00d26a", "negative": "#ff4b4b", "neutral": "#7a7a7a"},
                    title="Score de sentiment [-1, +1]  (source: Google News)",
                )
                st.plotly_chart(fig_sent, use_container_width=True)

            display_cols = [c for c in ["security_id", "sentiment_score", "sentiment_label",
                                         "n_articles", "confidence", "sample_headline"] if c in sent.columns]
            st.dataframe(
                sent[display_cols].style.format({
                    "sentiment_score": "{:+.3f}",
                    "confidence": "{:.2f}",
                    "n_articles": "{:.0f}",
                }),
                use_container_width=True,
            )


# ────────────────────────────────────────────────────────────
# TAB 10: REPORTING
# ────────────────────────────────────────────────────────────

with tab_report:
    st.subheader("📄 Export & Reporting")
    st.info("Génération de rapports PDF — lancez `scripts/generate_tearsheet.py`")

    rm = data.get("risk_metrics", {})
    if positions.empty or not rm:
        st.warning("Données insuffisantes pour générer un rapport complet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Résumé exécutif")
            exec_data = {
                "Valeur portefeuille": f"{positions['market_value_eur'].sum():,.0f} €",
                "P&L non-réalisé": f"{positions['unrealized_pnl_eur'].sum():+,.0f} €",
                "Rendement annualisé": f"{rm.get('annual_return', 0):.2%}",
                "Volatilité": f"{rm.get('annual_volatility', 0):.2%}",
                "Sharpe Ratio": f"{rm.get('sharpe', 0):.2f}",
                "Max Drawdown": f"{rm.get('max_drawdown', 0):.2%}",
                "VaR 95%": f"{rm.get('var_95_1d', 0):.2%}",
            }
            for k, v in exec_data.items():
                st.text(f"{k}: {v}")

        with col2:
            if not positions.empty:
                csv = positions.to_csv(index=False)
                st.download_button(
                    label="⬇️ Télécharger positions (CSV)",
                    data=csv,
                    file_name="positions_snapshot.csv",
                    mime="text/csv",
                )
