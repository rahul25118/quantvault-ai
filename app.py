import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from quant_engine import (
    engineer_quant_features,
    fetch_live_macro_news,
    fetch_live_market_data,
    run_econometric_and_causal_suite,
    train_and_evaluate_conformal_engine,
)

# Start with sidebar collapsed so the dashboard takes 100% full screen width, with a '>' toggle button at top-left
st.set_page_config(
    page_title="QuantVault AI | Autonomous Risk Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to use full screen width cleanly and prevent metric text truncation
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        white-space: normal !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("QuantVault AI — Autonomous Risk Intelligence")
st.markdown(
    "Live multi-asset econometrics (`GARCH(1,1)-t`), Orthogonalized Double ML Causal Inference, "
    "Gaussian Precision Contagion Network, and **Adaptive Conformal Inference (ACI)**."
)

st.sidebar.header("Engine Parameters")
lookback_period = st.sidebar.selectbox("Historical Calibration Window", ["5y", "3y", "2y"], index=0)
aci_gamma = st.sidebar.slider("ACI Online Learning Rate (gamma)", 0.01, 0.15, 0.05, 0.01)
if st.sidebar.button("Refresh Live Market & Macro Data"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_full_pipeline(period: str, gamma: float):
    df_clean = fetch_live_market_data(period=period)
    feat_df, model_df, garch_fit = engineer_quant_features(df_clean)
    econ_res = run_econometric_and_causal_suite(model_df, garch_fit)
    ml_res = train_and_evaluate_conformal_engine(feat_df, model_df, gamma=gamma)
    news = fetch_live_macro_news(limit=12)
    return df_clean, model_df, econ_res, ml_res, news

with st.spinner("Ingesting live market APIs & running econometric/conformal models..."):
    df_clean, model_df, econ_res, ml_res, news_items = load_full_pipeline(lookback_period, aci_gamma)

brief = ml_res["live_brief"]
metrics = ml_res["metrics"]
causal = econ_res["causal_ate"]
garch_s = econ_res["garch_summary"]

st.subheader(f"Live Autonomous Risk Brief (Market Close: {brief['as_of_date']})")
is_elevated = (brief["tail_risk_prob"] >= brief["alert_thresh_pct"]) or (brief["latest_vix"] > causal["vix_threshold"])

if is_elevated:
    st.error(
        f"**ELEVATED TAIL-RISK REGIME:** Tail-Risk Probability is **{brief['tail_risk_prob']}%** "
        f"(Alert Threshold: {brief['alert_thresh_pct']}%). Defensive hedging / position sizing reduction advised."
    )
else:
    st.success(
        f"**NORMAL VOLATILITY REGIME:** Tail-Risk Probability is **{brief['tail_risk_prob']}%** "
        f"(Alert Threshold: {brief['alert_thresh_pct']}%). Standard risk allocation supported."
    )

# Clean 3 + 2 responsive card layout so nothing ever gets cut off with '...'
r1_c1, r1_c2, r1_c3 = st.columns(3)
r1_c1.metric("India VIX Level", f"{brief['latest_vix']}", f"Causal Threshold: {causal['vix_threshold']}")
r1_c2.metric("GARCH(1,1)-t Volatility", f"{brief['latest_garch_vol']}%", f"Persistence: {garch_s['persistence']}")
r1_c3.metric("Next-Day |Nifty| Forecast", f"{brief['pred_abs_ret']}%", f"OOS RMSE: {metrics['oos_rmse']}%")

r2_c1, r2_c2 = st.columns(2)
r2_c1.metric("95% ACI Risk Bound", f"[0%, {brief['aci_upper_95']}%]", f"OOS Coverage: {metrics['aci_cov_95']}%")
r2_c2.metric("Tail-Risk Shock Probability", f"{brief['tail_risk_prob']}%", f"OOS AUC: {metrics['oos_auc']}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Adaptive Conformal Volatility",
    "2. Econometrics & Causal ATE",
    "3. Systemic Contagion Graph",
    "4. Live Equity & Macro Headlines"
])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        ax1.plot(ml_res["dates_test"], ml_res["y_reg_test"], color="#1f77b4", alpha=0.85, label="Actual |Nifty Return| (%)")
        ax1.plot(ml_res["dates_test"], ml_res["test_reg_preds"], color="#d62728", linestyle="--", label="Point Forecast")
        ax1.fill_between(
            ml_res["dates_test"], ml_res["aci_lower_95"], ml_res["aci_upper_95"],
            color="#ff9896", alpha=0.35, label=f"95% Adaptive Conformal Band (Cov: {metrics['aci_cov_95']}%)"
        )
        ax1.set_title("Out-of-Sample Volatility & 95% Adaptive Conformal Prediction Interval", fontweight="bold")
        ax1.set_ylabel("Daily Absolute Return (%)")
        ax1.legend(loc="upper right")
        st.pyplot(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        fpr, tpr, _ = roc_curve(ml_res["y_clf_test"], ml_res["test_clf_probs"])
        ax2.plot(fpr, tpr, color="#2ca02c", linewidth=2, label=f"Tail-Risk RF (AUC = {metrics['oos_auc']})")
        ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
        ax2.set_title("Out-of-Sample Tail-Risk ROC Curve", fontweight="bold")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    st.info(
        f"**Non-Stationarity Recovery Audit:** Static Split Conformal coverage degraded to **{metrics['static_cov_95']}%** "
        f"under regime shifts, whereas online **Adaptive Conformal Inference (ACI)** restored empirical coverage to **{metrics['aci_cov_95']}%**."
    )

with tab2:
    st.markdown("### Orthogonalized Double ML Causal Impact")
    st.write(
        f"Controlling for crude oil, USD/INR, US 10Y yields, realized volatility, and banking beta, "
        f"entering an **Elevated Fear Regime (IndiaVIX > {causal['vix_threshold']})** causally increases next-day "
        f"tail-risk probability by **+{causal['ate_pct']}%** (SE: `{causal['se_pct']}%`, `t = {causal['t_stat']}`, `p = {causal['p_value']}`)."
    )
    st.markdown("### GARCH(1,1) Student-t Parameters")
    st.json(garch_s)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**ADF & KPSS Stationarity Diagnostics**")
        st.dataframe(econ_res["stationarity_df"], use_container_width=True)
    with c_right:
        st.markdown("**Granger Causality Lead-Lag Tests**")
        st.dataframe(econ_res["granger_df"], use_container_width=True)

with tab3:
    col_g1, col_g2 = st.columns([1.4, 1])
    with col_g1:
        G = econ_res["graph"]
        strength = econ_res["node_strength"]
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        pos = nx.circular_layout(G)
        edge_colors = ["#d62728" if d["weight"] < 0 else "#1f77b4" for _, _, d in G.edges(data=True)]
        edge_widths = [abs(d["weight"]) * 6 for _, _, d in G.edges(data=True)]
        node_sizes = [650 + strength[n] * 900 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=node_sizes, node_color="#ffbb78", edgecolors="black")
        nx.draw_networkx_edges(G, pos, ax=ax3, width=edge_widths, edge_color=edge_colors, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax3, font_size=9.5, font_weight="bold")
        ax3.set_title("Systemic Contagion Network (Precision Matrix Partial Correlations)", fontweight="bold")
        ax3.axis("off")
        st.pyplot(fig3)

    with col_g2:
        st.markdown("**Systemic Hub Contagion Strength**")
        hub_df = pd.DataFrame(
            sorted(strength.items(), key=lambda x: x[1], reverse=True),
            columns=["Asset / Macro Node", "Conditional Contagion Strength"]
        )
        st.dataframe(hub_df, use_container_width=True)

with tab4:
    st.markdown("### Real-Time Indian Equity & Macro Headlines (Last 7 Days)")
    if news_items:
        for item in news_items:
            st.markdown(f"- **[{item['title']}]({item['link']})**")
            st.caption(f"Published: {item['published']}")
    else:
        st.warning("Live RSS news feed temporarily unavailable; econometric models remain active.")
