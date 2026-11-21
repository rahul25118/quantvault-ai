import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from quant_engine import (
    compute_portfolio_risk_allocation,
    engineer_quant_features,
    fetch_live_macro_news,
    fetch_live_market_data,
    run_econometric_and_causal_suite,
    score_news_sentiment,
    send_telegram_risk_alert,
    simulate_what_if_stress,
    train_and_evaluate_conformal_engine,
)

st.set_page_config(
    page_title="QuantVault AI | Institutional Risk Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

css_style = "<style>.block-container { padding: 1.4rem; max-width: 100%; } [data-testid='stMetricValue'] { font-size: 1.6rem !important; white-space: normal !important; } [data-testid='stMetricLabel'] { font-size: 0.88rem !important; white-space: normal !important; }</style>"
st.markdown(css_style, unsafe_allow_html=True)

st.title("QuantVault AI — Autonomous Risk & Decision Intelligence")
st.markdown(
    "Live multi-asset econometrics (`GARCH(1,1)-t`), Orthogonalized Double ML Causal Inference, "
    "Gaussian Precision Contagion Network, **Adaptive Conformal Inference (ACI)**, and **Multimodal NLP Risk Fusion**."
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
    raw_news = fetch_live_macro_news(limit=12)
    sentiment_res = score_news_sentiment(raw_news)
    return df_clean, model_df, econ_res, ml_res, sentiment_res

with st.spinner("Ingesting live market APIs, NLP news sentiment & running conformal models..."):
    df_clean, model_df, econ_res, ml_res, sent_res = load_full_pipeline(lookback_period, aci_gamma)

brief = ml_res["live_brief"]
metrics = ml_res["metrics"]
causal = econ_res["causal_ate"]
garch_s = econ_res["garch_summary"]
default_alloc = compute_portfolio_risk_allocation(500000.0, 1.0, brief, sent_res["news_fear_index"])

st.subheader(f"Live Autonomous Risk Brief (Market Close: {brief['as_of_date']})")
is_elevated = (default_alloc["multimodal_risk_pct"] >= brief["alert_thresh_pct"]) or (brief["latest_vix"] > causal["vix_threshold"])

if is_elevated:
    st.error(
        f"**ELEVATED TAIL-RISK REGIME:** Quant Tail-Risk is **{brief['tail_risk_prob']}%** | "
        f"Multimodal Risk Score (Quant + NLP News): **{default_alloc['multimodal_risk_pct']}%** "
        f"(Threshold: {brief['alert_thresh_pct']}%). Suggested Equity Allocation: **{default_alloc['recommended_equity_pct']}%**."
    )
else:
    st.success(
        f"**NORMAL VOLATILITY REGIME:** Quant Tail-Risk is **{brief['tail_risk_prob']}%** | "
        f"Multimodal Risk Score: **{default_alloc['multimodal_risk_pct']}%** "
        f"(Threshold: {brief['alert_thresh_pct']}%). Suggested Equity Allocation: **{default_alloc['recommended_equity_pct']}%**."
    )

r1_c1, r1_c2, r1_c3 = st.columns(3)
r1_c1.metric("India VIX Level", f"{brief['latest_vix']}", f"Causal Threshold: {causal['vix_threshold']}")
r1_c2.metric("GARCH(1,1)-t Volatility", f"{brief['latest_garch_vol']}%", f"Persistence: {garch_s['persistence']}")
r1_c3.metric("Next-Day |Nifty| Forecast", f"{brief['pred_abs_ret']}%", f"OOS RMSE: {metrics['oos_rmse']}%")

r2_c1, r2_c2, r2_c3 = st.columns(3)
r2_c1.metric("95% ACI Risk Bound", f"[0%, {brief['aci_upper_95']}%]", f"OOS Coverage: {metrics['aci_cov_95']}%")
r2_c2.metric("Quant Tail-Risk Prob", f"{brief['tail_risk_prob']}%", f"OOS AUC: {metrics['oos_auc']}")
r2_c3.metric("Multimodal Risk (Quant+NLP)", f"{default_alloc['multimodal_risk_pct']}%", f"News Fear Index: {sent_res['news_fear_index']}/100")

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Adaptive Conformal Volatility",
    "2. Econometrics & Causal ATE",
    "3. Systemic Contagion Graph",
    "4. Live News & NLP Sentiment",
    "5. Portfolio Sizing & CVaR",
    "6. What-If Stress Simulator",
    "7. Auto Telegram Alerts"
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
    st.markdown("### Real-Time Indian Equity Headlines & NLP Sentiment Index")
    s1, s2, s3 = st.columns(3)
    s1.metric("News Fear Index (0-100)", f"{sent_res['news_fear_index']}", "50 = Neutral | >50 = Bearish Fear")
    s2.metric("Bullish Headlines", f"{sent_res['bullish_count']} / {sent_res['total_count']}")
    s3.metric("Bearish Headlines", f"{sent_res['bearish_count']} / {sent_res['total_count']}")

    st.divider()
    if sent_res["enriched_news"]:
        for item in sent_res["enriched_news"]:
            badge = "🟢 Bullish" if item["sentiment"] == "Bullish" else ("🔴 Bearish" if item["sentiment"] == "Bearish" else "⚪ Neutral")
            st.markdown(f"- **{badge} (`{item['score']}`)** — **[{item['title']}]({item['link']})**")
            st.caption(f"Published: {item['published']}")
    else:
        st.warning("Live RSS news feed temporarily unavailable; econometric models remain active.")

with tab5:
    st.markdown("### Dynamic Position Sizing & 95% ACI Tail-Loss (CVaR) Calculator")
    st.write("Adjust your total capital and daily risk budget to compute volatility-targeted & tail-risk-adjusted equity exposure.")
    p_col1, p_col2 = st.columns(2)
    user_capital = p_col1.number_input("Total Portfolio Capital (INR ₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
    target_vol = p_col2.slider("Target Daily Portfolio Volatility (%)", 0.3, 2.5, 1.0, 0.1)

    alloc = compute_portfolio_risk_allocation(float(user_capital), float(target_vol), brief, sent_res["news_fear_index"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Equity Exposure", f"{alloc['recommended_equity_pct']}%", f"Deploy ₹{alloc['equity_inr']:,.0f}")
    m2.metric("Safe Cash / Liquid Hedge", f"{alloc['safe_cash_pct']}%", f"Hold ₹{alloc['cash_inr']:,.0f}")
    m3.metric("95% ACI Max 1-Day Drawdown", f"₹{alloc['hedged_aci_loss_inr']:,.0f}", f"Saves ₹{alloc['capital_saved_inr']:,.0f} vs Unhedged")

    st.info(
        f"**Execution Guidance:** Under current GARCH volatility (`{brief['latest_garch_vol']}%`) and Multimodal Risk (`{alloc['multimodal_risk_pct']}%`), "
        f"deploying **₹{alloc['equity_inr']:,.0f}** caps your 95% worst-case single-day Nifty shock loss at **₹{alloc['hedged_aci_loss_inr']:,.0f}** "
        f"(instead of **₹{alloc['unhedged_aci_loss_inr']:,.0f}** if 100% invested)."
    )

with tab6:
    st.markdown("### Interactive What-If Macro & Volatility Stress Simulator")
    st.write("Simulate sudden macro shocks to inspect how our Causal Double-ML & Conformal models re-price next-day market risk.")
    w1, w2, w3 = st.columns(3)
    sim_vix = w1.slider("Simulated India VIX Level", 10.0, 38.0, float(brief["latest_vix"]), 0.5)
    sim_crude = w2.slider("Crude Oil 1-Day Shock (%)", -10.0, 15.0, 0.0, 0.5)
    sim_fx = w3.slider("USD/INR Depreciation Shock (%)", -1.5, 3.0, 0.0, 0.1)

    stress_res = simulate_what_if_stress(sim_vix, sim_crude, sim_fx, brief, causal)

    st_c1, st_c2, st_c3 = st.columns(3)
    st_c1.metric("Simulated Tail-Risk Prob", f"{stress_res['stressed_prob']}%", f"{stress_res['prob_delta']:+.2f}% vs Live")
    st_c2.metric("Simulated 95% ACI Upper Bound", f"{stress_res['stressed_aci_upper']}%", f"{stress_res['aci_delta']:+.2f}% vs Live")
    st_c3.metric(
        "Regime Status",
        "EXTREME CRISIS" if stress_res["stressed_prob"] >= 60 else ("ELEVATED RISK" if stress_res["stressed_prob"] >= brief["alert_thresh_pct"] else "NORMAL"),
        f"Causal VIX Cutoff: {causal['vix_threshold']}"
    )

with tab7:
    st.markdown("### Autonomous Morning Telegram Risk Alert Bot")
    st.write(
        "Receive automated daily risk briefs on your phone at **9:00 AM IST** (15 mins before NSE market open) via GitHub Actions, "
        "or send an instant test alert right now."
    )
    tg_token = st.text_input("Telegram Bot Token (from @BotFather)", value=os.getenv("TELEGRAM_BOT_TOKEN", ""), type="password")
    tg_chat_id = st.text_input("Telegram Chat ID (from @userinfobot)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

    if st.button("Send Instant Risk Brief to Telegram Now"):
        ok, status_msg = send_telegram_risk_alert(tg_token, tg_chat_id, brief, default_alloc)
        if ok:
            st.success(status_msg)
        else:
            st.error(status_msg)

    with st.expander("How to enable 24x7 Automated 9:00 AM IST Daily Alerts (Free on GitHub)"):
        st.markdown("1. Open Telegram, message **`@BotFather`** -> `/newbot` to copy your **Bot Token**.")
        st.markdown("2. Message **`@userinfobot`** to copy your numeric **Chat ID**.")
        st.markdown("3. In your GitHub repo (`rahul25118/quantvault-ai`) -> **Settings -> Secrets and variables -> Actions**, add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.")
        st.markdown("4. The included `.github/workflows/daily_alert.yml` workflow runs every weekday at **9:00 AM IST** automatically!")
