import io
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve
from quant_engine import (
    analyze_custom_portfolio_cvar,
    compute_portfolio_risk_allocation,
    engineer_quant_features,
    fetch_live_macro_news,
    fetch_live_market_data,
    generate_institutional_pdf_tearsheet,
    optimize_nifty_put_hedge,
    run_aci_strategy_backtest,
    run_econometric_and_causal_suite,
    score_news_sentiment,
    send_telegram_risk_alert,
    simulate_what_if_stress,
    train_and_evaluate_conformal_engine,
)

st.set_page_config(
    page_title="QuantVault AI | Institutional Risk Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

terminal_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #0E192F 0%, #070B14 55%, #05080F 100%);
    color: #E2E8F0;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

.qv-hero {
    background: linear-gradient(135deg, rgba(14, 23, 42, 0.92) 0%, rgba(11, 18, 33, 0.96) 100%);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-left: 4px solid #00F0FF;
    border-radius: 12px;
    padding: 1.15rem 1.45rem;
    margin-bottom: 1.15rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
}

.qv-badge {
    display: inline-block;
    background: rgba(0, 240, 255, 0.12);
    color: #00F0FF;
    border: 1px solid rgba(0, 240, 255, 0.35);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}

.qv-title {
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0;
}

.qv-subtitle {
    color: #94A3B8;
    font-size: 0.9rem;
    margin-top: 0.3rem;
    margin-bottom: 0;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(16, 24, 43, 0.88), rgba(10, 16, 30, 0.95));
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-top: 2px solid rgba(0, 240, 255, 0.45);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.18s ease, border-color 0.18s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 240, 255, 0.55);
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94A3B8 !important;
    white-space: normal !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.48rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    white-space: normal !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

.regime-card-danger {
    background: linear-gradient(90deg, rgba(255, 59, 92, 0.16) 0%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(255, 59, 92, 0.45);
    border-left: 5px solid #FF3B5C;
    border-radius: 10px;
    padding: 0.95rem 1.2rem;
    margin-bottom: 1.05rem;
    color: #FFE4E6;
}

.regime-card-safe {
    background: linear-gradient(90deg, rgba(0, 230, 118, 0.14) 0%, rgba(15, 23, 42, 0.92) 100%);
    border: 1px solid rgba(0, 230, 118, 0.4);
    border-left: 5px solid #00E676;
    border-radius: 10px;
    padding: 0.95rem 1.2rem;
    margin-bottom: 1.05rem;
    color: #DCFCE7;
}

.stTabs [data-baseweb="tab-list"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    background-color: rgba(11, 18, 33, 0.85);
    padding: 0.45rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.15);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 0.4rem 0.75rem;
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.82rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.18), rgba(0, 150, 255, 0.15)) !important;
    color: #00F0FF !important;
    border: 1px solid rgba(0, 240, 255, 0.4) !important;
}

.news-card {
    background: rgba(13, 21, 38, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 9px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}

.news-card a {
    color: #F1F5F9 !important;
    text-decoration: none;
    font-weight: 600;
}

.news-card a:hover {
    color: #00F0FF !important;
}
</style>
"""
st.markdown(terminal_css, unsafe_allow_html=True)

st.markdown(
    """
    <div class="qv-hero">
        <div class="qv-badge">● INSTITUTIONAL HEDGE-FUND DESK &nbsp;|&nbsp; GARCH(1,1)-t • DOUBLE ML • ACI • NLP • BSM OPTIONS</div>
        <h1 class="qv-title">QuantVault AI — Institutional Risk & Decision Intelligence</h1>
        <p class="qv-subtitle">
            End-to-end Indian equity tail-risk telemetry, multi-asset CVaR attribution, OOS strategy backtesting, options tail-hedging, and automated PDF reporting.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Engine Parameters")
lookback_period = st.sidebar.selectbox("Historical Calibration Window", ["5y", "3y", "2y"], index=0)
aci_gamma = st.sidebar.slider("ACI Online Learning Rate (gamma)", 0.01, 0.15, 0.05, 0.01)
if st.sidebar.button("🔄 Refresh Live Market & Macro Data"):
    st.cache_data.clear()

@st.cache_data(ttl=3600)
def load_full_pipeline(period: str, gamma: float):
    df_clean = fetch_live_market_data(period=period)
    feat_df, model_df, garch_fit = engineer_quant_features(df_clean)
    econ_res = run_econometric_and_causal_suite(model_df, garch_fit)
    ml_res = train_and_evaluate_conformal_engine(feat_df, model_df, gamma=gamma)
    raw_news = fetch_live_macro_news(limit=12)
    sentiment_res = score_news_sentiment(raw_news)
    bt_res = run_aci_strategy_backtest(model_df, ml_res, target_vol_pct=1.0)
    return df_clean, model_df, econ_res, ml_res, sentiment_res, bt_res

with st.spinner("Initializing QuantVault Institutional Terminal & live econometric pipelines..."):
    df_clean, model_df, econ_res, ml_res, sent_res, bt_res = load_full_pipeline(lookback_period, aci_gamma)

brief = ml_res["live_brief"]
metrics = ml_res["metrics"]
causal = econ_res["causal_ate"]
garch_s = econ_res["garch_summary"]
default_alloc = compute_portfolio_risk_allocation(500000.0, 1.0, brief, sent_res["news_fear_index"])

# Top Action Bar with Instant PDF Tear-Sheet Download
hdr_col1, hdr_col2 = st.columns([3, 1])
with hdr_col1:
    st.markdown(f"#### 📡 Live Autonomous Risk Brief &nbsp; `CLOSE: {brief['as_of_date']}`")
with hdr_col2:
    pdf_bytes = generate_institutional_pdf_tearsheet(brief, causal, garch_s, metrics, default_alloc, sent_res, bt_res)
    st.download_button(
        label="📄 Download Executive Risk Tear-Sheet (PDF)",
        data=pdf_bytes,
        file_name=f"QuantVault_Risk_TearSheet_{brief['as_of_date']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

is_elevated = (default_alloc["multimodal_risk_pct"] >= brief["alert_thresh_pct"]) or (brief["latest_vix"] > causal["vix_threshold"])

if is_elevated:
    st.markdown(
        f"""
        <div class="regime-card-danger">
            <strong>🚨 ELEVATED TAIL-RISK REGIME DETECTED</strong><br>
            Quant Tail-Risk Probability is <strong>{brief['tail_risk_prob']}%</strong> &nbsp;|&nbsp;
            Multimodal Risk Score (Quant + NLP News): <strong>{default_alloc['multimodal_risk_pct']}%</strong>
            (Alert Cutoff: {brief['alert_thresh_pct']}%) &nbsp;|&nbsp;
            Recommended Equity Exposure: <strong>{default_alloc['recommended_equity_pct']}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="regime-card-safe">
            <strong>✅ NORMAL VOLATILITY REGIME</strong><br>
            Quant Tail-Risk Probability is <strong>{brief['tail_risk_prob']}%</strong> &nbsp;|&nbsp;
            Multimodal Risk Score (Quant + NLP News): <strong>{default_alloc['multimodal_risk_pct']}%</strong>
            (Alert Cutoff: {brief['alert_thresh_pct']}%) &nbsp;|&nbsp;
            Recommended Equity Exposure: <strong>{default_alloc['recommended_equity_pct']}%</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Full 8 Glassmorphic KPI Cards (4 + 4 Institutional Grid)
r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
r1_c1.metric("India VIX Level", f"{brief['latest_vix']}", f"Causal Cutoff: {causal['vix_threshold']}")
r1_c2.metric("GARCH(1,1)-t Volatility", f"{brief['latest_garch_vol']}%", f"Persistence: {garch_s['persistence']}")
r1_c3.metric("Next-Day |Nifty| Forecast", f"{brief['pred_abs_ret']}%", f"OOS RMSE: {metrics['oos_rmse']}%")
r1_c4.metric("95% ACI Risk Bound", f"[0%, {brief['aci_upper_95']}%]", f"OOS Coverage: {metrics['aci_cov_95']}%")

st.write("")
r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
r2_c1.metric("Quant Tail-Risk Prob", f"{brief['tail_risk_prob']}%", f"OOS ROC-AUC: {metrics['oos_auc']}")
r2_c2.metric("Multimodal Risk (Quant+NLP)", f"{default_alloc['multimodal_risk_pct']}%", f"News Fear: {sent_res['news_fear_index']}/100")
r2_c3.metric("Suggested Equity Exposure", f"{default_alloc['recommended_equity_pct']}%", f"Safe Cash: {default_alloc['safe_cash_pct']}%")
r2_c4.metric("95% ACI 1-Day Max Loss", f"₹{default_alloc['hedged_aci_loss_inr']:,.0f}", f"Saves ₹{default_alloc['capital_saved_inr']:,.0f}")

st.write("")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "1. Adaptive Conformal Band",
    "2. Econometrics & Causal ATE",
    "3. Contagion Network",
    "4. Live News & NLP Fusion",
    "5. Portfolio Sizing & CVaR",
    "6. What-If Stress Lab",
    "7. Telegram Bot Alerts",
    "8. Multi-Asset Portfolio CSV",
    "9. Strategy Backtest Proof",
    "10. Options Tail-Hedge Lab"
])

def style_dark_ax(ax, fig):
    fig.patch.set_facecolor("#0B1221")
    ax.set_facecolor("#0E172A")
    ax.tick_params(colors="#94A3B8", labelsize=9)
    ax.xaxis.label.set_color("#CBD5E1")
    ax.yaxis.label.set_color("#CBD5E1")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#1E293B")
    ax.grid(True, linestyle="--", linewidth=0.5, color="#1E293B", alpha=0.8)

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(10, 4.5))
        style_dark_ax(ax1, fig1)
        ax1.plot(ml_res["dates_test"], ml_res["y_reg_test"], color="#00F0FF", linewidth=1.3, alpha=0.9, label="Actual |Nifty Return| (%)")
        ax1.plot(ml_res["dates_test"], ml_res["test_reg_preds"], color="#FBBF24", linestyle="--", linewidth=1.2, label="Point Forecast")
        ax1.fill_between(
            ml_res["dates_test"], ml_res["aci_lower_95"], ml_res["aci_upper_95"],
            color="#FF3B5C", alpha=0.25, label=f"95% Adaptive Conformal Band (Cov: {metrics['aci_cov_95']}%)"
        )
        ax1.set_title("Out-of-Sample Volatility & 95% Adaptive Conformal Interval", fontweight="bold")
        ax1.set_ylabel("Daily Absolute Return (%)")
        leg1 = ax1.legend(loc="upper right", facecolor="#0B1221", edgecolor="#334155")
        for text in leg1.get_texts():
            text.set_color("#E2E8F0")
        st.pyplot(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        style_dark_ax(ax2, fig2)
        fpr, tpr, _ = roc_curve(ml_res["y_clf_test"], ml_res["test_clf_probs"])
        ax2.plot(fpr, tpr, color="#00E676", linewidth=2.2, label=f"Tail-Risk RF (AUC = {metrics['oos_auc']})")
        ax2.plot([0, 1], [0, 1], color="#64748B", linestyle="--")
        ax2.set_title("Out-of-Sample Tail-Risk ROC", fontweight="bold")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        leg2 = ax2.legend(loc="lower right", facecolor="#0B1221", edgecolor="#334155")
        for text in leg2.get_texts():
            text.set_color("#E2E8F0")
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
        fig3.patch.set_facecolor("#0B1221")
        ax3.set_facecolor("#0B1221")
        pos = nx.circular_layout(G)
        edge_colors = ["#FF3B5C" if d["weight"] < 0 else "#00F0FF" for _, _, d in G.edges(data=True)]
        edge_widths = [abs(d["weight"]) * 6.5 for _, _, d in G.edges(data=True)]
        node_sizes = [750 + strength[n] * 950 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=node_sizes, node_color="#1E293B", edgecolors="#00F0FF", linewidths=2)
        nx.draw_networkx_edges(G, pos, ax=ax3, width=edge_widths, edge_color=edge_colors, alpha=0.85)
        nx.draw_networkx_labels(G, pos, ax=ax3, font_size=9.5, font_weight="bold", font_color="#F8FAFC")
        ax3.set_title("Systemic Contagion Network (Precision Partial Correlations)", fontweight="bold", color="#F8FAFC")
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

    st.write("")
    if sent_res["enriched_news"]:
        for item in sent_res["enriched_news"]:
            badge = "🟢 BULLISH" if item["sentiment"] == "Bullish" else ("🔴 BEARISH" if item["sentiment"] == "Bearish" else "⚪ NEUTRAL")
            st.markdown(
                f"""
                <div class="news-card">
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#00F0FF;">{badge} ({item['score']})</span><br>
                    <a href="{item['link']}" target="_blank">{item['title']}</a><br>
                    <span style="font-size:0.76rem; color:#64748B;">Published: {item['published']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
        "Receive automated daily risk briefs on your phone at **9:00 AM IST** (15 mins before NSE market open), "
        "or send an instant test alert right now."
    )
    tg_token = st.text_input("Telegram Bot Token (from @BotFather)", value=os.getenv("TELEGRAM_BOT_TOKEN", ""), type="password")
    tg_chat_id = st.text_input("Telegram Chat ID (from @userinfobot)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

    if st.button("🚀 Send Instant Risk Brief to Telegram Now"):
        ok, status_msg = send_telegram_risk_alert(tg_token, tg_chat_id, brief, default_alloc)
        if ok:
            st.success(status_msg)
        else:
            st.error(status_msg)

    with st.expander("How to get your Free Telegram Bot Token & Chat ID in 60 seconds"):
        st.markdown("1. Open Telegram, search **`@BotFather`** and send `/newbot` to get your **Bot Token** (then send `/start` to your new bot).")
        st.markdown("2. Search **`@userinfobot`** and send `/start` to copy your numeric **Chat ID**.")
        st.markdown("3. Paste both above and click **Send Instant Risk Brief to Telegram Now**.")

# --- TAB 8: MULTI-ASSET PORTFOLIO CSV UPLOAD & LIVE BETA / CVaR ATTRIBUTION ---
with tab8:
    st.markdown("### Multi-Asset Portfolio CSV Upload & Component 95% CVaR Attribution")
    st.write("Upload a CSV with `ticker` and `weight` columns (e.g., `RELIANCE, 25`) or test with our institutional NSE large-cap basket.")

    sample_df = pd.DataFrame({
        "ticker": ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "LT"],
        "weight": [22, 20, 18, 15, 13, 12]
    })

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded_csv = st.file_uploader("Upload Portfolio Holdings CSV (`ticker`, `weight`)", type=["csv"])
    with up_col2:
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="sample_nse_portfolio.csv",
            mime="text/csv",
            use_container_width=True,
        )

    active_holdings = pd.read_csv(uploaded_csv) if uploaded_csv is not None else sample_df
    try:
        with st.spinner("Computing live stock-level Nifty Beta & Component 95% CVaR..."):
            port_audit = analyze_custom_portfolio_cvar(active_holdings, period="1y")

        pa1, pa2, pa3 = st.columns(3)
        pa1.metric("Portfolio Nifty Beta", f"{port_audit['portfolio_beta']}x", "1.00x = Market Neutral Beta")
        pa2.metric("Annualized Portfolio Vol", f"{port_audit['portfolio_ann_vol']}%", "1-Year Realized")
        pa3.metric("Portfolio 1-Day 95% CVaR", f"{port_audit['portfolio_cvar_95']}%", "Expected Tail-Day Loss")

        st.dataframe(port_audit["attribution_df"], use_container_width=True)
    except Exception as e:
        st.error(f"Portfolio Attribution Error: {e}")

# --- TAB 9: OUT-OF-SAMPLE STRATEGY BACKTEST PROOF ---
with tab9:
    st.markdown("### Out-of-Sample Strategy Backtest: 100% Nifty vs QuantVault ACI Dynamic Sizing")
    st.write("Empirical proof comparing passive Nifty Buy-and-Hold against online Adaptive Conformal + Volatility-Targeted risk sizing.")

    bh_s = bt_res["bh_stats"]
    qv_s = bt_res["qv_stats"]

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("QuantVault Sharpe Ratio", f"{qv_s['sharpe']}", f"vs {bh_s['sharpe']} Buy & Hold")
    b2.metric("QuantVault Sortino Ratio", f"{qv_s['sortino']}", f"vs {bh_s['sortino']} Buy & Hold")
    b3.metric("QuantVault Max Drawdown", f"{qv_s['max_dd_pct']}%", f"Saves {bt_res['dd_reduction_pct']}% Drawdown")
    b4.metric("Avg Active Equity Weight", f"{bt_res['avg_equity_weight_pct']}%", "Remainder in Liquid Yield")

    fig_bt, ax_bt = plt.subplots(figsize=(10.5, 4.3))
    style_dark_ax(ax_bt, fig_bt)
    ax_bt.plot(bt_res["dates"], bt_res["bh_equity"], color="#94A3B8", linestyle="--", linewidth=1.4, label=f"100% Nifty Buy & Hold (Max DD: {bh_s['max_dd_pct']}%)")
    ax_bt.plot(bt_res["dates"], bt_res["qv_equity"], color="#00E676", linewidth=2.0, label=f"QuantVault ACI Risk-Managed (Max DD: {qv_s['max_dd_pct']}%)")
    ax_bt.set_title("Out-of-Sample Cumulative Wealth Curve (Base = 100)", fontweight="bold")
    ax_bt.set_ylabel("Portfolio Index Value")
    leg_bt = ax_bt.legend(loc="upper left", facecolor="#0B1221", edgecolor="#334155")
    for text in leg_bt.get_texts():
        text.set_color("#E2E8F0")
    st.pyplot(fig_bt)

# --- TAB 10: BLACK-SCHOLES OPTIONS TAIL-HEDGE OPTIMIZER ---
with tab10:
    st.markdown("### Options Tail-Hedge Strike & Cost Optimizer (Black-Scholes-Merton)")
    st.write("Compute optimal OTM Nifty Put protection contracts and hedging cost (in bps) using live India VIX implied volatility.")

    latest_spot = float(df_clean["nifty_close"].iloc[-1]) if "nifty_close" in df_clean.columns else 24200.0
    o1, o2, o3 = st.columns(3)
    hedge_cap = o1.number_input("Portfolio Value to Hedge (INR ₹)", min_value=100000, max_value=500000000, value=2500000, step=100000)
    otm_offset = o2.slider("Put Strike OTM Buffer (%)", 0.5, 6.0, 2.0, 0.5)
    dte_days = o3.slider("Days to Expiry (DTE)", 2, 30, 7, 1)

    opt_res = optimize_nifty_put_hedge(latest_spot, float(brief["latest_vix"]), float(hedge_cap), otm_offset, dte_days)

    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Recommended Put Strike", f"{opt_res['recommended_strike']} PE", f"Spot: {opt_res['spot_nifty']}")
    oc2.metric("Est. BSM Put Premium", f"₹{opt_res['put_premium_pts']}", f"Delta: {opt_res['put_delta']}")
    oc3.metric("Total Hedge Cost", f"₹{opt_res['total_hedge_cost_inr']:,.0f}", f"{opt_res['hedge_cost_bps']} bps ({opt_res['lots_required']} lots)")
    oc4.metric("5% Crash Net Loss", f"₹{opt_res['hedged_5pct_crash_loss']:,.0f}", f"vs ₹{opt_res['unhedged_5pct_crash_loss']:,.0f} Unhedged")

st.divider()
disclaimer_html = (
    '<div style="background:rgba(15,23,42,0.85); border:1px solid rgba(148,163,184,0.18); '
    'border-radius:8px; padding:0.85rem 1.1rem; font-size:0.78rem; color:#94A3B8;">'
    '<strong style="color:#E2E8F0;">INSTITUTIONAL &amp; REGULATORY DISCLAIMER (NON-ADVISORY RISK SOFTWARE):</strong> '
    'QuantVault AI is a quantitative risk telemetry, econometric modeling, and stress-testing software infrastructure. '
    'Outputs (GARCH volatility, Double ML causal estimates, Adaptive Conformal bounds, and CVaR sizing metrics) are statistical risk diagnostics '
    'for institutional/educational decision support only and do not constitute SEBI-registered investment advice, research recommendations, or solicitation to buy/sell securities.'
    '</div>'
)
st.markdown(disclaimer_html, unsafe_allow_html=True)


st.divider()
st.markdown(
    """
    <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(148,163,184,0.18); border-radius:8px; padding:0.85rem 1.1rem; font-size:0.78rem; color:#94A3B8;">
        <strong style="color:#E2E8F0;">INSTITUTIONAL & REGULATORY DISCLAIMER (NON-ADVISORY RISK SOFTWARE):</strong>
        QuantVault AI is a quantitative risk telemetry, econometric modeling, and stress-testing software infrastructure.
        Outputs (GARCH volatility, Double ML causal estimates, Adaptive Conformal bounds, and CVaR sizing metrics) are statistical risk diagnostics
        for institutional/educational decision support only and do not constitute SEBI-registered investment advice, research recommendations, or solicitation to buy/sell securities.
    </div>
    """,
    unsafe_allow_html=True,
)
