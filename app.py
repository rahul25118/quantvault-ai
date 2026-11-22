import os
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    background: radial-gradient(circle at 12% 8%, #101D36 0%, #060A13 52%, #04070D 100%);
    color: #E2E8F0;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    max-width: 100%;
}

/* Institutional Hero Banner */
.qv-hero {
    background: linear-gradient(135deg, rgba(15, 25, 46, 0.94) 0%, rgba(9, 15, 28, 0.98) 100%);
    border: 1px solid rgba(0, 240, 255, 0.28);
    border-left: 4px solid #00F0FF;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.5);
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
    margin-bottom: 0.4rem;
}

.qv-title {
    font-size: 1.78rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0;
}

.qv-subtitle {
    color: #94A3B8;
    font-size: 0.88rem;
    margin-top: 0.25rem;
    margin-bottom: 0;
}

/* Live Bloomberg-Style Telemetry Strip */
.ticker-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 1.2rem;
    align-items: center;
    background: rgba(11, 18, 33, 0.92);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 9px;
    padding: 0.55rem 1.1rem;
    margin-bottom: 1.1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
}

.ticker-item {
    color: #CBD5E1;
}

.ticker-val {
    color: #00F0FF;
    font-weight: 700;
}

/* Glassmorphic KPI Cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(17, 26, 46, 0.90), rgba(9, 15, 28, 0.96));
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-top: 2px solid rgba(0, 240, 255, 0.5);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.38);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 240, 255, 0.65);
    box-shadow: 0 10px 26px rgba(0, 240, 255, 0.12);
}

[data-testid="stMetricLabel"] {
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94A3B8 !important;
    white-space: normal !important;
}

[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    white-space: normal !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
}

/* Regime Alert Cards */
.regime-card-danger {
    background: linear-gradient(120deg, rgba(255, 59, 92, 0.18) 0%, rgba(13, 20, 36, 0.95) 100%);
    border: 1px solid rgba(255, 59, 92, 0.45);
    border-left: 5px solid #FF3B5C;
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    height: 100%;
    color: #FFE4E6;
}

.regime-card-safe {
    background: linear-gradient(120deg, rgba(0, 230, 118, 0.15) 0%, rgba(13, 20, 36, 0.95) 100%);
    border: 1px solid rgba(0, 230, 118, 0.42);
    border-left: 5px solid #00E676;
    border-radius: 12px;
    padding: 1.15rem 1.3rem;
    height: 100%;
    color: #DCFCE7;
}

/* Wrapped Institutional Tabs */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    background-color: rgba(11, 18, 33, 0.88);
    padding: 0.45rem;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.16);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    padding: 0.42rem 0.82rem;
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.83rem;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 240, 255, 0.20), rgba(0, 140, 255, 0.16)) !important;
    color: #00F0FF !important;
    border: 1px solid rgba(0, 240, 255, 0.45) !important;
}

/* News Feed Cards */
.news-card {
    background: rgba(13, 21, 38, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 9px;
    padding: 0.78rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.15s ease;
}

.news-card:hover {
    border-color: rgba(0, 240, 255, 0.45);
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
        <div class="qv-badge">● AUTONOMOUS QUANT TERMINAL &nbsp;|&nbsp; GARCH(1,1)-t • DOUBLE ML • ACI • NLP FUSION</div>
        <h1 class="qv-title">QuantVault AI — Institutional Risk & Decision Intelligence</h1>
        <p class="qv-subtitle">
            Real-time Indian equity tail-risk quantification, causal macro regime diagnostics, precision contagion networks, and dynamic CVaR position sizing.
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
    return df_clean, model_df, econ_res, ml_res, sentiment_res

with st.spinner("Initializing QuantVault Institutional Terminal & live econometric pipelines..."):
    df_clean, model_df, econ_res, ml_res, sent_res = load_full_pipeline(lookback_period, aci_gamma)

brief = ml_res["live_brief"]
metrics = ml_res["metrics"]
causal = econ_res["causal_ate"]
garch_s = econ_res["garch_summary"]
default_alloc = compute_portfolio_risk_allocation(500000.0, 1.0, brief, sent_res["news_fear_index"])

# Live Telemetry Strip
st.markdown(
    f"""
    <div class="ticker-strip">
        <span class="ticker-item">CLOSE: <span class="ticker-val">{brief['as_of_date']}</span></span>
        <span class="ticker-item">INDIA VIX: <span class="ticker-val">{brief['latest_vix']}</span></span>
        <span class="ticker-item">GARCH VOL: <span class="ticker-val">{brief['latest_garch_vol']}%</span></span>
        <span class="ticker-item">95% ACI CAP: <span class="ticker-val">{brief['aci_upper_95']}%</span></span>
        <span class="ticker-item">MULTIMODAL RISK: <span class="ticker-val">{default_alloc['multimodal_risk_pct']}%</span></span>
        <span class="ticker-item">OPTIMAL EQUITY: <span class="ticker-val">{default_alloc['recommended_equity_pct']}%</span></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Row: Regime Diagnostic Card + Live Interactive Risk Gauge Speedometer
is_elevated = (default_alloc["multimodal_risk_pct"] >= brief["alert_thresh_pct"]) or (brief["latest_vix"] > causal["vix_threshold"])
top_left, top_right = st.columns([1.65, 1])

with top_left:
    if is_elevated:
        st.markdown(
            f"""
            <div class="regime-card-danger">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#FF8FA3; margin-bottom:0.35rem;">
                    REGIME STATUS: ELEVATED TAIL-RISK ALERT
                </div>
                <div style="font-size:1.15rem; font-weight:700; margin-bottom:0.55rem;">
                    🚨 Defensive Capital Preservation Mode Active
                </div>
                <div style="font-size:0.9rem; line-height:1.55;">
                    • <strong>Quant Tail-Risk Probability:</strong> {brief['tail_risk_prob']}% (Alert Cutoff: {brief['alert_thresh_pct']}%)<br>
                    • <strong>Multimodal Risk Score (Quant + NLP News):</strong> {default_alloc['multimodal_risk_pct']}%<br>
                    • <strong>Recommended Equity Allocation:</strong> {default_alloc['recommended_equity_pct']}% &nbsp;|&nbsp; <strong>Safe Cash/Hedge:</strong> {default_alloc['safe_cash_pct']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="regime-card-safe">
                <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#6EE7B7; margin-bottom:0.35rem;">
                    REGIME STATUS: NORMAL VOLATILITY ENVELOPE
                </div>
                <div style="font-size:1.15rem; font-weight:700; margin-bottom:0.55rem;">
                    ✅ Standard Risk-Budget Deployment Active
                </div>
                <div style="font-size:0.9rem; line-height:1.55;">
                    • <strong>Quant Tail-Risk Probability:</strong> {brief['tail_risk_prob']}% (Alert Cutoff: {brief['alert_thresh_pct']}%)<br>
                    • <strong>Multimodal Risk Score (Quant + NLP News):</strong> {default_alloc['multimodal_risk_pct']}%<br>
                    • <strong>Recommended Equity Allocation:</strong> {default_alloc['recommended_equity_pct']}% &nbsp;|&nbsp; <strong>Safe Cash/Hedge:</strong> {default_alloc['safe_cash_pct']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with top_right:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=default_alloc["multimodal_risk_pct"],
        number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC", "family": "JetBrains Mono"}},
        delta={"reference": brief["alert_thresh_pct"], "increasing": {"color": "#FF3B5C"}, "decreasing": {"color": "#00E676"}},
        title={"text": "Multimodal Tail-Risk Meter", "font": {"size": 13, "color": "#94A3B8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
            "bar": {"color": "#00F0FF", "thickness": 0.28},
            "bgcolor": "#0B1221",
            "borderwidth": 1,
            "bordercolor": "#1E293B",
            "steps": [
                {"range": [0, 30], "color": "rgba(0, 230, 118, 0.22)"},
                {"range": [30, 60], "color": "rgba(251, 191, 36, 0.22)"},
                {"range": [60, 100], "color": "rgba(255, 59, 92, 0.25)"},
            ],
            "threshold": {
                "line": {"color": "#FF3B5C", "width": 3},
                "thickness": 0.8,
                "value": brief["alert_thresh_pct"],
            },
        },
    ))
    fig_gauge.update_layout(
        height=185,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(13, 20, 36, 0.92)",
        font={"color": "#E2E8F0"},
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.write("")

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Adaptive Conformal Band",
    "2. Econometrics & Causal ATE",
    "3. Contagion Network",
    "4. Live News & NLP Fusion",
    "5. Portfolio Sizing & CVaR",
    "6. What-If Stress Lab",
    "7. Telegram Bot Alerts"
])

# --- TAB 1: INTERACTIVE PLOTLY ACI VOLATILITY & ROC ---
with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"],
            y=ml_res["aci_upper_95"],
            mode="lines",
            line=dict(width=0),
            name="95% ACI Upper Bound",
            showlegend=False,
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"],
            y=ml_res["aci_lower_95"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 59, 92, 0.22)",
            name=f"95% Adaptive Conformal Band (Cov: {metrics['aci_cov_95']}%)",
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"],
            y=ml_res["y_reg_test"],
            mode="lines",
            line=dict(color="#00F0FF", width=1.6),
            name="Actual |Nifty Return| (%)",
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"],
            y=ml_res["test_reg_preds"],
            mode="lines",
            line=dict(color="#FBBF24", width=1.3, dash="dash"),
            name="Point Forecast (%)",
        ))
        fig1.update_layout(
            title="Out-of-Sample Volatility & 95% Adaptive Conformal Interval (Interactive)",
            yaxis_title="Daily Absolute Return (%)",
            hovermode="x unified",
            height=430,
            paper_bgcolor="#0B1221",
            plot_bgcolor="#0E172A",
            font=dict(color="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=45, r=20, t=55, b=40),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fpr, tpr, _ = roc_curve(ml_res["y_clf_test"], ml_res["test_clf_probs"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#00E676", width=2.5),
            fill="tozeroy", fillcolor="rgba(0, 230, 118, 0.12)",
            name=f"Tail-Risk RF (AUC = {metrics['oos_auc']})"
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#64748B", dash="dash"),
            name="Random Baseline"
        ))
        fig2.update_layout(
            title="Out-of-Sample Tail-Risk ROC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=430,
            paper_bgcolor="#0B1221",
            plot_bgcolor="#0E172A",
            font=dict(color="#E2E8F0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            margin=dict(l=45, r=20, t=55, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        f"**Non-Stationarity Recovery Audit:** Static Split Conformal coverage degraded to **{metrics['static_cov_95']}%** "
        f"under regime shifts, whereas online **Adaptive Conformal Inference (ACI)** restored empirical coverage to **{metrics['aci_cov_95']}%**."
    )

# --- TAB 2: ECONOMETRICS & CAUSAL ATE ---
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

# --- TAB 3: INTERACTIVE PLOTLY CONTAGION NETWORK ---
with tab3:
    col_g1, col_g2 = st.columns([1.45, 1])
    with col_g1:
        G = econ_res["graph"]
        strength = econ_res["node_strength"]
        pos = nx.circular_layout(G)

        edge_traces = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d["weight"]
            edge_color = "#FF3B5C" if w < 0 else "#00F0FF"
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(1.5, abs(w) * 9), color=edge_color),
                hoverinfo="none",
                showlegend=False,
            ))

        node_x, node_y, node_text, node_hover, node_sizes = [], [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_text.append(n)
            node_hover.append(f"<b>{n}</b><br>Contagion Strength: {strength[n]:.4f}")
            node_sizes.append(28 + strength[n] * 36)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=12, family="Inter"),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color="#0F172A",
                line=dict(width=2.5, color="#00F0FF"),
            ),
            showlegend=False,
        )

        fig3 = go.Figure(data=edge_traces + [node_trace])
        fig3.update_layout(
            title="Interactive Systemic Contagion Graph (Cyan = Positive | Crimson = Inverse Spillover)",
            height=450,
            paper_bgcolor="#0B1221",
            plot_bgcolor="#0B1221",
            font=dict(color="#E2E8F0"),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.35, 1.35]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.35, 1.35]),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_g2:
        st.markdown("**Systemic Hub Contagion Strength**")
        hub_df = pd.DataFrame(
            sorted(strength.items(), key=lambda x: x[1], reverse=True),
            columns=["Asset / Macro Node", "Conditional Contagion Strength"]
        )
        st.dataframe(
            hub_df,
            use_container_width=True,
            column_config={
                "Conditional Contagion Strength": st.column_config.ProgressColumn(
                    "Contagion Strength",
                    format="%.4f",
                    min_value=0.0,
                    max_value=float(hub_df["Conditional Contagion Strength"].max()) if not hub_df.empty else 1.0,
                )
            },
        )

# --- TAB 4: LIVE NEWS & NLP SENTIMENT ---
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

# --- TAB 5: PORTFOLIO POSITION SIZING & CVaR ---
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

# --- TAB 6: WHAT-IF MACRO STRESS SIMULATOR ---
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

# --- TAB 7: AUTONOMOUS TELEGRAM RISK ALERTS ---
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
