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
    page_title="QuantVault | Institutional Risk Workstation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

amc_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #0B0E14;
    color: #F1F5F9;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.2rem;
    padding-left: 1.4rem;
    padding-right: 1.4rem;
    max-width: 1440px;
}

/* Executive Command Bar */
.amc-topbar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 0.6rem;
    background-color: #131722;
    border: 1px solid #1E2533;
    border-radius: 8px;
    padding: 0.9rem 1.25rem;
    margin-bottom: 0.9rem;
}

.amc-brand-title {
    font-size: 1.22rem;
    font-weight: 700;
    letter-spacing: -0.015em;
    color: #F8FAFC;
    margin: 0;
}

.amc-brand-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748B;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}

.amc-status-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    align-items: center;
}

.amc-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.25rem 0.65rem;
    border-radius: 5px;
    background-color: #191F2E;
    border: 1px solid #283246;
    color: #CBD5E1;
}

.amc-pill-live {
    color: #10B981;
    border-color: rgba(16, 185, 129, 0.35);
    background-color: rgba(16, 185, 129, 0.08);
}

/* Swiss Precision KPI Cards (Zero Truncation) */
[data-testid="stMetric"] {
    background-color: #131722;
    border: 1px solid #1E2533;
    border-radius: 8px;
    padding: 0.78rem 0.85rem;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #3B82F6;
    background-color: #161B28;
}

[data-testid="stMetricLabel"] {
    font-size: 0.69rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #94A3B8 !important;
    white-space: normal !important;
    line-height: 1.25 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #F8FAFC !important;
    white-space: normal !important;
    line-height: 1.3 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.71rem !important;
    white-space: normal !important;
}

/* Executive Regime Panels */
.exec-panel-danger {
    background-color: #131722;
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    height: 100%;
}

.exec-panel-normal {
    background-color: #131722;
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    height: 100%;
}

/* Institutional Workstation Module Selector (All 7 Modules Wrap Cleanly on Every Screen) */
div[role="radiogroup"] {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    background-color: #131722;
    border: 1px solid #1E2533;
    border-radius: 8px;
    padding: 0.5rem;
    margin-bottom: 0.85rem;
}

div[role="radiogroup"] > label {
    background-color: #191F2E;
    border: 1px solid #263042;
    border-radius: 6px;
    padding: 0.42rem 0.8rem !important;
    margin: 0 !important;
    cursor: pointer;
    transition: all 0.15s ease;
}

div[role="radiogroup"] > label:hover {
    border-color: #3B82F6;
}

div[role="radiogroup"] > label[data-checked="true"],
div[role="radiogroup"] > label:has(input:checked) {
    background-color: #1E293B !important;
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6;
}

div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

div[role="radiogroup"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
    margin: 0 !important;
}

/* Chart Header Box */
.chart-header {
    background-color: #131722;
    border: 1px solid #1E2533;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    padding: 0.65rem 0.95rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #F8FAFC;
}

/* Editorial News Feed Rows */
.amc-news-row {
    background-color: #131722;
    border: 1px solid #1E2533;
    border-radius: 7px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.55rem;
}

.amc-news-row a {
    color: #F1F5F9 !important;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.92rem;
}

.amc-news-row a:hover {
    color: #60A5FA !important;
}
</style>
"""
st.markdown(amc_css, unsafe_allow_html=True)

st.sidebar.header("⚙️ Workstation Calibration")
lookback_period = st.sidebar.selectbox("Historical Calibration Window", ["5y", "3y", "2y"], index=0)
aci_gamma = st.sidebar.slider("ACI Online Learning Rate (gamma)", 0.01, 0.15, 0.05, 0.01)
if st.sidebar.button("🔄 Refresh Market Telemetry"):
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

with st.spinner("Loading Institutional Risk Workstation & econometric pipelines..."):
    df_clean, model_df, econ_res, ml_res, sent_res = load_full_pipeline(lookback_period, aci_gamma)

brief = ml_res["live_brief"]
metrics = ml_res["metrics"]
causal = econ_res["causal_ate"]
garch_s = econ_res["garch_summary"]
default_alloc = compute_portfolio_risk_allocation(500000.0, 1.0, brief, sent_res["news_fear_index"])

# Top Institutional Command Bar
st.markdown(
    f"""
    <div class="amc-topbar">
        <div>
            <div class="amc-brand-title">QUANTVAULT <span style="color:#3B82F6; font-weight:500;">| Institutional Risk Workstation</span></div>
            <div class="amc-brand-sub">GARCH(1,1)-t • Orthogonalized Double ML • Adaptive Conformal Inference • NLP Telemetry</div>
        </div>
        <div class="amc-status-group">
            <span class="amc-pill amc-pill-live">● TELEMETRY LIVE</span>
            <span class="amc-pill">NSE CLOSE: {brief['as_of_date']}</span>
            <span class="amc-pill">WINDOW: {lookback_period.upper()}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Split-View Executive Briefing + Capital Allocation Donut Ring
is_elevated = (default_alloc["multimodal_risk_pct"] >= brief["alert_thresh_pct"]) or (brief["latest_vix"] > causal["vix_threshold"])
top_left, top_right = st.columns([1.65, 1])

with top_left:
    if is_elevated:
        st.markdown(
            f"""
            <div class="exec-panel-danger">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.74rem; color:#F87171; letter-spacing:0.06em; margin-bottom:0.3rem;">
                    PORTFOLIO REGIME DIAGNOSTIC // ELEVATED TAIL-RISK
                </div>
                <div style="font-size:1.1rem; font-weight:600; color:#F8FAFC; margin-bottom:0.6rem;">
                    Defensive Volatility &amp; Drawdown Shield Active
                </div>
                <div style="font-size:0.87rem; color:#CBD5E1; line-height:1.6;">
                    • <strong>Quantitative Tail-Risk Probability:</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#F8FAFC;">{brief['tail_risk_prob']}%</span> (Cutoff: {brief['alert_thresh_pct']}%)<br>
                    • <strong>Multimodal Risk Score (Quant + NLP):</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#F8FAFC;">{default_alloc['multimodal_risk_pct']}%</span> (News Fear: {sent_res['news_fear_index']}/100)<br>
                    • <strong>Target Capital Posture:</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#60A5FA;">{default_alloc['recommended_equity_pct']}%</span> Equity / <span style="font-family:'IBM Plex Mono',monospace; color:#10B981;">{default_alloc['safe_cash_pct']}%</span> Liquid Cash
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="exec-panel-normal">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.74rem; color:#34D399; letter-spacing:0.06em; margin-bottom:0.3rem;">
                    PORTFOLIO REGIME DIAGNOSTIC // NORMAL VOLATILITY ENVELOPE
                </div>
                <div style="font-size:1.1rem; font-weight:600; color:#F8FAFC; margin-bottom:0.6rem;">
                    Standard Institutional Risk-Budget Deployment
                </div>
                <div style="font-size:0.87rem; color:#CBD5E1; line-height:1.6;">
                    • <strong>Quantitative Tail-Risk Probability:</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#F8FAFC;">{brief['tail_risk_prob']}%</span> (Cutoff: {brief['alert_thresh_pct']}%)<br>
                    • <strong>Multimodal Risk Score (Quant + NLP):</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#F8FAFC;">{default_alloc['multimodal_risk_pct']}%</span> (News Fear: {sent_res['news_fear_index']}/100)<br>
                    • <strong>Target Capital Posture:</strong> <span style="font-family:'IBM Plex Mono',monospace; color:#60A5FA;">{default_alloc['recommended_equity_pct']}%</span> Equity / <span style="font-family:'IBM Plex Mono',monospace; color:#10B981;">{default_alloc['safe_cash_pct']}%</span> Liquid Cash
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with top_right:
    fig_donut = go.Figure(data=[go.Pie(
        labels=["Equity Exposure", "Liquid Cash / Hedge"],
        values=[default_alloc["recommended_equity_pct"], default_alloc["safe_cash_pct"]],
        hole=0.68,
        marker=dict(colors=["#3B82F6", "#1E293B"], line=dict(color="#0B0E14", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b>: %{value:.1f}%<extra></extra>",
    )])
    fig_donut.update_layout(
        height=175,
        margin=dict(l=15, r=15, t=32, b=10),
        paper_bgcolor="#131722",
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=0.62, font=dict(size=11, color="#CBD5E1")),
        title=dict(text="Recommended Capital Posture", font=dict(size=12, color="#94A3B8"), x=0.04, y=0.94),
        annotations=[dict(
            text=f"<b>{default_alloc['recommended_equity_pct']}%</b><br><span style='font-size:10px;color:#94A3B8'>EQUITY</span>",
            x=0.28, y=0.5, font=dict(size=15, color="#F8FAFC", family="IBM Plex Mono"), showarrow=False
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

st.write("")

# Unified 8-Cell Swiss Precision KPI Matrix (4 x 2 Grid)
r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
r1_c1.metric("India VIX Level", f"{brief['latest_vix']}", f"Cutoff: {causal['vix_threshold']}")
r1_c2.metric("GARCH(1,1)-t Vol", f"{brief['latest_garch_vol']}%", f"Persist: {garch_s['persistence']}")
r1_c3.metric("Next-Day |Nifty|", f"{brief['pred_abs_ret']}%", f"RMSE: {metrics['oos_rmse']}%")
r1_c4.metric("95% ACI Bound", f"0% – {brief['aci_upper_95']}%", f"Cov: {metrics['aci_cov_95']}%")

st.write("")
r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
r2_c1.metric("Tail-Risk Prob", f"{brief['tail_risk_prob']}%", f"AUC: {metrics['oos_auc']}")
r2_c2.metric("Multimodal Risk", f"{default_alloc['multimodal_risk_pct']}%", f"Fear: {sent_res['news_fear_index']}/100")
r2_c3.metric("Equity Exposure", f"{default_alloc['recommended_equity_pct']}%", f"Cash: {default_alloc['safe_cash_pct']}%")
r2_c4.metric("95% Max 1D Loss", f"₹{default_alloc['hedged_aci_loss_inr']:,.0f}", f"Saves ₹{default_alloc['capital_saved_inr']:,.0f}")

st.write("")

# All 7 Modules Visible & Wrapped Cleanly Across Mobile, Tablet, and Desktop
MODULES = [
    "01. Conformal Volatility",
    "02. Causal Econometrics",
    "03. Contagion Network",
    "04. NLP News Telemetry",
    "05. CVaR Position Sizing",
    "06. Macro Stress Lab",
    "07. Alert Automation",
]
active_module = st.radio("Workstation Module", MODULES, horizontal=True, label_visibility="collapsed")

plotly_clean_cfg = {"displayModeBar": "hover", "displaylogo": False}

# --- MODULE 01: INTERACTIVE PLOTLY ACI VOLATILITY & ROC (ZERO OVERLAP) ---
if active_module == "01. Conformal Volatility":
    col_a, col_b = st.columns([1.85, 1])
    with col_a:
        st.markdown('<div class="chart-header">Out-of-Sample Volatility &amp; 95% Adaptive Conformal Band</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"], y=ml_res["aci_upper_95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"], y=ml_res["aci_lower_95"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(59, 130, 246, 0.18)",
            name=f"95% ACI Band (Cov: {metrics['aci_cov_95']}%)",
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"], y=ml_res["y_reg_test"],
            mode="lines", line=dict(color="#E2E8F0", width=1.4),
            name="Realized |Nifty Return| (%)",
        ))
        fig1.add_trace(go.Scatter(
            x=ml_res["dates_test"], y=ml_res["test_reg_preds"],
            mode="lines", line=dict(color="#F59E0B", width=1.2, dash="dot"),
            name="Point Forecast (%)",
        ))
        fig1.update_layout(
            yaxis_title="Daily Absolute Return (%)",
            hovermode="x unified",
            height=410,
            paper_bgcolor="#131722",
            plot_bgcolor="#0E121B",
            font=dict(color="#94A3B8", family="Inter", size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#CBD5E1"),
            ),
            xaxis=dict(gridcolor="#1E2533", showline=True, linecolor="#1E2533"),
            yaxis=dict(gridcolor="#1E2533", showline=True, linecolor="#1E2533"),
            margin=dict(l=48, r=18, t=20, b=75),
        )
        st.plotly_chart(fig1, use_container_width=True, config=plotly_clean_cfg)

    with col_b:
        st.markdown('<div class="chart-header">Out-of-Sample Tail-Risk ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(ml_res["y_clf_test"], ml_res["test_clf_probs"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            line=dict(color="#10B981", width=2.2),
            fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.12)",
            name=f"Tail-Risk RF (AUC = {metrics['oos_auc']})"
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#475569", dash="dash", width=1.2),
            name="Random Baseline"
        ))
        fig2.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=410,
            paper_bgcolor="#131722",
            plot_bgcolor="#0E121B",
            font=dict(color="#94A3B8", family="Inter", size=11),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0,
                font=dict(size=11, color="#CBD5E1"),
            ),
            xaxis=dict(gridcolor="#1E2533", range=[0, 1]),
            yaxis=dict(gridcolor="#1E2533", range=[0, 1.03]),
            margin=dict(l=48, r=18, t=20, b=75),
        )
        st.plotly_chart(fig2, use_container_width=True, config=plotly_clean_cfg)

    st.info(
        f"**Non-Stationarity Recovery Audit:** Static Split Conformal coverage degraded to **{metrics['static_cov_95']}%** "
        f"under regime shifts, whereas online **Adaptive Conformal Inference (ACI)** restored empirical coverage to **{metrics['aci_cov_95']}%**."
    )

# --- MODULE 02: ECONOMETRICS & CAUSAL ATE ---
elif active_module == "02. Causal Econometrics":
    st.markdown("#### Orthogonalized Double ML Causal Regime Impact")
    st.write(
        f"Controlling for crude oil, USD/INR, US 10Y yields, realized volatility, and banking beta, "
        f"entering an **Elevated Fear Regime (IndiaVIX > {causal['vix_threshold']})** causally increases next-day "
        f"tail-risk probability by **+{causal['ate_pct']}%** (SE: `{causal['se_pct']}%`, `t = {causal['t_stat']}`, `p = {causal['p_value']}`)."
    )
    st.markdown("#### GARCH(1,1) Student-t Parameters")
    st.json(garch_s)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**ADF & KPSS Stationarity Diagnostics**")
        st.dataframe(econ_res["stationarity_df"], use_container_width=True)
    with c_right:
        st.markdown("**Granger Causality Lead-Lag Tests**")
        st.dataframe(econ_res["granger_df"], use_container_width=True)

# --- MODULE 03: INTERACTIVE CONTAGION NETWORK ---
elif active_module == "03. Contagion Network":
    col_g1, col_g2 = st.columns([1.45, 1])
    with col_g1:
        st.markdown('<div class="chart-header">Gaussian Precision Contagion Network (Blue = Positive | Red = Inverse)</div>', unsafe_allow_html=True)
        G = econ_res["graph"]
        strength = econ_res["node_strength"]
        pos = nx.circular_layout(G)

        edge_traces = []
        for u, v, d in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            w = d["weight"]
            edge_color = "rgba(239, 68, 68, 0.78)" if w < 0 else "rgba(59, 130, 246, 0.78)"
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(1.5, abs(w) * 8), color=edge_color),
                hoverinfo="none",
                showlegend=False,
            ))

        node_x, node_y, node_text, node_hover, node_sizes = [], [], [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            node_text.append(n)
            node_hover.append(f"<b>{n}</b><br>Conditional Contagion Strength: {strength[n]:.4f}")
            node_sizes.append(26 + strength[n] * 34)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(color="#F8FAFC", size=12, family="Inter"),
            hovertext=node_hover,
            hoverinfo="text",
            marker=dict(
                size=node_sizes,
                color="#1E293B",
                line=dict(width=2, color="#3B82F6"),
            ),
            showlegend=False,
        )

        fig3 = go.Figure(data=edge_traces + [node_trace])
        fig3.update_layout(
            height=430,
            paper_bgcolor="#131722",
            plot_bgcolor="#131722",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.38, 1.38]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.38, 1.38]),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True, config=plotly_clean_cfg)

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

# --- MODULE 04: NLP NEWS TELEMETRY ---
elif active_module == "04. NLP News Telemetry":
    st.markdown("#### Real-Time Indian Equity Headlines & NLP Sentiment Index")
    s1, s2, s3 = st.columns(3)
    s1.metric("News Fear Index (0-100)", f"{sent_res['news_fear_index']}", "50 = Neutral | >50 = Bearish Fear")
    s2.metric("Bullish Headlines", f"{sent_res['bullish_count']} / {sent_res['total_count']}")
    s3.metric("Bearish Headlines", f"{sent_res['bearish_count']} / {sent_res['total_count']}")

    st.write("")
    if sent_res["enriched_news"]:
        for item in sent_res["enriched_news"]:
            tag_color = "#10B981" if item["sentiment"] == "Bullish" else ("#EF4444" if item["sentiment"] == "Bearish" else "#94A3B8")
            st.markdown(
                f"""
                <div class="amc-news-row">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.74rem; font-weight:600; color:{tag_color};">
                        {item['sentiment'].upper()} ({item['score']})
                    </span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748B;"> &nbsp;•&nbsp; {item['published']}</span><br>
                    <a href="{item['link']}" target="_blank">{item['title']}</a>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("Live RSS news feed temporarily unavailable; econometric models remain active.")

# --- MODULE 05: CVaR POSITION SIZING ---
elif active_module == "05. CVaR Position Sizing":
    st.markdown("#### Dynamic Position Sizing & 95% ACI Tail-Loss (CVaR) Calculator")
    st.write("Calibrate portfolio capital and daily volatility budget to compute tail-risk-adjusted equity exposure.")
    p_col1, p_col2 = st.columns(2)
    user_capital = p_col1.number_input("Total Portfolio Capital (INR ₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
    target_vol = p_col2.slider("Target Daily Portfolio Volatility (%)", 0.3, 2.5, 1.0, 0.1)

    alloc = compute_portfolio_risk_allocation(float(user_capital), float(target_vol), brief, sent_res["news_fear_index"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Equity Exposure", f"{alloc['recommended_equity_pct']}%", f"Deploy ₹{alloc['equity_inr']:,.0f}")
    m2.metric("Safe Cash / Liquid Reserve", f"{alloc['safe_cash_pct']}%", f"Hold ₹{alloc['cash_inr']:,.0f}")
    m3.metric("95% ACI Max 1-Day Drawdown", f"₹{alloc['hedged_aci_loss_inr']:,.0f}", f"Saves ₹{alloc['capital_saved_inr']:,.0f} vs Unhedged")

    st.info(
        f"**Execution Guidance:** Under current GARCH volatility (`{brief['latest_garch_vol']}%`) and Multimodal Risk (`{alloc['multimodal_risk_pct']}%`), "
        f"deploying **₹{alloc['equity_inr']:,.0f}** caps your 95% worst-case single-day Nifty shock loss at **₹{alloc['hedged_aci_loss_inr']:,.0f}** "
        f"(compared to **₹{alloc['unhedged_aci_loss_inr']:,.0f}** if 100% invested)."
    )

# --- MODULE 06: MACRO STRESS LAB ---
elif active_module == "06. Macro Stress Lab":
    st.markdown("#### Interactive What-If Macro & Volatility Stress Simulator")
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

# --- MODULE 07: ALERT AUTOMATION ---
elif active_module == "07. Alert Automation":
    st.markdown("#### Autonomous Pre-Market Telegram Risk Alert Bot")
    st.write(
        "Receive automated daily risk briefs at **9:00 AM IST** (15 mins before NSE market open), "
        "or dispatch an instant risk diagnostic alert now."
    )
    tg_token = st.text_input("Telegram Bot Token (from @BotFather)", value=os.getenv("TELEGRAM_BOT_TOKEN", ""), type="password")
    tg_chat_id = st.text_input("Telegram Chat ID (from @userinfobot)", value=os.getenv("TELEGRAM_CHAT_ID", ""))

    if st.button("Dispatch Instant Risk Brief to Telegram"):
        ok, status_msg = send_telegram_risk_alert(tg_token, tg_chat_id, brief, default_alloc)
        if ok:
            st.success(status_msg)
        else:
            st.error(status_msg)

    with st.expander("Setup Guide: Free Telegram Bot Token & Chat ID"):
        st.markdown("1. Open Telegram, search **`@BotFather`** and send `/newbot` to generate your **Bot Token** (then send `/start` to your new bot).")
        st.markdown("2. Search **`@userinfobot`** and send `/start` to copy your numeric **Chat ID**.")
        st.markdown("3. Paste both fields above and click **Dispatch Instant Risk Brief to Telegram**.")

st.divider()
disclaimer_html = (
    '<div style="background-color:#131722; border:1px solid #1E2533; '
    'border-radius:6px; padding:0.85rem 1.1rem; font-size:0.76rem; color:#64748B; line-height:1.5;">'
    '<strong style="color:#94A3B8;">INSTITUTIONAL &amp; REGULATORY DISCLAIMER (NON-ADVISORY RISK SOFTWARE):</strong> '
    'QuantVault is a quantitative risk telemetry, econometric modeling, and stress-testing software infrastructure. '
    'Outputs (GARCH volatility, Double ML causal estimates, Adaptive Conformal bounds, and CVaR sizing metrics) are statistical risk diagnostics '
    'for institutional and educational decision support only and do not constitute SEBI-registered investment advice, research recommendations, or solicitation to buy or sell securities.'
    '</div>'
)
st.markdown(disclaimer_html, unsafe_allow_html=True)
