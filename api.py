from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from quant_engine import (
    compute_portfolio_risk_allocation,
    engineer_quant_features,
    fetch_live_macro_news,
    fetch_live_market_data,
    run_econometric_and_causal_suite,
    score_news_sentiment,
    simulate_what_if_stress,
    train_and_evaluate_conformal_engine,
)

app = FastAPI(
    title="QuantVault AI — Institutional Risk Telemetry API",
    description="Non-advisory B2B quantitative market risk, Adaptive Conformal Inference (ACI), Double ML causal diagnostics, and CVaR sizing API.",
    version="1.0.0",
)


class StressTestRequest(BaseModel):
    sim_vix: float = Field(..., ge=8.0, le=60.0, description="Simulated India VIX level")
    crude_shock_pct: float = Field(0.0, ge=-30.0, le=40.0, description="Crude oil 1-day shock (%)")
    usdinr_shock_pct: float = Field(0.0, ge=-5.0, le=10.0, description="USD/INR depreciation shock (%)")


@app.get("/health")
def health_check():
    return {
        "status": "active",
        "service": "QuantVault AI Enterprise Risk API",
        "compliance": "Non-Advisory Quantitative Telemetry Software",
    }


@app.get("/api/v1/live-risk")
def get_live_risk_telemetry(
    capital_inr: float = Query(1000000.0, gt=0, description="Portfolio capital in INR"),
    target_daily_vol_pct: float = Query(1.0, gt=0.1, le=5.0, description="Target daily portfolio volatility (%)"),
    lookback: str = Query("5y", pattern="^(2y|3y|5y)$"),
):
    try:
        df_clean = fetch_live_market_data(period=lookback)
        feat_df, model_df, garch_fit = engineer_quant_features(df_clean)
        econ_res = run_econometric_and_causal_suite(model_df, garch_fit)
        ml_res = train_and_evaluate_conformal_engine(feat_df, model_df, gamma=0.05)
        sent_res = score_news_sentiment(fetch_live_macro_news(limit=12))

        brief = ml_res["live_brief"]
        alloc = compute_portfolio_risk_allocation(
            capital_inr=capital_inr,
            target_daily_vol_pct=target_daily_vol_pct,
            live_brief=brief,
            news_fear_index=sent_res["news_fear_index"],
        )

        return {
            "market_close_date": brief["as_of_date"],
            "regime": "ELEVATED_TAIL_RISK" if alloc["multimodal_risk_pct"] >= brief["alert_thresh_pct"] else "NORMAL",
            "telemetry": {
                "india_vix": brief["latest_vix"],
                "causal_vix_threshold": econ_res["causal_ate"]["vix_threshold"],
                "garch_volatility_pct": brief["latest_garch_vol"],
                "next_day_abs_return_forecast_pct": brief["pred_abs_ret"],
                "aci_95_upper_bound_pct": brief["aci_upper_95"],
                "quant_tail_risk_prob_pct": brief["tail_risk_prob"],
                "news_fear_index": sent_res["news_fear_index"],
                "multimodal_risk_score_pct": alloc["multimodal_risk_pct"],
            },
            "allocation_sizing": alloc,
            "disclaimer": "Quantitative statistical risk telemetry only. Not SEBI-registered investment advice.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/stress-test")
def run_macro_stress_test(payload: StressTestRequest):
    try:
        df_clean = fetch_live_market_data(period="5y")
        feat_df, model_df, garch_fit = engineer_quant_features(df_clean)
        econ_res = run_econometric_and_causal_suite(model_df, garch_fit)
        ml_res = train_and_evaluate_conformal_engine(feat_df, model_df, gamma=0.05)

        sim_out = simulate_what_if_stress(
            sim_vix=payload.sim_vix,
            crude_shock_pct=payload.crude_shock_pct,
            usdinr_shock_pct=payload.usdinr_shock_pct,
            live_brief=ml_res["live_brief"],
            causal_ate=econ_res["causal_ate"],
        )
        return {
            "baseline_date": ml_res["live_brief"]["as_of_date"],
            "scenario_inputs": payload.model_dump(),
            "stress_results": sim_out,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
