import os
from quant_engine import (
    compute_portfolio_risk_allocation,
    engineer_quant_features,
    fetch_live_macro_news,
    fetch_live_market_data,
    run_econometric_and_causal_suite,
    score_news_sentiment,
    send_telegram_risk_alert,
    train_and_evaluate_conformal_engine,
)

if __name__ == "__main__":
    df_clean = fetch_live_market_data(period="5y")
    feat_df, model_df, garch_fit = engineer_quant_features(df_clean)
    econ_res = run_econometric_and_causal_suite(model_df, garch_fit)
    ml_res = train_and_evaluate_conformal_engine(feat_df, model_df, gamma=0.05)
    sent_res = score_news_sentiment(fetch_live_macro_news(limit=12))
    alloc = compute_portfolio_risk_allocation(500000.0, 1.0, ml_res["live_brief"], sent_res["news_fear_index"])

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    ok, msg = send_telegram_risk_alert(token, chat_id, ml_res["live_brief"], alloc)
    print("Morning Alert Status:", ok, msg)
