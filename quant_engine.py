import contextlib
import io
import warnings
import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
import yfinance as yf
from arch import arch_model
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, root_mean_squared_error
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

warnings.filterwarnings("ignore")

TICKERS = {
    "Nifty50": "^NSEI",
    "BankNifty": "^NSEBANK",
    "IndiaVIX": "^INDIAVIX",
    "USD_INR": "INR=X",
    "CrudeOil": "BZ=F",
    "US10Y_Yield": "^TNX",
}

ML_FEATURES = [
    "GARCH_Cond_Vol", "IndiaVIX_Level", "VIX_Diff", "VIX_Diff_Lag1",
    "Vol_5d", "Vol_21d", "Vol_Ratio_5_21", "Vol_Spike_21d",
    "Nifty_Ret", "Nifty_Ret_Lag1", "Nifty_Ret_Lag2",
    "BankNifty_Ret", "Bank_Nifty_Spread", "Bank_Beta_21d",
    "USDINR_Ret", "USDINR_Z21", "Crude_Ret", "US10Y_Diff"
]

def fetch_live_market_data(period: str = "5y") -> pd.DataFrame:
    """Ingests live multi-asset and macro data from Yahoo Finance with zero look-ahead alignment."""
    raw_data = {}
    for name, symbol in TICKERS.items():
        df_temp = yf.download(symbol, period=period, interval="1d", progress=False)
        if isinstance(df_temp.columns, pd.MultiIndex):
            df_temp.columns = df_temp.columns.get_level_values(0)
        if not df_temp.empty:
            raw_data[name] = df_temp["Close"]
            if name == "Nifty50":
                raw_data["Nifty50_Volume"] = df_temp["Volume"]

    df = pd.DataFrame(raw_data)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep="first")]

    # Align to NSE Equity Calendar
    df = df.dropna(subset=["Nifty50"]).copy()
    df[["BankNifty", "IndiaVIX", "USD_INR", "CrudeOil", "US10Y_Yield"]] = (
        df[["BankNifty", "IndiaVIX", "USD_INR", "CrudeOil", "US10Y_Yield"]].ffill()
    )
    df.dropna(inplace=True)

    # Repair zero-volume trading sessions causally
    df["Nifty50_Volume"] = df["Nifty50_Volume"].replace(0, np.nan)
    df["Nifty50_Volume"] = (
        df["Nifty50_Volume"]
        .fillna(df["Nifty50_Volume"].rolling(window=5, min_periods=1).median())
        .ffill()
    )
    return df

def fetch_live_macro_news(limit: int = 15) -> list:
    """Fetches strictly current-week Indian equity & market headlines (hard-rejecting anything older than 7 days)."""
    from email.utils import parsedate_to_datetime
    headlines = []
    seen_titles = set()
    now_utc = pd.Timestamp.now(tz="UTC")

    rss_urls = [
        "https://news.google.com/rss/search?q=Nifty+50+stock+market+India&hl=en-IN&gl=IN&ceid=IN:en&scoring=n",
        "https://news.google.com/rss/search?q=Sensex+FII+DII+Indian+equities&hl=en-IN&gl=IN&ceid=IN:en&scoring=n",
        "https://news.google.com/rss/search?q=Indian+economy+rupee+inflation+markets&hl=en-IN&gl=IN&ceid=IN:en&scoring=n",
    ]

    for url in rss_urls:
        try:
            r = requests.get(url, timeout=10)
            root = ET.fromstring(r.content)
            for item in root.findall(".//item")[:12]:
                raw_title = item.find("title").text if item.find("title") is not None else ""
                clean_title = raw_title.strip()
                if not clean_title or clean_title.lower() in seen_titles:
                    continue
                if any(w in clean_title.lower() for w in ["bulletin", "economic survey", "working paper", "october", "september", "august"]):
                    continue
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                try:
                    dt = pd.Timestamp(parsedate_to_datetime(pub_date)).tz_convert("UTC")
                except Exception:
                    continue
                if (now_utc - dt).total_seconds() > 7 * 86400:
                    continue
                seen_titles.add(clean_title.lower())
                link = item.find("link").text if item.find("link") is not None else ""
                headlines.append({
                    "title": clean_title,
                    "link": link.strip(),
                    "published": pub_date,
                    "dt": dt
                })
        except Exception:
            continue

    headlines.sort(key=lambda x: x["dt"], reverse=True)
    return headlines[:limit]

def engineer_quant_features(df_clean: pd.DataFrame):
    """Builds strictly causal quantitative features, targets, and GARCH(1,1)-t volatility."""
    eda_df = pd.DataFrame(index=df_clean.index)
    eda_df["Nifty_Ret"] = np.log(df_clean["Nifty50"] / df_clean["Nifty50"].shift(1)) * 100
    eda_df["BankNifty_Ret"] = np.log(df_clean["BankNifty"] / df_clean["BankNifty"].shift(1)) * 100
    eda_df["USDINR_Ret"] = np.log(df_clean["USD_INR"] / df_clean["USD_INR"].shift(1)) * 100
    eda_df["Crude_Ret"] = np.log(df_clean["CrudeOil"] / df_clean["CrudeOil"].shift(1)) * 100
    eda_df["VIX_Diff"] = df_clean["IndiaVIX"].diff()
    eda_df["US10Y_Diff"] = df_clean["US10Y_Yield"].diff()
    eda_df["IndiaVIX_Level"] = df_clean["IndiaVIX"]
    eda_df.dropna(inplace=True)

    feat_df = eda_df.copy()
    feat_df["Vol_5d"] = feat_df["Nifty_Ret"].rolling(5).std()
    feat_df["Vol_21d"] = feat_df["Nifty_Ret"].rolling(21).std()
    feat_df["Vol_Ratio_5_21"] = feat_df["Vol_5d"] / (feat_df["Vol_21d"] + 1e-6)
    feat_df["Bank_Nifty_Spread"] = feat_df["BankNifty_Ret"] - feat_df["Nifty_Ret"]
    feat_df["Bank_Beta_21d"] = (
        feat_df["BankNifty_Ret"].rolling(21).cov(feat_df["Nifty_Ret"])
        / (feat_df["Nifty_Ret"].rolling(21).var() + 1e-6)
    )
    feat_df["USDINR_Z21"] = (
        feat_df["USDINR_Ret"] - feat_df["USDINR_Ret"].rolling(21).mean()
    ) / (feat_df["USDINR_Ret"].rolling(21).std() + 1e-6)
    feat_df["Vol_Spike_21d"] = df_clean.loc[feat_df.index, "Nifty50_Volume"] / (
        df_clean.loc[feat_df.index, "Nifty50_Volume"].rolling(21).mean() + 1e-6
    )
    feat_df["Nifty_Ret_Lag1"] = feat_df["Nifty_Ret"].shift(1)
    feat_df["Nifty_Ret_Lag2"] = feat_df["Nifty_Ret"].shift(2)
    feat_df["VIX_Diff_Lag1"] = feat_df["VIX_Diff"].shift(1)

    # Targets (t + 1)
    next_ret = feat_df["Nifty_Ret"].shift(-1)
    feat_df["Target_NextDay_AbsRet"] = next_ret.abs()
    feat_df["Target_TailRisk_Class"] = ((next_ret < -0.85) | (next_ret.abs() > 1.25)).astype(int)

    # Fit GARCH(1,1) with Student-t innovations
    valid_ret = feat_df["Nifty_Ret"].dropna()
    garch = arch_model(valid_ret, vol="Garch", p=1, q=1, dist="t", mean="Constant")
    garch_fit = garch.fit(disp="off")
    feat_df["GARCH_Cond_Vol"] = garch_fit.conditional_volatility

    model_df = feat_df.dropna().copy()
    return feat_df, model_df, garch_fit

def run_econometric_and_causal_suite(model_df: pd.DataFrame, garch_fit):
    """Runs Stationarity, Granger Causality, Double ML Causal ATE, and Precision Matrix Contagion Graph."""
    # 1. Stationarity
    stat_rows = []
    for col in ["Nifty_Ret", "BankNifty_Ret", "USDINR_Ret", "Crude_Ret", "VIX_Diff", "Target_NextDay_AbsRet"]:
        adf_stat, adf_p, _, _, _, _ = adfuller(model_df[col], autolag="AIC")
        kpss_stat, kpss_p, _, _ = kpss(model_df[col], regression="c", nlags="auto")
        stat_rows.append({
            "Variable": col,
            "ADF_Stat": round(adf_stat, 3),
            "ADF_pVal": round(adf_p, 4),
            "KPSS_Stat": round(kpss_stat, 3),
            "KPSS_pVal": round(kpss_p, 4),
            "Regime": "Stationary I(0)" if (adf_p < 0.05 and kpss_p >= 0.05) else "Volatility Regime Shift"
        })

    # 2. Granger Causality
    granger_rows = []
    for pred in ["IndiaVIX_Level", "Crude_Ret", "USDINR_Ret", "Bank_Nifty_Spread"]:
        gc_data = model_df[["Target_NextDay_AbsRet", pred]]
        with contextlib.redirect_stdout(io.StringIO()):
            gc_res = grangercausalitytests(gc_data, maxlag=3)
        p1, p2, p3 = gc_res[1][0]["ssr_ftest"][1], gc_res[2][0]["ssr_ftest"][1], gc_res[3][0]["ssr_ftest"][1]
        granger_rows.append({
            "Predictor": pred,
            "Lag1_p": round(p1, 4),
            "Lag2_p": round(p2, 4),
            "Lag3_p": round(p3, 4),
            "Significant": min(p1, p2, p3) < 0.05
        })

    # 3. Double ML Causal ATE of VIX Fear Regime on Next-Day Tail Risk
    vix_q75 = float(model_df["IndiaVIX_Level"].quantile(0.75))
    T = (model_df["IndiaVIX_Level"] > vix_q75).astype(int).values
    Y = model_df["Target_TailRisk_Class"].values
    W = model_df[["Crude_Ret", "USDINR_Ret", "US10Y_Diff", "Vol_21d", "Bank_Beta_21d"]].values

    m_Y = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(W, Y)
    m_T = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42).fit(W, T)
    Y_res = Y - m_Y.predict(W)
    T_res = T - m_T.predict(W)

    ate = float(np.sum(T_res * Y_res) / np.sum(T_res ** 2))
    se_ate = float(np.sqrt(np.mean((Y_res - ate * T_res) ** 2) / np.sum(T_res ** 2)))
    t_stat = ate / se_ate
    p_val = float(2 * (1 - stats.norm.cdf(abs(t_stat))))

    # 4. Systemic Contagion Precision Graph
    graph_cols = ["Nifty_Ret", "BankNifty_Ret", "USDINR_Ret", "Crude_Ret", "VIX_Diff", "US10Y_Diff"]
    corr_mat = model_df[graph_cols].corr().values
    prec_mat = np.linalg.inv(corr_mat)
    d = np.sqrt(np.diag(prec_mat))
    partial_corr = -prec_mat / np.outer(d, d)
    np.fill_diagonal(partial_corr, 0)

    G = nx.Graph()
    for i, node_i in enumerate(graph_cols):
        G.add_node(node_i)
        for j in range(i + 1, len(graph_cols)):
            w = partial_corr[i, j]
            if abs(w) > 0.05:
                G.add_edge(node_i, graph_cols[j], weight=round(float(w), 3), abs_weight=abs(float(w)))

    node_strength = {n: round(sum(d["abs_weight"] for _, _, d in G.edges(n, data=True)), 4) for n in G.nodes()}

    params = garch_fit.params
    return {
        "stationarity_df": pd.DataFrame(stat_rows),
        "granger_df": pd.DataFrame(granger_rows),
        "garch_summary": {
            "alpha": round(float(params["alpha[1]"]), 4),
            "beta": round(float(params["beta[1]"]), 4),
            "persistence": round(float(params["alpha[1]"] + params["beta[1]"]), 4),
            "nu_dof": round(float(params["nu"]), 2),
        },
        "causal_ate": {
            "vix_threshold": round(vix_q75, 2),
            "ate_pct": round(ate * 100, 2),
            "se_pct": round(se_ate * 100, 2),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 4),
        },
        "graph": G,
        "node_strength": node_strength,
    }

def train_and_evaluate_conformal_engine(feat_df: pd.DataFrame, model_df: pd.DataFrame, gamma: float = 0.05):
    """Chronological 3-Way Split + Adaptive Conformal Inference (ACI) + Tail Risk Classifier + Live Forecast."""
    X = model_df[ML_FEATURES].copy()
    y_reg = model_df["Target_NextDay_AbsRet"].values
    y_clf = model_df["Target_TailRisk_Class"].values
    dates = model_df.index

    n = len(X)
    train_end = int(n * 0.70)
    cal_end = int(n * 0.85)

    X_train, y_reg_train, y_clf_train = X.iloc[:train_end], y_reg[:train_end], y_clf[:train_end]
    X_cal, y_reg_cal, y_clf_cal = X.iloc[train_end:cal_end], y_reg[train_end:cal_end], y_clf[train_end:cal_end]
    X_test, y_reg_test, y_clf_test = X.iloc[cal_end:], y_reg[cal_end:], y_clf[cal_end:]
    dates_test = dates[cal_end:]

    # 1. Volatility Regressor
    reg_model = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.02, max_depth=3, min_samples_leaf=25, l2_regularization=5.0, random_state=42
    )
    reg_model.fit(X_train, y_reg_train)

    cal_preds = np.maximum(0, reg_model.predict(X_cal))
    nonconformity_scores = np.abs(y_reg_cal - cal_preds)

    n_cal = len(X_cal)
    q95_level = min(np.ceil((n_cal + 1) * 0.95) / n_cal, 1.0)
    q95_static = float(np.quantile(nonconformity_scores, q95_level, method="higher"))

    test_reg_preds = np.maximum(0, reg_model.predict(X_test))
    static_cov_95 = float(np.mean((y_reg_test >= np.maximum(0, test_reg_preds - q95_static)) &
                                  (y_reg_test <= (test_reg_preds + q95_static))) * 100)

    # 2. Online Adaptive Conformal Inference (ACI)
    alpha_target = 0.05
    alpha_t = alpha_target
    aci_lower_95, aci_upper_95, aci_hits = [], [], []
    score_history = list(nonconformity_scores)

    for t in range(len(y_reg_test)):
        recent_scores = np.array(score_history[-90:])
        q_adaptive = float(np.quantile(recent_scores, np.clip(1.0 - alpha_t, 0.01, 0.999)))
        pred_t = float(test_reg_preds[t])
        low_t = max(0.0, pred_t - q_adaptive)
        high_t = pred_t + q_adaptive

        aci_lower_95.append(low_t)
        aci_upper_95.append(high_t)
        hit = int((y_reg_test[t] >= low_t) and (y_reg_test[t] <= high_t))
        aci_hits.append(hit)

        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - (1 - hit)), 0.005, 0.50))
        score_history.append(abs(float(y_reg_test[t]) - pred_t))

    aci_cov_95 = float(np.mean(aci_hits) * 100)
    oos_rmse = float(root_mean_squared_error(y_reg_test, test_reg_preds))

    # 3. Tail Risk Classifier
    X_train_full = pd.concat([X_train, X_cal])
    y_clf_train_full = np.concatenate([y_clf_train, y_clf_cal])
    clf_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=30, max_features=0.5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf_model.fit(X_train_full, y_clf_train_full)

    cal_probs = clf_model.predict_proba(X_cal)[:, 1]
    fpr_c, tpr_c, thresh_c = roc_curve(y_clf_cal, cal_probs)
    opt_thresh = float(thresh_c[np.argmax(tpr_c - fpr_c)])

    test_clf_probs = clf_model.predict_proba(X_test)[:, 1]
    oos_auc = float(roc_auc_score(y_clf_test, test_clf_probs))
    oos_brier = float(brier_score_loss(y_clf_test, test_clf_probs))

    # 4. Live Inference on Latest Available Trading Day
    latest_row = feat_df.iloc[[-1]].copy()
    X_live = latest_row[ML_FEATURES]
    live_vol_pred = max(0.0, float(reg_model.predict(X_live)[0]))
    live_q_adaptive = float(np.quantile(np.array(score_history[-90:]), np.clip(1.0 - alpha_t, 0.01, 0.999)))
    live_upper_95 = live_vol_pred + live_q_adaptive
    live_tail_prob = float(clf_model.predict_proba(X_live)[0, 1])

    return {
        "dates_test": dates_test,
        "y_reg_test": y_reg_test,
        "test_reg_preds": test_reg_preds,
        "aci_lower_95": np.array(aci_lower_95),
        "aci_upper_95": np.array(aci_upper_95),
        "y_clf_test": y_clf_test,
        "test_clf_probs": test_clf_probs,
        "metrics": {
            "oos_rmse": round(oos_rmse, 4),
            "static_cov_95": round(static_cov_95, 2),
            "aci_cov_95": round(aci_cov_95, 2),
            "oos_auc": round(oos_auc, 4),
            "oos_brier": round(oos_brier, 4),
            "opt_thresh": round(opt_thresh, 4),
        },
        "live_brief": {
            "as_of_date": str(latest_row.index[0].date()),
            "latest_vix": round(float(latest_row["IndiaVIX_Level"].iloc[0]), 2),
            "latest_garch_vol": round(float(latest_row["GARCH_Cond_Vol"].iloc[0]), 4),
            "pred_abs_ret": round(live_vol_pred, 4),
            "aci_upper_95": round(live_upper_95, 4),
            "tail_risk_prob": round(live_tail_prob * 100, 2),
            "alert_thresh_pct": round(opt_thresh * 100, 2),
        }
    }
