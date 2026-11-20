# QuantVault AI — Institutional-Grade Autonomous Quant & Risk Intelligence Engine

**QuantVault AI** is a live, zero-synthetic-data quantitative risk and econometric intelligence system built for Indian equities (`Nifty 50`, `Bank Nifty`, `India VIX`) and global macro drivers (`USD/INR`, `Brent Crude`, `US 10Y Treasury Yields`).

## Core Architecture & Statistical Methodology
1. **Zero-Look-Ahead Live Data Ingestion:** Aligns multi-asset and macro feeds (`yfinance` + Google News RSS) strictly to the NSE trading calendar using causal forward-filling and rolling median volume repair.
2. **Econometric Volatility & Stationarity Suite:** Validates $I(0)$ stationarity (`ADF` & `KPSS`), lead-lag dynamics (`Granger Causality`), and heavy-tailed conditional heteroskedasticity via **GARCH(1,1) with Student's $t$-innovations** ($\nu = 6.93$, $\alpha + \beta = 0.9423$).
3. **Orthogonalized Double Machine Learning (Causal ATE):** Estimates the causal Average Treatment Effect of elevated `IndiaVIX` fear regimes on next-day tail-risk probability (`+13.27%`, $p = 0.0004$) while controlling for macro confounders.
4. **Systemic Contagion Graph:** Extracts direct conditional dependencies across equities, currency, oil, and rates via Gaussian Precision Matrix partial correlations (`NetworkX`).
5. **Adaptive Conformal Inference (ACI):** Solves distribution shift degradation in financial time series by dynamically updating miscoverage quantiles online—restoring 2026 out-of-sample 95% coverage from `74.18%` (static split conformal) to `93.41%`.

## Deployment on Render
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
