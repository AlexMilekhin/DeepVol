# DeepVol: Neural Volatility Surface Modeling & Forecasting

**DeepVol** is a comprehensive quantitative finance library designed for high-fidelity volatility surface modeling, risk-neutral density (RND) extraction, and predictive analytics using deep learning. This project bridges classical stochastic modeling with modern machine learning architectures, specifically leveraging a **Mixture-of-Experts (MoE)** network to forecast implied volatility dynamics.

## 🌟 Key Features

* **Arbitrage-Free Surface Calibration**: Implementation of SVI (Slice) and global SSVI (Surface) models with strict adherence to no-arbitrage constraints.
* **Risk-Neutral Density (RND) Extraction**: Derivation of probability density functions (PDF) via the Breeden-Litzenberger identity, providing higher-order moments (Skewness, Kurtosis) for tail-risk analysis.
* **Neural Volatility Forecasting**: A regime-aware Mixture-of-Experts (MoE) architecture that specializes in predicting 10-day changes in total variance.
* **Automated Data Pipeline**: End-to-end orchestration from live market data ingestion (Yahoo Finance) to feature engineering and Parquet-based persistence.
* **Vectorized Engineering**: High-throughput Black-Scholes pricing and IV inversion using optimized NumPy operations and Brent’s method.

---

## 🏗 Quantitative Engine

### 1. Volatility Surface Modeling (`src/models/`)

The framework supports two primary calibration routines to ensure a smooth, arbitrage-free surface:

* **SVI (Stochastic Volatility Inspired)**: Calibrates individual expiry slices using the Gatheral (2004) parameterization:

$$w(k) = a + b(\rho(k-m) + \sqrt{(k-m)^2 + \sigma^2})$$

* **Global SSVI (Surface SVI)**: Fits the entire surface simultaneously to ensure term-structure consistency and eliminate calendar arbitrage:

$$w(k, \theta) = \frac{\theta}{2} \left[ 1 + \rho \phi(\theta) k + \sqrt{(\phi(\theta) k + \rho)^2 + (1 - \rho^2)} \right]$$

with the power-law function $$\phi(\theta) = \eta \theta^{-\gamma}$$

### 2. Risk-Neutral Density Extraction (`src/features.py`)

Utilizing the Breeden-Litzenberger identity, the library reconstructs the market-implied PDF from the calibrated surface:

$$f(K) = e^{rT} \frac{\partial^2 C}{\partial K^2}$$

The module performs numerical integration via Simpson’s rule on a 5,000-point grid to extract:

* **Skewness & Kurtosis**: Quantifying market-implied asymmetry and tail risk.
* **Martingale Diagnostic**: Monitoring the error  to ensure model consistency with the forward price. $$|E^{RN}[S_T] - F|$$

---

## 🧠 Mixture-of-Experts (MoE) Forecasting

The forecasting engine utilizes a specialized deep learning architecture to predict the **10-trading-day change in total variance** (h=10):

$$y_t = w_{t+h} - w_t$$

### Architecture & Design

* **LSTM Encoder**: Processes historical sequences of volatility and regime features.
* **Regime-Aware Gating**: A gating network maps input features (e.g., IV z-scores, momentum, term-structure slope) to mixture weights over **4 specialized expert heads**.
* **Feature Enrichment**: Includes rolling z-scores, IV momentum, and volatility-of-volatility (vol-of-vol) indicators to capture shifting market regimes.
* **Physics-Informed Sample Weighting**: Employs a "physics-based weight-decay" mechanism that penalizes the loss function for training samples exhibiting high SVI/SSVI fit costs, calendar arbitrage violations, or RND martingale errors.
* **Benchmark Performance**: The MoE model achieves a **7.7% MAE improvement** over the 
σ -persistence baseline.

---

## 📂 Project Structure

```text
.
├── src/
│   ├── models/
│   │   ├── bs_engine.py        # Vectorized BS pricing & Brent IV inversion
│   │   └── svi_fit.py          # SVI and SSVI surface fitting logic
│   ├── features.py             # RND moment extraction (skew, kurtosis)
│   └── data_loader.py          # Live data ingestion & cleaning
├── main.py                     # Live production pipeline orchestrator
├── historical_pipeline.py      # Historical backtesting & dataset generation
├── moe_feature_enrichment.py   # ML feature engineering (regime indicators)
├── notebooks/                  # Research & development walkthroughs
├── moe_forecaster.ipynb        #model notebook
├── moe_forecaster.pt           #model
├── batch_process_tickers.py    # script for multiple-ticker processing in historical pipeline
├──setup.py
├── requirements.txt
└── README.md

```

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/alexmilekhin/deepvol.git
cd deepvol
pip install -e .

```

### Running the Live Pipeline

Calibrate the surface and extract features for a specific ticker.

```bash
python main.py --ticker {TICKER} --data-dir ./data --metrics-dir ./results

```

The pipeline automatically fetches the current 13-week Treasury yield (`^IRX`) as the risk-free rate proxy.

### Training Data Generation

Process historical data to generate enriched feature sets for ML training:

```bash
python historical_pipeline.py --file data/options_data.parquet --ticker {TICKER}
python moe_feature_enrichment.py --input historical_results/historical_svi_{TICKER}.parquet

```

---

## 📝 License & Author

**Author**: Alexander Milekhin — milekhin.alexander@gmail.com

Distributed under the MIT License. See `LICENSE` for more information.