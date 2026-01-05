# Volatility Surface Modeling & ML Forecasting

A quantitative finance library for modeling equity volatility surfaces, extracting risk-neutral densities (RND), and forecasting implied volatility using deep learning architectures. This project combines classical quantitative finance theory with modern machine learning architectures.

## 🌟 Key Features

* **Surface Calibration**: Implements SVI and global SSVI (Surface SVI) fitting with strict adherence to no-arbitrage constraints.
* **Risk-Neutral Density (RND) Extraction**: Derives probability density functions from the volatility surface via the Breeden-Litzenberger identity, extracting Skewness, Kurtosis, and Martingale diagnostic metrics.
* **Predictive Analytics**: Utilizes an LSTM + Self-Attention network to forecast implied volatility dynamics, weighting training samples by market liquidity metrics.
* **Automated Production Pipeline**: End-to-end orchestration from live market data ingestion (Yahoo Finance/OpenBB) to feature engineering and metric persistence in Parquet format.

## 🏗 System Architecture & Design Choices

### 1. Quantitative Modeling Engine (`src/models/`)
* **Vectorized Greeks & IV**: The `ImpliedVolatilityCalculator` utilizes vectorized NumPy operations for high-throughput pricing. Implied Volatility inversion is implemented via Brent's method for numerical stability over a wide range of moneyness.
* **No-Arbitrage Constraints**: The `SSVIFitter` enforces the Gatheral/Jacquier conditions ($\eta(1+|\rho|) \leq 2$) and utilizes monotone theta enforcement across the term structure to eliminate calendar arbitrage.
* **RND Feature Engineering**: Computes RND mean/variance/skew/kurtosis via Simpson-rule integration of a Breeden–Litzenberger density implied by a smoothed SSVI surface on a 5,000-point strike grid.

### 2. Deep Learning Architecture (`src/models/forecaster.py`)
**LSTM + temporal self-attention** model for per-expiry ATM IV forecasting, using ATM/skew/curvature + SVI/SSVI params + RND moments. Trained with weighted MSE and arbitrage/consistency penalties (martingale error, fit cost, calendar violations); Adam + LR decay, grad clipping, early stopping; evaluated with MSE/MAE vs naïve baseline.

### 3. Engineering Patterns (`main.py`)
* **Dynamic Risk-Free Rates**: The pipeline automatically fetches the current 13-week Treasury Bill yield (`^IRX`) to ensure the pricing model uses the most recent market-implied risk-free rate.
* **Modular Data Handling**: Separates data collection, cleaning, and modeling into distinct stages. All intermediate outputs are stored as Parquet files to preserve schema integrity and optimize I/O performance.

## 📂 Project Structure

├── data/
├── metrics/
├── notebooks/
├── src/
│   ├── data_loader.py      # Live data ingestion & Parquet persistence
│   ├── features.py         # RND moment extraction (Skew, Kurtosis) via Breeden-Litzenberger        
|   ├── models.py
│   └── models/
│       ├── bs_engine.py    # Data cleaning, vectorized BS pricing & Brent IV inversion
│       ├── svi_fit.py      # SVI and SSVI surface fitting
│       └── forecaster.py   # LSTM-Attention forecasting with weighted loss
├── main.py                 # CLI Pipeline orchestrator
├── setup.py                # Package installation configuration
├── requirements.txt
├── viewing.py              # Data quality check


## 🚀 Getting Started

### Installation

```bash
pip install -e .
```

### Running the Pipeline (Example: QQQ)

Execute the full suite—from data collection and surface calibration to RND feature extraction—with a single command:

```bash
python main.py --ticker QQQ --data-dir ./data --metrics-dir ./results
```

### Future Improvements

## Conditional diffusion model ## 
Hybrid Forecasting: Combine Self-Attention LSTMs for temporal priors with Conditional U-Nets for spatial refinement.

Structural Fidelity: Use 5x5 kernels, Reflection Padding, and Bilinear Upsampling to ensure surface smoothness and eliminate localized artifacts.

Market Conditioning: Injects SSVI parameters and Risk-Neutral Density (RND) metrics via FiLM blocks to adapt generation to specific volatility regimes.

Physics-Informed Constraints: Employs a specialized loss function to penalize Calendar and Butterfly arbitrage violations, ensuring financial validity

## MoE transformer ##

## 📝 License

Distributed under the MIT License.

## Author
Alexander Milekhin milekhin.alexander@gmail.com

