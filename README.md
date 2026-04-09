# 🛰️ Soil Moisture Prediction via Sentinel-2: Hybrid PCA Ensemble

This repository hosts a state-of-the-art predictive pipeline for soil moisture estimation from high-resolution Sentinel-2 satellite imagery. By combining multi-spectral physics-based feature engineering with a hybrid dimensionality reduction approach, this model overcomes the data-scarcity limitations of traditional Deep Learning.

## 🚀 The Flagship Model: `Hybrid_best_model.ipynb`

While this repository contains multiple experimental baselines (CNNs, ConvGRUs, etc.), the **Hybrid PCA Ensemble** is our top-performing solution, specifically engineered to maximize $R^2$ on datasets with ~1,000 samples.

### 🧠 Methodology Summary
1.  **Massive Feature Engineering**:
    *   **Spectral Physics**: Extraction of 12+ critical indices (NDVI, NDMI, NDWI, MNDWI, SAVI, BSI, EVI, NBR, NERC, MTCI) directly correlated with surface water content.
    *   **Multi-Scale Statistics**: Capturing spatial context across 31x31, 7x7, and 3x3 grids around the ground-truth probe.
    *   **Temporal Context**: Incorporation of smoothed target encoding, historical moisture lags, and trigonometric seasonal encoding.
2.  **Advanced Hybrid Reduction**:
    *   **High-Impact Selection**: Automatic retention of features with >0.001 Gini importance.
    *   **Latent Variance Capture**: Remaining variance is compressed into principal components (PCA) to maintain 95% of the spectral information while eliminating collinearity.
3.  **Tuned Ensemble**:
    *   A weighted **Voting Regressor** blending `HistGradientBoosting`, `ExtraTrees`, and `RandomForest`.
    *   Automated hyperparameter optimization via `TimeSeriesSplit` and `GrideSearch` to ensure zero temporal leakage.

### 📈 Performance Metrics
*   **Coefficient of Determination ($R^2$):** ~0.87 (Test Set)
*   **Root mean square error (RMSE):** < 0.025
*   **Pearson Correlation:** ~0.93

---

## 📁 Repository Structure

- **`notebooks/`**:
    - `Hybrid_best_model.ipynb`: **The Best Model**. Implements the Hybrid PCA methodology.
    - `Soil_Moisture_CNNLSTM.ipynb`: Exploration of temporal CNN-LSTM architectures.
    - `leakage_safe_cnn_temporal.ipynb`: Colab notebook for the ConvGRU baseline.
    - `Data_exploration_image.ipynb`: Image distribution and spectral index analysis.
- **`src/`**: Modular source code for baseline generation and model definitions.
- **`docs/`**: Supplemental architectural documentation and legacy READMEs.
- **`data/`**: (Git-ignored) Placeholder for local dataset storage.

---

## 🛠️ Setup & Usage

1. **Environment**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Placement**: 
   Place your 31x31 `.tif` patches in `data/All_3_areas_cropped/` and your ground truth file in `data/Combined_Area1_Area2_Area3_Area5_Area6_with_tif.csv`.
3. **Execution**:
   Open `notebooks/Hybrid_best_model.ipynb` in Google Colab or a local Jupyter server and run all cells.

---

## 🛡️ Leakage Prevention
This project implements strict **Temporal Embargo** gaps. Training, Validation, and Test sets are split chronologically with one-sample buffers ensuring the model never "sees the future" during training.
