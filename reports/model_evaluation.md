# AgriYield AI — Model Evaluation & Performance Report

## Executive Summary

- **Champion Model**: `HistGradientBoosting`
- **Evaluation Methodology**: Chronological Out-of-Time Validation (Train: 1997–2011, Test: 2012–2015)
- **Training Records**: 202,487
- **Test Records (Unseen Historical Future)**: 37,726
- **Target Variable**: Agricultural Crop Yield ($	ext{Tonnes / Hectare}$)

---

## 1. Candidate Model Benchmarking

Models were trained on pre-harvest agricultural features (`State_Name`, `District_Name`, `Season`, `Crop`, `Crop_Category`, `Farm_Scale`, `Log_Area`, `Year_Index`, `Is_Kharif_Monsoon`, `Hist_Crop_State_Yield`) without post-harvest target leakage.

| Model Candidate | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | Training Time (s) | Evaluation Note |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Mean)** | 5.6576 | 12.5961 | -0.0027 | 0.0 | Unconditioned sample mean |
| **Ridge Regression** | 1.3765 | 4.5257 | 0.8706 | 0.35 | Linear regularized with standard scaling |
| **Random Forest Regressor** | 1.2434 | 4.5536 | 0.869 | 5.88 | 60 trees, max depth 14, min leaf 10 |
| **HistGradientBoosting** | **1.2562** | **4.5183** | **0.871** | 5.96 | **Champion**: Non-linear boosting with categorical handling |

---

## 2. Performance Breakdown by Major Crops (Top Test Sample Sizes)

| Crop | Test Samples | Mean Actual Yield (t/ha) | Mean Predicted Yield (t/ha) | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Rice | 2,490 | 2.304 | 2.155 | 0.5382 | 0.7314 | 0.3814 |
| Maize | 2,429 | 2.639 | 2.427 | 0.7374 | 1.0862 | 0.4694 |
| Moong(Green Gram) | 1,893 | 0.519 | 0.522 | 0.152 | 0.215 | 0.3584 |
| Urad | 1,730 | 0.556 | 0.544 | 0.1398 | 0.1905 | 0.4268 |
| Groundnut | 1,411 | 1.361 | 1.308 | 0.3689 | 0.5294 | 0.4574 |
| Sesamum | 1,340 | 0.739 | 0.461 | 0.4289 | 3.7739 | 0.0026 |
| Sugarcane | 1,229 | 57.873 | 56.731 | 12.8607 | 20.6393 | 0.5162 |
| Wheat | 1,139 | 2.272 | 2.329 | 0.4799 | 0.6464 | 0.6452 |
| Arhar/Tur | 1,138 | 0.872 | 0.869 | 0.26 | 0.3889 | 0.308 |
| Onion | 1,132 | 12.841 | 12.749 | 2.8305 | 4.1609 | 0.6816 |
| Rapeseed &Mustard | 1,103 | 0.816 | 0.808 | 0.2234 | 0.3218 | 0.5239 |
| Potato | 1,082 | 13.817 | 13.143 | 3.0353 | 4.1595 | 0.6812 |
| Gram | 991 | 0.922 | 0.924 | 0.2801 | 0.4127 | 0.144 |
| Dry chillies | 959 | 1.764 | 1.624 | 0.678 | 1.3401 | 0.4609 |
| Jowar | 921 | 1.259 | 1.229 | 0.4365 | 0.6966 | 0.2081 |

---

## 3. Performance Breakdown Across Agricultural Seasons

| Season | Test Samples | Mean Actual Yield (t/ha) | Mean Predicted Yield (t/ha) | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Kharif | 14,986 | 3.695 | 3.452 | 0.8833 | 2.6292 | 0.9316 |
| Rabi | 11,707 | 2.949 | 2.922 | 0.7019 | 1.9033 | 0.8683 |
| Whole Year | 5,002 | 14.249 | 13.678 | 4.1127 | 10.6513 | 0.8097 |
| Summer | 3,473 | 3.85 | 3.549 | 0.7648 | 2.6327 | 0.884 |
| Winter | 1,449 | 6.792 | 6.296 | 1.458 | 4.6375 | 0.9253 |
| Autumn | 1,109 | 1.474 | 1.478 | 0.5383 | 1.1514 | -0.1877 |

---

## 4. Residual Diagnostics & Error Distributions

- **Mean Residual ($y - \hat{y}$)**: `0.2276` t/ha (Minimal systemic bias)
- **Median Residual**: `-0.0298` t/ha
- **Residual Standard Deviation**: `4.5126` t/ha
- **10th Percentile Error**: `-0.7647` t/ha
- **90th Percentile Error**: `1.2177` t/ha

---

## 5. Agronomic & Methodological Insights

1. **Chronological Validity**: By partitioning training data up to 2011 and evaluating exclusively on 2012–2015, the model demonstrates genuine predictive capacity for forward-looking harvest seasons rather than interpolated cross-validation.
2. **Grain vs High-Biomass Crops**: Food grains like Rice and Wheat demonstrate tight MAE bounds (~0.4 to 0.6 tonnes/ha). Higher absolute errors in crops like Sugarcane and Banana reflect their natural biological yield range (50–100+ tonnes/ha) rather than model degradation.
3. **No Target Leakage**: Sown area (`Area`) and pre-harvest agro-climatic categories are preserved as valid predictors, while post-outcome variables (`Production`) are strictly excluded.
