# AgriYield AI — Scientific Methodology & Machine Learning Design

## 1. Problem Formulation

### 1.1 Prediction Unit
The fundamental prediction unit is defined as:
$$\text{Yield} = \frac{\text{Production}}{\text{Area}} \quad (\text{Metric Tonnes / Hectare})$$
for a given **District-Crop-Season-Year** observation.

### 1.2 Operational Timing & Pre-Harvest Information Availability
In agribusiness and agricultural policy, predictions are required **at planting or early vegetative stages** to guide input procurement, credit disbursement, and storage logistics.

- **Available Pre-Harvest Information**:
  - Administrative Region (`State_Name`, `District_Name`)
  - Sowing Season (`Season`: Kharif, Rabi, Summer, Whole Year)
  - Target Commodity (`Crop`, `Crop_Category`)
  - Cultivated Acreage (`Area`, `Log_Area`, `Farm_Scale`)
  - Calendar Cropping Year (`Crop_Year`, `Year_Index`)
  - Historical Agronomic Baseline (`Hist_Crop_State_Yield` computed strictly from prior years)

- **Post-Outcome Variables (Strictly Excluded)**:
  - Total Harvest Output (`Production`): Knowing production directly calculates yield, causing total target leakage.
  - Same-Year Regional Yields: Future or concurrent regional averages are excluded to preserve genuine out-of-time forecasting.

---

## 2. Validation Strategy: Chronological Partitioning

### 2.1 The Fallacy of Random K-Fold Splitting in Agriculture
In time-series agricultural statistics, random K-Fold cross-validation produces artificially optimistic error metrics due to **temporal leakage**:
- The model trains on 2014 data to predict 2012 yields.
- Autocorrelated weather shocks and macro-policy changes leak across folds.

### 2.2 Chronological Out-of-Time Protocol
AgriYield AI implements a strict chronological split:
- **Training Set**: 1997–2011 (15 crop years, 202,487 observations, 84.3% of data)
- **Out-of-Time Test Set**: 2012–2015 (4 unseen future crop years, 37,726 observations, 15.7% of data)

This protocol directly evaluates how well a model trained on historical data forecasts future harvest seasons.

---

## 3. Feature Engineering Formulations

### 3.1 Log-Transformed Cultivated Area
Cultivated acreage exhibits extreme positive skewness (0.04 ha to 8,000,000 ha). We apply natural log transformation:
$$\text{Log\_Area} = \ln(\text{Area} + 1)$$
This dampens extreme leverage while preserving operational ordering.

### 3.2 Operational Landholding Scale
Landholdings are categorized into Ministry of Agriculture operational brackets:
$$\text{Scale} = \begin{cases} 
\text{Marginal}, & \text{Area} < 1.0\text{ ha} \\
\text{Small}, & 1.0 \le \text{Area} < 2.0\text{ ha} \\
\text{Semi-Medium}, & 2.0 \le \text{Area} < 4.0\text{ ha} \\
\text{Medium}, & 4.0 \le \text{Area} < 10.0\text{ ha} \\
\text{Large}, & \text{Area} \ge 10.0\text{ ha}
\end{cases}$$

### 3.3 Leak-Free Historical Benchmark
For each crop $c$ and state $s$, the historical benchmark is calculated strictly using training records ($\text{Year} \le 2011$):
$$\bar{Y}_{s,c} = \frac{1}{|T_{s,c}|} \sum_{i \in T_{s,c}} Y_i, \quad T_{s,c} = \{i : \text{State}_i = s, \text{Crop}_i = c, \text{Year}_i \le 2011\}$$
For test instances where the state-crop pair was not observed in the training period, fallback heirarchies (Crop National Mean $\to$ Global Mean) are applied.

---

## 4. Model Architectures & Selection

1. **Baseline Regressor (Sample Mean)**:
   - $\hat{y} = \bar{y}_{\text{train}} = 4.28\text{ t/ha}$.
   - Serves as the zero-intelligence benchmark ($R^2 \approx 0$).

2. **Ridge Regression**:
   - $\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2$
   - Combines `StandardScaler` on numerical features and `OrdinalEncoder` / one-hot representation for linear regularized modeling.

3. **Random Forest Regressor**:
   - Non-linear ensemble of 60 decision trees with `max_depth=14`, `min_samples_leaf=10`.
   - Models non-linear interactions between crop categories, landholding sizes, and regional baselines.

4. **HistGradientBoostingRegressor (Champion)**:
   - Modern histogram-based gradient boosted decision trees.
   - Built-in categorical feature handling, high sample capacity, and robust resistance to noise.
   - Attains the lowest test error ($\text{MAE} = 1.256\text{ t/ha}$, $\text{RMSE} = 4.518\text{ t/ha}$, $R^2 = 0.871$).

---

## 5. Metric Selection Rationale

- **MAE (Mean Absolute Error)**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$. Directly interpretable in metric tonnes per hectare; less sensitive to extreme legitimate crop harvests.
- **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$. Penalizes large prediction errors heavily, highlighting high-stakes under- or over-predictions.
- **$R^2$ Score (Coefficient of Determination)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Quantifies the proportion of historical variance explained by the model over the baseline.
