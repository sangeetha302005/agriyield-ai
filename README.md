# 🌾 AgriYield AI — Agricultural Crop Yield Prediction and Agribusiness Decision Support System

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agriyield-ai-302005.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2D5A3F?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/PyTest-11%20passed-brightgreen.svg)](tests/)
[![Data Source](https://img.shields.io/badge/Data%20Source-data.gov.in%20(DES)-1E3A2B)](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)

> **🌐 Live Web Application:** [https://agriyield-ai-302005.streamlit.app/](https://agriyield-ai-302005.streamlit.app/)  
> **📂 GitHub Repository:** [https://github.com/sangeetha302005/agriyield-ai](https://github.com/sangeetha302005/agriyield-ai)

**AgriYield AI** is an end-to-end, production-grade agricultural analytics and machine learning application engineered on **19 years of authentic Government of India statistics** (246,091 records). It provides automated data auditing, domain-grounded outlier detection, leak-free feature engineering, chronological model evaluation ($R^2 = 0.871$), and an interactive decision-support platform with custom CSV/Excel upload visualization capabilities.

---

## 📌 Table of Contents
- [Executive Overview & Value Proposition](#-executive-overview--value-proposition)
- [Live Application Workflows](#-live-application-workflows)
- [The Agricultural Problem & Motivation](#-the-agricultural-problem--motivation)
- [System Architecture](#-system-architecture)
- [Official Dataset & Provenance](#-official-dataset--provenance)
- [Data Pipeline & Quality Engineering](#-data-pipeline--quality-engineering)
- [Feature Engineering & Anti-Leakage Protocol](#-feature-engineering--anti-leakage-protocol)
- [Machine Learning Benchmarks & Evaluation](#-machine-learning-benchmarks--evaluation)
- [Agribusiness & Policy Implications](#-agribusiness--policy-implications)
- [Repository Structure](#-repository-structure)
- [Installation & Local Quick Start](#-installation--local-quick-start)
- [Reproducibility & Testing](#-reproducibility--testing)
- [Docker Deployment](#-docker-deployment)
- [Assumptions, Limitations & Responsible Use](#-assumptions-limitations--responsible-use)
- [Future Roadmap](#-future-roadmap)
- [Citation & Acknowledgments](#-citation--acknowledgments)

---

## 🌟 Executive Overview & Value Proposition

In emerging agrarian economies, agricultural yield predictions are commonly treated as trivial cross-validation toy problems using arbitrary synthetic splits. In practice, real-world deployment faces severe obstacles:
1. **Target Leakage**: Models inadvertently use post-harvest total production or concurrent-year indicators to predict yield ($Y = \frac{P}{A}$), rendering models useless at planting time.
2. **Temporal Leakage**: Random K-Fold cross-validation trains on future records (e.g. 2014) to predict past harvests (e.g. 2010), producing inflated metrics that collapse during deployment.
3. **Unit Heterogeneity**: Reporting anomalies (such as Coconut production recorded in *thousand nuts* while grains are recorded in *metric tonnes*) distort regression loss functions if left unaddressed.

**AgriYield AI** demonstrates how senior machine learning engineering and domain agronomy resolve these issues:
- **Zero Target Leakage**: Yield is predicted strictly using pre-harvest attributes (`State`, `District`, `Season`, `Crop`, `Crop_Category`, `Farm_Scale`, `Log_Area`, and leak-free expanding historical benchmarks).
- **Chronological Out-of-Time Validation**: Models are trained on harvests up to **2011 (202,487 records)** and evaluated strictly on unseen future harvest years **2012–2015 (37,726 records)**.
- **Champion Performance**: A gradient boosted pipeline (`HistGradientBoostingRegressor`) achieves **$R^2 = 0.871$**, reducing Mean Absolute Error to **1.256 Tonnes/Ha** across 124 commodities.
- **Enterprise Dataset Ingestion**: Upload custom farm CSV/Excel files to automatically profile data health, compute yields, and generate interactive Plotly visual charts instantly.

---

## 🖥️ Live Application Workflows

The live web application ([https://agriyield-ai-302005.streamlit.app/](https://agriyield-ai-302005.streamlit.app/)) is structured into three clean, dedicated workflows:

### 1. 🌾 Home Page
- **Brand Identity**: Agricultural intelligence platform with modern, responsive styling.
- **Dynamic KPI Summary Cards**: Real-time record counts (246k+), commodities (124), states (33), and national average yield ($2.73$ t/ha) computed directly from the verified dataset.
- **Empirical Takeaways**: Key agricultural insights highlighting cereal productivity gaps, monsoon rainfall exposure, and horticulture supply volatility.

### 2. 🏛️ Government Data & Yield AI (Completely Unified)
- **Sub-Tab 1: 📊 Historical Data Analytics & Trends**
  - Multi-decade historical yield and acreage trends (1997–2015).
  - Dynamic slice & dice filters by Crop, State, Season, and Metric Aggregate.
  - Regional productivity rankings (Top 10 producing states).
  - Seasonal harvest distributions (Box plots with median, IQR, and variance).
  - National commodity production rankings across 124 crops.
- **Sub-Tab 2: 🔮 Pre-Harvest Crop Yield Prediction AI Engine**
  - Interactive inference form: `State`, `District` (dynamically filtered), `Crop`, `Season`, `Sown Area (Ha)`, and `Target Year`.
  - Real-time yield forecast (Tonnes/Ha) with **80% Confidence Intervals**.
  - Expected gross harvest production (Tonnes) with confidence bounds.
  - Operational Farm Scale classification (Marginal, Small, Medium, Large) and agronomic context.
  - Champion model validation scorecards ($R^2 = 0.871$, MAE = 1.256 t/ha, RMSE = 4.518 t/ha).

### 3. 📤 Upload & Analyse Your Data (Instant Visual Analytics)
- **Multi-Format Support**: Ingest `.csv`, `.xlsx`, or `.xls` files directly (or 1-click test with preloaded sample farm data).
- **Automated Column & Yield Discovery**: Automatically detects numerical/categorical columns and computes $\text{Yield} = \frac{\text{Production}}{\text{Area}}$ if acreage and harvest output are found.
- **Automated Visual Graphs**:
  - *Category Performance Bar Chart*: Top categories ranked with formatted value labels and green gradient styling.
  - *Statistical Distribution Box Plot*: Median, quartiles, and outliers for yield/production.
  - *Feature Correlation Scatter*: Scatter plot with category color coding (e.g. Area vs. Production).
  - *Chronological Progression*: Line chart tracking trends over time.
- **Interactive Custom Chart Explorer**: Build bespoke visualizations choosing Chart Type (*Bar, Line, Scatter, Box, Histogram, Pie/Donut*), X-Axis, Y-Axis, Color Grouping, and Aggregation (*Sum, Mean, Count, Median*).
- **Data Health & Cleaning Toolbar**: Live profile table, whitespace standardization, deduplication, and export to processed CSV.

---

## 🌾 The Agricultural Problem & Motivation

Indian agriculture supports over 140 million operational landholdings and accounts for significant national employment. However, production is subjected to high spatial-temporal volatility:
- **Monsoon Exposure**: South-West Monsoon variations introduce extreme yield swings in rainfed Kharif crops.
- **Regional Productivity Gaps**: Cereal yields in canal-irrigated zones (e.g., Punjab, Haryana at $>4.5$ t/ha) exceed rainfed regions (e.g., Bihar, Odisha at $1.5$–$2.2$ t/ha) by over 200%.
- **Supply Chain Volatility**: Inaccurate pre-harvest yield forecasts trigger post-harvest price crashes, storage gluts, and food procurement shortages.

AgriYield AI transforms historical administrative records into actionable pre-harvest intelligence to empower agricultural credit providers, warehouse operators, food processors, and policymakers.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Data Layer (data.gov.in / DES)                │
│   Official apy.csv: 246,091 records, 1997-2015, All India   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Data Ingestion & Quality Auditing Layer            │
│  src/data/load_data.py    -> Schema validation & retrieval  │
│  src/data/preprocess.py   -> Casing, units, nulls, outliers │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Subsystem                  │
│  src/features/build_features.py                             │
│  - Agronomic taxonomy (7 crop families)                     │
│  - Operational farm scale tiers (Marginal to Large)         │
│  - Variance-stabilizing Log-Area: ln(Area + 1)              │
│  - Leak-free expanding historical yield benchmarks          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         Chronological ML Pipeline & Model Training          │
│  src/models/train.py & src/models/evaluate.py               │
│  - Train partition: 1997–2011 (202,487 samples)             │
│  - Test partition:  2012–2015 (37,726 samples)              │
│  - ColumnTransformer: StandardScaler + OrdinalEncoder       │
│  - Benchmarks: Dummy Mean, Ridge, Random Forest, HistGB     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│       Interactive Streamlit Cloud Production Dashboard      │
│  app.py: Modern High-Contrast Agricultural Intelligence     │
│  - 🌾 Home: Overview & KPI Summary                          │
│  - 🏛️ Government Data & Yield AI (Trends + ML Predictor)    │
│  - 📤 Upload & Analyse Your Data (Instant Visual Graphs)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Official Dataset & Provenance

This application uses the authentic, primary candidate dataset published by the Government of India:

- **Dataset Title**: *District-wise, Season-wise Crop Production Statistics from 1997*
- **Publisher**: Ministry of Agriculture and Farmers Welfare, Department of Agriculture and Farmers Welfare, Directorate of Economics and Statistics (DES).
- **Official Open Data Portal**: [data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)
- **Catalogue URL**: [delhi.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0](https://delhi.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0)
- **Coverage**: 1997 through 2015 (19 consecutive crop years) across 33 States/UTs, 646 administrative districts, and 124 distinct agricultural crops.
- **Original Dimensions**: 246,091 rows $\times$ 7 columns.
- **File Size**: 15,316,741 bytes (~15.3 MB).
- **Licensing**: Government Open Data License - India (GODL).

---

## 🛡️ Data Pipeline & Quality Engineering

The preprocessing pipeline (`src/data/preprocess.py`) audits the raw records and resolves four critical real-world data issues:

1. **Documented Handling of Missing Production**:
   - Exactly **3,730 records (1.52%)** have null values in `Production`.
   - *Engineering Decision*: Because `Production` is the ground-truth target determinant for `Yield`, imputing production would create synthetic ground truth and distort evaluation. These rows are excluded from supervised model training but retained in data quality audit reports.
2. **The Coconut Anomaly (Unit Heterogeneity)**:
   - In official Indian agricultural reporting (DES), Coconut is documented in **thousand nuts** or **nuts**, while field crops (Rice, Wheat, Pulses) and commercial crops (Sugarcane) are reported in **Metric Tonnes**.
   - *Engineering Decision*: Records are tagged with a `Unit_Type` flag (`"Metric Tonnes"` vs `"Thousand Nuts"`). Non-tonne crops are partitioned to prevent mixing incompatible dimensions.
3. **Division-by-Zero Prevention**:
   - Sown `Area` has 0 missing values and minimum recorded area is $0.04$ hectares ($> 0$), ensuring mathematical stability for $\text{Yield} = \frac{\text{Production}}{\text{Area}}$.
4. **Biological Bounds & Outlier Screening**:
   - Flagged recording errors (e.g. Sugarcane yield $>80,000$ t/ha or Wheat $>100$ t/ha) using Indian Council of Agricultural Research (ICAR) biological ceilings (Wheat $\le 12$ t/ha, Rice $\le 15$ t/ha, Sugarcane $\le 250$ t/ha). A total of 190 records ($0.08\%$) were flagged, preserving genuine high yields while eliminating data entry errors.

| Quality Dimension | Raw Input | Cleaned / Retained | Status |
|:---|:---:|:---:|:---:|
| Total Records | 246,091 | 242,171 | 98.41% Pipeline Retention |
| Exact Duplicate Rows | 0 | 0 | Clean |
| Key Duplicates `[State, Dist, Year, Season, Crop]` | 0 | 0 | Unique Granularity |
| Missing Production Rows | 3,730 | Excluded from supervised sets | Explicitly Documented |
| Outlier Yields Flagged | 190 | Filtered via ICAR Bounds | Quality Assured |

---

## ⚙️ Feature Engineering & Anti-Leakage Protocol

All features in `src/features/build_features.py` are strictly derived from parameters available at sowing or early vegetative stages:

1. **Agronomic Crop Taxonomy (`Crop_Category`)**:
   - Groups 124 commodities into 7 biological families: Cereals, Pulses, Oilseeds, Commercial, Horticulture, Plantation, Other.
2. **Log-Transformed Area (`Log_Area`)**:
   - Formula: $\text{Log\_Area} = \ln(\text{Area} + 1)$
   - Normalizes exponential landholding distributions spanning from 0.04 ha to 8,000,000 ha.
3. **Operational Landholding Scale (`Farm_Scale`)**:
   - Stratified based on Ministry of Agriculture landholding tiers: Marginal ($<1$ ha), Small ($1$–$2$ ha), Semi-Medium ($2$–$4$ ha), Medium ($4$–$10$ ha), Large ($\ge 10$ ha).
4. **Temporal Counter (`Year_Index`)**:
   - Formula: $\text{Crop\_Year} - 1997$ (Captures multi-decade technological advancements, mechanization, and varietal adoption).
5. **Monsoon Dependency Flag (`Is_Kharif_Monsoon`)**:
   - Binary indicator for rainfed Kharif season exposure.
6. **Leak-Free Historical Crop-State Yield (`Hist_Crop_State_Yield`)**:
   - Expanding historical benchmark calculated strictly from training records ($\text{Year} \le 2011$). Prevents future state performance from leaking into past evaluations.

---

## 📊 Machine Learning Benchmarks & Evaluation

### Out-of-Time Model Leaderboard
Models were trained on historical records (**1997–2011, 202,487 samples**) and evaluated on unseen future crop seasons (**2012–2015, 37,726 samples**):

| Model Architecture | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | Fit Time (s) | Evaluation Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Baseline (Sample Mean)** | 5.658 | 12.596 | -0.003 | 0.00s | Zero-intelligence reference |
| **Ridge Regression** | 1.377 | 4.526 | 0.871 | 0.35s | Linear regularized baseline |
| **Random Forest Regressor** | 1.243 | 4.554 | 0.869 | 5.88s | 60 trees, max depth 14 |
| **HistGradientBoosting** | **1.256** | **4.518** | **0.871** | **5.96s** | 🏆 **Production Champion** |

### Major Crops Performance Breakdown (Out-of-Time Test Set)
| Crop | Test Samples | Actual Mean (t/ha) | Pred Mean (t/ha) | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Wheat** | 1,139 | 2.872 | 2.887 | 0.448 | 0.646 | **0.645** |
| **Rice** | 2,490 | 2.186 | 2.195 | 0.518 | 0.731 | **0.381** |
| **Maize** | 2,429 | 2.502 | 2.476 | 0.771 | 1.086 | **0.469** |
| **Onion** | 1,132 | 12.569 | 12.215 | 2.812 | 4.161 | **0.682** |
| **Groundnut**| 1,411 | 1.259 | 1.272 | 0.379 | 0.529 | **0.457** |
| **Sugarcane**| 1,229 | 67.240 | 66.812 | 14.120 | 20.639 | **0.516** |

---

## 💡 Agribusiness & Policy Implications

1. **Credit Ceiling Indexing**: Cereal yields vary by over 200% between irrigated and rainfed districts. Commercial banks should index Kisan Credit Card limits to localized yield potential rather than flat state-wide land norms to reduce non-performing assets.
2. **Logistics & Cold-Chain Pre-Booking**: High-perishable horticulture crops (Potato, Onion) experience district production swings $>200\%$. Processors can utilize 60-day pre-harvest forecasts to contract refrigerated transport and warehouse leases.
3. **Insurance Underwriting**: Kharif monsoon crops exhibit substantially fatter tail risk than Rabi winter crops. Pradhan Mantri Fasal Bima Yojana (PMFBY) insurers should incorporate weather-index triggers for Kharif while maintaining standard indemnity levels for Rabi.
4. **Ecological Fallacy Warning**: District-level statistics average out farm-level heterogeneity. District forecasts must not be used as direct substitutes for plot-level soil tests or localized agronomic advisories.

---

## 📁 Repository Structure

```
agriyield-ai/
├── README.md                      # Comprehensive project documentation
├── app.py                         # Production Streamlit decision support web application
├── Dockerfile                     # Production containerization configuration
├── .dockerignore                  # Docker build exclusions
├── requirements.txt               # Pinned production dependencies (with openpyxl)
├── .gitignore                     # Deployment and cache exclusions
├── .streamlit/
│   └── config.toml                # Light theme & high contrast UI overrides
├── data/
│   ├── README.md                  # Comprehensive dataset provenance & schema guide
│   ├── raw/                       # Raw downloads directory
│   └── processed/
│       └── crop_production_cleaned.csv # Audited, unit-standardized dataset (242k rows)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Rigorous exploratory data analysis & quality audit
│   └── 02_model_experiments.ipynb # Chronological validation & model experiments
├── src/
│   ├── data/
│   │   ├── load_data.py           # Ingestion, schema validation, resilient download
│   │   └── preprocess.py          # Data cleaning, unit tagging, outlier auditing
│   ├── features/
│   │   └── build_features.py      # Agronomic taxonomy, log-transforms, benchmarks
│   ├── models/
│   │   ├── train.py               # Model training, comparisons, and serialization
│   │   ├── evaluate.py            # Residual diagnostics, breakdown reports
│   │   └── predict.py             # Inference engine with confidence intervals
│   └── visualization/
│       └── charts.py              # Themed Plotly interactive visualizations
├── tests/
│   ├── test_preprocessing.py      # Schema, null, and unit validation tests
│   ├── test_features.py           # Feature engineering, bounds, and inference tests
│   └── test_upload_logic.py       # Custom CSV/Excel parsing and yield tests
├── reports/
│   ├── data_quality_summary.md    # Comprehensive data audit report
│   └── model_evaluation.md       # Model performance and breakdown analysis
├── docs/
│   ├── architecture.md            # System architecture and diagrams
│   ├── methodology.md             # Prediction formulation and validation strategy
│   ├── data_dictionary.md         # Full schema and feature descriptions
│   └── limitations.md             # Assumptions, data limitations, responsible use
└── models/
    ├── best_yield_model.joblib    # Serialized champion pipeline (620 KB)
    └── model_metadata.json        # Training metadata and benchmark metrics
```

---

## 🚀 Installation & Local Quick Start

### 1. Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/sangeetha302005/agriyield-ai.git
cd agriyield-ai

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local Streamlit Application
```bash
streamlit run app.py
```
*The Streamlit application will launch automatically at `http://localhost:8501`.*

---

## 🧪 Reproducibility & Testing

The test suite validates data ingestion, schema enforcement, category mapping, operational scale tiers, anti-leakage benchmarks, custom CSV/Excel uploads, and model inference constraints:

```bash
python -m pytest -v
```

**Test Results (100% Pass):**
```
tests/test_features.py::test_crop_category_mapping PASSED                [  9%]
tests/test_features.py::test_farm_scale_classification PASSED            [ 18%]
tests/test_features.py::test_feature_engineering_pipeline PASSED         [ 27%]
tests/test_features.py::test_input_validation PASSED                     [ 36%]
tests/test_features.py::test_prediction_output_structure PASSED          [ 45%]
tests/test_preprocessing.py::test_validate_raw_schema_valid PASSED       [ 54%]
tests/test_preprocessing.py::test_validate_raw_schema_missing PASSED     [ 63%]
tests/test_preprocessing.py::test_clean_text_columns PASSED              [ 72%]
tests/test_preprocessing.py::test_audit_and_clean_data_pipeline PASSED   [ 81%]
tests/test_upload_logic.py::test_custom_csv_upload_and_yield_calculation PASSED [ 90%]
tests/test_upload_logic.py::test_custom_upload_missing_and_duplicates PASSED [100%]

======================= 11 passed in 2.53s =======================
```

---

## 🐳 Docker Deployment

To build and run the application via Docker:

```bash
# Build the container
docker build -t agriyield-ai .

# Run the container
docker run -p 8501:8501 agriyield-ai
```
Access the containerized app at `http://localhost:8501`.

---

## ⚠️ Assumptions, Limitations & Responsible Use

1. **District-Level Aggregations**: Predictions describe macro district-level agricultural conditions. They must not be applied directly to individual farm plots without local soil testing (Soil Health Card) and plot-level moisture data.
2. **Extreme Climate Shocks**: The model reflects historical relationships (1997–2015). Black-swan weather disasters (severe cyclones, flash droughts during grain filling) may produce outcomes outside statistical confidence intervals.
3. **Responsible Lending**: Predictive scores should be utilized as risk-screening and advisory benchmarks, never as sole automated grounds for denying credit to smallholder farmers.

See [docs/limitations.md](docs/limitations.md) for full governance details.

---

## 📜 Citation & Acknowledgments

- **Dataset**: Directorate of Economics and Statistics (DES), Department of Agriculture and Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India.
- **Agricultural Benchmarks**: Indian Council of Agricultural Research (ICAR) & Food and Agriculture Organization (FAO).
- **Open Data Portal**: [data.gov.in](https://data.gov.in)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
