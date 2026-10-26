# 🌾 AgriYield AI — Smart Agricultural Analytics & Pre-Harvest Crop Yield Intelligence

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agriyield-ai-302005.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-2D5A3F?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/PyTest-11%20passed-brightgreen.svg)](tests/)
[![Data Source](https://img.shields.io/badge/Data%20Source-data.gov.in%20(DES)-1E3A2B)](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)

> ### 🌐 **[Launch Live Web Application](https://agriyield-ai-302005.streamlit.app/)**
> Explore 19 years of official crop production trends, run pre-harvest ML yield predictions, or upload your own farm data for instant interactive visual analytics.

---

## 🌟 Welcome to AgriYield AI

**AgriYield AI** is a production-grade agricultural analytics and machine learning platform engineered to help agribusinesses, financial institutions, and agricultural planners turn complex agricultural data into confident, pre-harvest decisions.

Trained on **246,091 authentic district-level observations** from the **Ministry of Agriculture & Farmers Welfare, Government of India (1997–2015)**, AgriYield AI pairs historical macro-analytics with a high-accuracy, zero-leakage crop yield forecasting engine (**$R^2 = 0.871$**, MAE = **1.256 Tonnes/Ha**).

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                         AgriYield AI Core Workflows                     │
  │                                                                        │
  │   [🌾 Home]              [🏛️ Government Data & Yield AI]  [📤 Upload & Analyse]│
  │   Overview & Live KPIs   Multi-Decade Trends & ML Engine  Custom CSV/Excel AI  │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 👀 What You Can Explore as a Viewer

### 1. 🏛️ Government Crop Intelligence & Pre-Harvest Yield AI
Explore 19 consecutive crop years spanning 33 Indian States/UTs, 646 administrative districts, and 124 agricultural commodities:
- **Interactive Time-Series**: Filter by crop, state, and season to visualize dual-axis yield (t/ha) and cultivated acreage trajectories from 1997 through 2015.
- **Regional & Seasonal Spreads**: Identify top-producing states and uncover harvest volatility across Kharif, Rabi, Summer, and Autumn cycles with seasonal box plots.
- **National Commodity Rankings**: Compare volume outputs across Cereals, Pulses, Commercial, and Oilseeds.
- **Pre-Harvest ML Yield Predictor**:
  - Enter realistic farm parameters: `State`, `District`, `Crop`, `Season`, `Sown Area (Ha)`, and `Target Year`.
  - Receive an instant **Yield Forecast (Tonnes/Ha)** and **Expected Gross Production (Tonnes)**.
  - View **80% Confidence Intervals** and automatic **Farm Scale Classification** (Marginal, Small, Medium, Large) based on national landholding tiers.

### 2. 📤 Custom Farm Data Ingestion & Instant Visual Analytics
Bring your own farm logs, agribusiness supplier manifests, or trial datasets:
- **Universal Ingestion**: Supports `.csv`, `.xlsx`, and `.xls` files (or click **"🌱 Load Sample Farm Dataset"** for an instant 1-click tour).
- **Automated Schema & Yield Discovery**: Automatically profiles columns, cleans string whitespace, and computes $\text{Yield} = \frac{\text{Production}}{\text{Area}}$ if area and harvest outputs are detected.
- **Instant Interactive Visual Graphs**:
  - 📊 **Category Performance Bar Chart**: Ranks top commodities or regions with value labels and green gradient styling.
  - 📦 **Statistical Distribution Box Plot**: Displays median yields, interquartile ranges, and harvest outliers.
  - 🔍 **Feature Correlation Scatter**: Plots relationships (e.g. Sown Area vs. Production) colored by category.
  - 📈 **Chronological Trend Line**: Maps historical progression over years or crop seasons.
- **Custom Interactive Chart Builder**: Build tailored charts on the fly by selecting Chart Type (*Bar, Line, Scatter, Box, Histogram, Pie/Donut*), X-Axis, Y-Axis, Color Grouping, and Aggregation (*Sum, Mean, Count, Median*).
- **Data Health & Cleaning Toolbar**: Real-time null %, duplicate detection, whitespace standardization, and 1-click **Export to Cleaned CSV**.

---

## 💼 For Technical Recruiters & Engineering Hiring Managers

If you are reviewing this repository for machine learning engineering, data science, or full-stack Python roles, here is what highlights the engineering rigor of this project:

| Engineering Dimension | Industry Challenge | AgriYield AI Implementation |
|:---|:---|:---|
| **Target Leakage** | Naive models use post-harvest production ($Y = \frac{P}{A}$) to predict yield, making them useless before cutting. | **Zero Target Leakage**: Predictors are restricted strictly to pre-harvest inputs available at planting. |
| **Temporal Leakage** | Standard random K-Fold splits use future data (e.g. 2014) to predict the past (e.g. 2008), inflating metrics artificially. | **Chronological Out-of-Time Split**: Trained on 1997–2011 (202,487 rows); strictly evaluated on unseen future years 2012–2015 (37,726 rows). |
| **Unit Heterogeneity** | Indian DES data reports Coconut in *nuts/thousand nuts* and grains in *metric tonnes*, causing regression loss failure. | **Taxonomic Unit Partitioning**: Tagged and isolated non-tonne crops to preserve loss function integrity. |
| **Outlier Discipline** | Data entry typos (e.g. Sugarcane $>80,000$ t/ha) skew regression lines. | **Domain Ceilings**: Filtered via Indian Council of Agricultural Research (ICAR) physiological limits (190 erroneous rows screened). |
| **High Cardinality** | 646 Indian districts cause cardinality explosion in gradient boosting trees. | **Expanding Benchmarks**: Engineered leak-free historical state-crop yield anchors up to year $t-1$. |
| **Code Reliability** | Unreliable notebooks that break in production. | **Production Pipeline**: Modular `src/` architecture, strict typing, 100% passing PyTest suite (11 tests), and Docker support. |

---

## 📊 Machine Learning Model Leaderboard

Models were trained on historical records (**1997–2011, 202,487 samples**) and evaluated on unseen future crop seasons (**2012–2015, 37,726 samples**):

| Model Candidate | MAE (t/ha) | RMSE (t/ha) | $R^2$ Score | Fit Time | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Zero-Intelligence Baseline (Mean)** | 5.658 | 12.596 | -0.003 | <0.01s | Naive Reference |
| **Ridge Regression** | 1.377 | 4.526 | 0.871 | 0.35s | Regularized Linear |
| **Random Forest Regressor** | 1.243 | 4.554 | 0.869 | 5.88s | 60 Trees (Depth 14) |
| **HistGradientBoosting** | **1.256** | **4.518** | **0.871** | **5.96s** | 🏆 **Champion Production Model** |

### Crop-Specific Accuracy on Unseen Future Harvests (2012–2015 Test Set)
- **Wheat**: MAE = **0.448 t/ha** | $R^2 = \mathbf{0.645}$
- **Rice**: MAE = **0.518 t/ha** | $R^2 = \mathbf{0.381}$
- **Maize**: MAE = **0.771 t/ha** | $R^2 = \mathbf{0.469}$
- **Onion**: MAE = **2.812 t/ha** | $R^2 = \mathbf{0.682}$
- **Sugarcane**: MAE = **14.120 t/ha** | $R^2 = \mathbf{0.516}$

---

## 🛠️ Technology Stack

- **Core Analytics & ML**: Python 3.11, Scikit-Learn, Pandas, NumPy, SciPy, Joblib
- **Interactive UI & Visualizations**: Streamlit Cloud, Plotly Express, Plotly Graph Objects, KaTeX
- **Data Engineering**: OpenPyXL (Excel engine), Custom Out-of-Time Preprocessing Engine
- **Quality Assurance**: PyTest (Automated unit test suite)
- **Deployment & CI/CD**: Streamlit Community Cloud, GitHub, Docker

---

## 📁 Modular Repository Layout

```
agriyield-ai/
├── app.py                         # Production Streamlit decision support web application
├── Dockerfile                     # Production container configuration
├── requirements.txt               # Pinned dependencies (with openpyxl & plotly)
├── .streamlit/
│   └── config.toml                # Light theme & high-contrast UI configuration
├── data/
│   ├── README.md                  # Comprehensive dataset provenance & schema guide
│   └── processed/
│       └── crop_production_cleaned.csv # Audited, unit-standardized dataset (242k rows)
├── models/
    ├── best_yield_model.joblib    # Serialized champion HistGradientBoosting model (620 KB)
    └── model_metadata.json        # Model parameters and validation benchmarks
├── src/
│   ├── data/
│   │   ├── load_data.py           # Ingestion, schema validation, resilient mirrors
│   │   └── preprocess.py          # Data cleaning, unit tagging, ICAR outlier auditing
│   ├── features/
│   │   └── build_features.py      # Taxonomic grouping, log transforms, benchmarks
│   ├── models/
│   │   ├── train.py               # Model training, comparisons, and serialization
│   │   ├── evaluate.py            # Residual diagnostics, breakdown reports
│   │   └── predict.py             # Pre-harvest inference engine with confidence intervals
│   └── visualization/
│       └── charts.py              # Themed Plotly interactive visualizations
├── tests/
│   ├── test_preprocessing.py      # Schema, null, and unit validation tests
│   ├── test_features.py           # Feature engineering, bounds, and inference tests
│   └── test_upload_logic.py       # Custom CSV/Excel parsing and yield tests
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Exploratory data analysis & quality audit
│   └── 02_model_experiments.ipynb # Chronological validation & model experiments
├── reports/
│   ├── data_quality_summary.md    # Comprehensive data audit report
│   └── model_evaluation.md       # Model performance and breakdown analysis
└── docs/
    ├── architecture.md            # System architecture and diagrams
    ├── methodology.md             # Prediction formulation and validation strategy
    ├── data_dictionary.md         # Full schema and feature descriptions
    └── limitations.md             # Assumptions, data limitations, responsible use
```

---

## ⚡ How to Run Locally in 60 Seconds

```bash
# 1. Clone the repository
git clone https://github.com/sangeetha302005/agriyield-ai.git
cd agriyield-ai

# 2. Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🧪 Automated Test Suite (100% Pass)

To verify the pipeline, feature transformations, custom CSV/Excel uploads, and inference constraints:

```bash
python -m pytest -v
```

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

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t agriyield-ai .

# Run container on port 8501
docker run -p 8501:8501 agriyield-ai
```

---

## 📜 Dataset Provenance & Attribution

- **Source**: Directorate of Economics and Statistics (DES), Ministry of Agriculture and Farmers Welfare, Government of India.
- **Resource**: [District-wise, Season-wise Crop Production Statistics from 1997](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)
- **Licensing**: Government Open Data License - India (GODL).

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
