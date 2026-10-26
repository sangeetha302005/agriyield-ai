# AgriYield AI — System Architecture Specification

## 1. Architectural Overview

AgriYield AI is engineered using a modular, decoupled data science and software architecture that separates data ingestion, schema validation, feature engineering, machine learning pipelines, and interactive visualization.

```mermaid
flowchart TD
    subgraph Data_Layer ["Data Layer"]
        A["Official GoI Raw CSV<br>(apy.csv: 246k records)"] --> B["src.data.load_data<br>Schema Validation"]
        B --> C["src.data.preprocess<br>Auditing, Units, Outliers"]
        C --> D["data/processed/<br>Cleaned CSV"]
    end

    subgraph Feature_Layer ["Feature Engineering Layer"]
        D --> E["src.features.build_features<br>Agronomic Taxonomy & Scale"]
        E --> F["Leak-Free Historical Benchmarks<br>(Train-only Expansion)"]
        F --> G["Chronological Split<br>(Train <=2011 | Test 2012-2015)"]
    end

    subgraph Model_Layer ["Machine Learning Layer"]
        G --> H["Model Benchmarking<br>(Dummy, Ridge, RF, HistGB)"]
        H --> I["Champion Selection<br>(HistGradientBoosting R2: 0.871)"]
        I --> J["Model Serialization<br>(joblib + metadata JSON)"]
    end

    subgraph Presentation_Layer ["Presentation & UI Layer"]
        J --> K["app.py<br>Streamlit Multipage Dashboard"]
        D --> K
        K --> L["Executive Overview"]
        K --> M["Crop Analytics Engine"]
        K --> N["Data Quality Audit"]
        K --> O["Pre-Harvest Yield Predictor"]
        K --> P["Model Evaluation & Diagnostics"]
        K --> Q["Agribusiness Insights"]
    end
```

---

## 2. Component Design & Modularity

### 2.1 Ingestion & Preprocessing Subsystem (`src/data/`)
- `load_data.py`:
  - Enforces canonical schema (`EXPECTED_COLUMNS`).
  - Implements resilient automated retrieval from primary Government endpoints with verifiable fallback mirror.
  - Generates actionable error messages on missing files or corrupt headers.
- `preprocess.py`:
  - Executes strip cleaning and casing normalization.
  - Implements documented missing value filtering for `Production`.
  - Flags non-tonne crops (`Unit_Type = "Thousand Nuts"` for Coconut).
  - Flags biological yield outliers based on Indian Council of Agricultural Research (ICAR) physiological thresholds.

### 2.2 Feature Engineering Subsystem (`src/features/`)
- `build_features.py`:
  - Implements taxonomic mapping from 124 agricultural commodities into 7 agronomic families (Cereals, Pulses, Oilseeds, Commercial, Horticulture, Plantation, Other).
  - Calculates operational farm landholding tiers (Marginal, Small, Semi-Medium, Medium, Large) adhering to Ministry of Agriculture brackets.
  - Computes $\text{Log\_Area} = \ln(\text{Area} + 1)$ to stabilize variance across farm scales.
  - Computes leak-free historical state-crop average yields strictly from training years $\le t-1$.

### 2.3 Model Training & Inference Subsystem (`src/models/`)
- `train.py`:
  - Manages the training lifecycle for candidate algorithms.
  - Utilizes `sklearn.compose.ColumnTransformer` for feature scaling and ordinal encoding.
  - Serializes champion pipeline (`models/best_yield_model.joblib`) and metadata (`models/model_metadata.json`).
- `evaluate.py`:
  - Computes comprehensive out-of-time evaluation metrics: MAE, RMSE, $R^2$, and MAPE.
  - Generates error breakdowns across major crops and agricultural seasons.
  - Computes residual diagnostics (mean, median, 10th/90th percentile errors).
- `predict.py`:
  - Validates user input parameters against physical and administrative boundaries.
  - Formats pre-harvest inputs into model feature matrix.
  - Generates yield forecast, expected gross harvest production, confidence intervals, and agribusiness advisory notices.

### 2.4 Presentation Subsystem (`app.py`, `src/visualization/`)
- `charts.py`:
  - Interactive Plotly chart builders utilizing a modern agricultural intelligence theme (`#1E3A2B`, `#2D5A3F`, `#D4A373`, `#8FBC8F`, `#F8FAF8`).
- `app.py`:
  - Multipage Streamlit application featuring persistent caching (`@st.cache_data`, `@st.cache_resource`), scoped custom CSS, and responsive layout cards.
