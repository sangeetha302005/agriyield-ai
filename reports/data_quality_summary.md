# AgriYield AI — Data Quality Audit & Transformation Report

## Executive Data Quality Profile

| Dimension | Raw Input Metric | Cleaned / Retained Metric | Quality Interpretation |
|:---|:---:|:---:|:---|
| **Total Records** | 246,091 | 242,171 (Yield-ready) | 98.41% usable data retention rate |
| **Total Features / Columns** | 7 | 11 (Engineered) | Schema expanded with agronomic features |
| **Exact Duplicate Rows** | 0 | 0 | Clean analytical uniqueness |
| **Key Duplicates (`[State, Dist, Year, Season, Crop]`)** | 0 | 0 | Strict administrative granularity |
| **Missing Production Records** | 3,730 (1.516%) | 0 (Excluded from supervised target) | Preserves authentic ground-truth labels |
| **Missing Area Records** | 0 (0.000%) | 0 | Area strictly recorded for all observations |
| **Zero Cultivated Area (`Area <= 0`)** | 0 | 0 | No zero-division errors in Yield computation |
| **Zero Production (`Production == 0`)** | 3,523 (1.432%) | 3,523 (Retained as 0 t/ha) | Valid agricultural crop failures |
| **Unit Anomalies (Coconut in Nuts)** | 1,958 rows | Partitioned with `Unit_Type` flag | Prevents metric tonnes / nuts contamination |
| **Biological Yield Outliers Flagged** | 190 (0.077%) | Excluded from training set | Based on ICAR physiological ceilings |

---

## 1. Primary Source Attribution & Provenance

- **Title**: District-wise, Season-wise Crop Production Statistics from 1997
- **Data Publisher**: Directorate of Economics and Statistics (DES), Department of Agriculture and Farmers Welfare, Ministry of Agriculture and Farmers Welfare, Government of India.
- **Official Open Data Portal**: [data.gov.in (Resource ID 87630)](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)
- **Catalogue URL**: [delhi.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0](https://delhi.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0)
- **Direct Official Endpoint**: `https://aps.dac.gov.in/APY/apy.csv`
- **Retrieval Date**: October 2026
- **License**: Government Open Data License - India (GODL)

---

## 2. Documented Transformation Decisions

### 2.1 Missing Value Treatment
- **Observation**: 3,730 rows out of 246,091 lack values in the `Production` column.
- **Domain Rationale**: Because `Yield` is calculated as $\frac{\text{Production}}{\text{Area}}$, imputing production via mean, median, or K-Nearest Neighbors would generate artificial ground-truth targets, compromising model integrity.
- **Action**: These 3,730 rows are flagged and excluded from supervised yield modeling, but tracked in quality reports.

### 2.2 Whitespace and String Standardization
- **Observation**: Raw category fields contained irregular trailing spaces (e.g. `'Kharif   '`, `'Whole Year '`).
- **Action**: Applied strip normalization across all text fields. State names standardized to Title Case, District names standardized to Uppercase, Season labels mapped to canonical 6 agricultural seasons.

### 2.3 Unit Heterogeneity (The Coconut Discrepancy)
- **Observation**: Coconut production yields maximum values exceeding $1,000,000,000$, with calculated yields $>38,000$ units/ha.
- **Agricultural Investigation**: In the official Government of India reporting system, Coconut is documented in **thousand nuts** or **nuts**, whereas grain crops (Rice, Wheat, Pulses, Oilseeds) and commercial crops (Sugarcane) are reported in **Metric Tonnes**.
- **Action**: Tagged each record with `Unit_Type = "Thousand Nuts"` or `"Metric Tonnes"`. For metric tonnes/hectare yield modeling, non-tonne crops are partitioned to prevent distortion of loss functions.

### 2.4 Biological Yield Bounds & Outlier Screening
- **Observation**: Data entry anomalies present in raw government reports (e.g. Sugarcane yield of 88,000 tonnes/ha, Wheat yield of 200 tonnes/ha).
- **Domain Rationale**: Agronomic research from the Indian Council of Agricultural Research (ICAR) and FAO establishes biological maximum crop yields under ideal irrigation and fertilization:
  - Rice: $\le 15.0$ t/ha
  - Wheat: $\le 12.0$ t/ha
  - Maize: $\le 15.0$ t/ha
  - Sugarcane: $\le 250.0$ t/ha
  - Pulses: $\le 6.0$ t/ha
- **Action**: 190 records (0.08% of dataset) exceeding these physiological bounds were flagged as measurement/recording errors and excluded from training folds while remaining transparently documented.

---

## 3. Data Integrity Sign-Off

The resulting dataset `data/processed/crop_production_cleaned.csv` is mathematically consistent, unit-audited, free of division-by-zero errors, and certified for leak-free machine learning training.
