# AgriYield AI — Data Documentation

## 1. Dataset Overview

This project utilizes the authentic, official Government of India agricultural statistics dataset:

- **Dataset Title**: District-wise, Season-wise Crop Production Statistics from 1997
- **Publisher**: Ministry of Agriculture and Farmers Welfare, Department of Agriculture and Farmers Welfare, Directorate of Economics and Statistics (DES), Government of India
- **Government Open Data Portal**: [data.gov.in](https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997)
- **Catalogue URL**: [delhi.data.gov.in Catalog Reference](https://delhi.data.gov.in/catalog/district-wise-season-wise-crop-production-statistics-0)
- **Direct Resource Endpoint**: `https://aps.dac.gov.in/APY/apy.csv`
- **File Format**: Comma-Separated Values (CSV, UTF-8)
- **Original File Size**: 15,316,741 bytes (~15.3 MB)
- **Original Row Count**: 246,091 records
- **Licensing**: Government Open Data License - India (GODL)

---

## 2. Temporal & Geographic Coverage

- **Time Horizon**: 1997 through 2015 (19 consecutive crop years).
- **Geographic Scope**: All-India coverage across 33 States and Union Territories, encompassing 646 administrative districts.
- **Seasons Covered**: Kharif, Rabi, Summer, Autumn, Winter, and Whole Year.
- **Crops Encompassed**: 124 distinct agricultural commodities spanning food grains, oilseeds, commercial cash crops, horticulture, fruits, and plantation crops.

---

## 3. Raw Schema & Column Definitions

| Column Name | Data Type | Units / Format | Description |
|:---|:---|:---|:---|
| `State_Name` | String | Categorical | Name of the Indian State or Union Territory |
| `District_Name` | String | Categorical | Name of the administrative District |
| `Crop_Year` | Integer | YYYY (1997–2015) | Agricultural calendar year of harvest |
| `Season` | String | Categorical (6 classes) | Agricultural season (Kharif, Rabi, Summer, Autumn, Winter, Whole Year) |
| `Crop` | String | Categorical (124 classes) | Botanical / common agricultural crop name |
| `Area` | Float | Hectares (ha) | Land area cultivated for the specific crop |
| `Production` | Float | Metric Tonnes (t) / Nuts* | Total gross harvest output reported |

> **Crucial Domain Note on Measurement Units (`*`)**:  
> In the official Government reporting convention, standard field crops (Cereals, Pulses, Oilseeds, Sugarcane) are reported in **Metric Tonnes**, while **Coconut** production in several states is reported in **nuts / thousand nuts**. The preprocessing pipeline addresses this unit heterogeneity explicitly to maintain strict mathematical consistency.

---

## 4. Directory Organization

```
data/
├── README.md                  <- This comprehensive documentation
├── raw/
│   └── crop_production_raw.csv <- Unmodified raw dataset downloaded from official source
└── processed/
    ├── crop_production_cleaned.csv  <- Cleaned, validated, and normalized dataset
    └── crop_production_engineered.csv <- Feature-engineered dataset ready for ML workflows
```

---

## 5. Instructions to Acquire / Verify Raw Dataset

If cloning the repository without the raw data file, download the dataset directly using the automated pipeline:

```bash
python -m src.data.load_data --download
```

Or manually place the official file `apy.csv` / `crop_production_raw.csv` into `data/raw/crop_production_raw.csv`.
