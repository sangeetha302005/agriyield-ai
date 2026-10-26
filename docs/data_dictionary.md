# AgriYield AI — Data Dictionary

## 1. Raw Dataset Schema

| Field Name | Type | Unit | Null % | Domain / Range | Description | Example |
|:---|:---|:---:|:---:|:---|:---|:---|
| `State_Name` | String | Categorical | 0.00% | 33 States / UTs | Official name of the State or Union Territory | "Uttar Pradesh" |
| `District_Name` | String | Categorical | 0.00% | 646 Districts | Administrative revenue district | "LUDHIANA" |
| `Crop_Year` | Integer | YYYY | 0.00% | 1997 – 2015 | Agricultural calendar harvest year | 2014 |
| `Season` | String | Categorical | 0.00% | 6 Seasons | Sowing and harvesting season | "Kharif", "Rabi" |
| `Crop` | String | Categorical | 0.00% | 124 Crops | Name of cultivated commodity | "Rice", "Wheat" |
| `Area` | Float | Hectares (ha) | 0.00% | 0.04 to 8,580,100 | Cultivated land area | 1250.0 |
| `Production` | Float | Metric Tonnes* | 1.52% | 0 to 1,250,800,000 | Gross agricultural harvest output | 3750.0 |

`*` *Note on Production Units: Coconut is recorded in nuts / thousand nuts in several reporting states; all other crops are recorded in Metric Tonnes.*

---

## 2. Engineered & Derived Features

| Field Name | Type | Formula / Logic | Unit | Description |
|:---|:---|:---|:---:|:---|
| `Yield` | Float | $\text{Production} / \text{Area}$ | Tonnes / Hectare | Primary target variable representing agricultural land productivity. |
| `Unit_Type` | String | `"Thousand Nuts"` if Coconut else `"Metric Tonnes"` | Category | Distinguishes grain/field crop metric weights from nut counts. |
| `Crop_Category` | String | Taxonomy lookup table | Category | Groups 124 crops into Cereals, Pulses, Oilseeds, Commercial, Horticulture, Plantation. |
| `Log_Area` | Float | $\ln(\text{Area} + 1)$ | Log Hectares | Normalized representation of cultivated area dampening skewness. |
| `Farm_Scale` | String | Landholding brackets | Category | Operational holding scale: Marginal (<1ha), Small (1-2ha), Semi-Medium (2-4ha), Medium (4-10ha), Large (>=10ha). |
| `Year_Index` | Integer | $\text{Crop\_Year} - 1997$ | Count | Linear temporal counter tracking multi-decade technological advancements. |
| `Is_Kharif_Monsoon` | Integer (0/1) | $1$ if $\text{Season} = \text{"Kharif"}$ else $0$ | Binary | Flags reliance on the South-West summer monsoon rainfall. |
| `Hist_Crop_State_Yield` | Float | Historical crop-state mean yield up to 2011 | Tonnes / Hectare | Anti-leakage regional agronomic benchmark computed strictly from training years. |
| `is_outlier_yield` | Boolean | $\text{Yield} > \text{Max\_Plausible\_Yield}$ | Flag | Identifies impossible physiological yields based on ICAR guidelines (e.g. Wheat >12 t/ha). |

---

## 3. Official Agricultural Seasons Definition

- **Kharif**: Sown with the onset of the South-West Monsoon (June–July) and harvested in Autumn (September–October). Major crops: Rice, Maize, Groundnut, Cotton, Soyabean.
- **Rabi**: Sown after the monsoon (October–November) and harvested in Spring (March–April). Major crops: Wheat, Gram, Mustard, Barley.
- **Summer / Zaid**: Short cropping season between Rabi and Kharif (March–June) with irrigation. Major crops: Vegetables, Pulses, Fodder.
- **Autumn**: Early harvest cycle in eastern and northeastern states.
- **Winter**: Late winter harvest cycle for special localized varietals.
- **Whole Year**: Perennial or continuous harvest crops spanning multiple seasons (Sugarcane, Banana, Coconut).
