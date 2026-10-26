# AgriYield AI — Assumptions, Limitations & Responsible Use

## 1. Underlying Assumptions

1. **Reporting Integrity**: District-level statistics reported by state agriculture departments to the Directorate of Economics and Statistics (DES) are assumed to reflect standard Crop Cutting Experiments (CCE) and revenue authority estimates.
2. **Pre-Harvest Area Availability**: Cultivated acreage (`Area`) is assumed to be cataloged or estimated by local revenue officers or remote sensing at the conclusion of the sowing window.
3. **Stationarity of Administrative Boundaries**: While new districts were created in India between 1997 and 2015, the model maps district identities based on reporting historical nomenclatures.

---

## 2. Dataset Limitations

1. **Aggregation Bias (Ecological Fallacy)**:
   - The data is aggregated at the **district administrative level**.
   - An average district yield of 3.0 tonnes/hectare may conceal a bimodal distribution within the district (e.g., highly mechanized, tubewell-irrigated commercial farms yielding 5.5 t/ha, alongside marginal rainfed plots yielding 1.2 t/ha).
   - District forecasts cannot be directly transposed onto an individual farmer's plot without local adjustment.

2. **Absence of Micro-Climatic & Soil Attributes in Raw GoI Records**:
   - The primary official Government of India dataset (`apy.csv`) records *only* administrative units, crop, season, year, area, and production.
   - It does not embed daily precipitation, surface temperature, evapotranspiration, soil organic carbon (SOC), or nitrogen-phosphorus-potassium (NPK) ratios.
   - Consequently, the model relies on historical regional proxies rather than direct biophysical simulation models (like DSSAT or APSIM).

3. **Temporal Horizon (1997–2015)**:
   - The official open dataset spans up to 2015. While representative of long-term structural relationships, it does not capture post-2015 agronomic developments such as the nationwide rollout of the PMFBY satellite verification system or recent climate volatility.

---

## 3. Model Limitations

1. **Extreme Climate Shocks**:
   - HistGradientBoosting learns relationships from observed training distributions. Unprecedented localized disasters (e.g. 1-in-100-year cloudbursts, severe cyclones, or unprecedented heatwaves during grain filling) may produce real outcomes outside the model's confidence bounds.

2. **High-Biomass Crops vs Food Grains**:
   - High-biomass perennial crops like Sugarcane (which naturally yield 50 to 120 tonnes/ha) exhibit higher absolute prediction errors ($\text{RMSE} \approx 20\text{ t/ha}$) than cereal grains like Wheat ($\text{RMSE} \approx 0.65\text{ t/ha}$), despite comparable relative percentage accuracy.

3. **Minor Crops with Low Sample Sizes**:
   - Crops with fewer than 50 historical observations across India possess wider uncertainty bounds and should be interpreted cautiously.

---

## 4. Responsible Use & Governance Guidelines

- **For Agricultural Lenders & Insurers**: Use AgriYield AI as a regional risk screening and benchmarking tool. Do not use it as the sole automated factor for denying credit to individual smallholder farmers.
- **For Agribusiness Supply Chains**: Use forecasts to guide regional procurement targets, warehouse leases, and transport scheduling 30–60 days ahead of harvest.
- **For Agronomists & Extension Officers**: Always prioritize on-ground soil health card measurements and direct field inspections over district-level statistical projections.
