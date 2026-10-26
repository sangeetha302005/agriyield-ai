"""
src/features/build_features.py
------------------------------
Feature engineering module for AgriYield AI.
Constructs agronomic categories, operational scale indicators, log-transforms,
and historical benchmark features while strictly preventing target leakage.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Canonical Agronomic Taxonomy Mapping for Indian Crops
CROP_CATEGORY_MAP: Dict[str, str] = {
    # Cereals & Millets
    "Rice": "Cereals", "Wheat": "Cereals", "Maize": "Cereals", "Bajra": "Cereals",
    "Jowar": "Cereals", "Ragi": "Cereals", "Barley": "Cereals", "Small millets": "Cereals",
    "Other Cereals & Millets": "Cereals", "Paddy": "Cereals", "Total foodgrain": "Cereals",
    "Other Kharif pulses": "Pulses", "Other Rabi pulses": "Pulses",
    # Pulses
    "Gram": "Pulses", "Arhar/Tur": "Pulses", "Moong(Green Gram)": "Pulses", "Urad": "Pulses",
    "Masoor": "Pulses", "Peas & beans (Pulses)": "Pulses", "Horse-gram": "Pulses",
    "Khesari": "Pulses", "Moth": "Pulses", "Lentil": "Pulses", "Other  Rabi pulses": "Pulses",
    "Other Pulses": "Pulses",
    # Oilseeds
    "Groundnut": "Oilseeds", "Rapeseed &Mustard": "Oilseeds", "Soyabean": "Oilseeds",
    "Sunflower": "Oilseeds", "Sesamum": "Oilseeds", "Castor seed": "Oilseeds",
    "Linseed": "Oilseeds", "Safflower": "Oilseeds", "Niger seed": "Oilseeds",
    "Oilseeds total": "Oilseeds", "Other Oilseeds": "Oilseeds",
    # Commercial & Cash Crops
    "Sugarcane": "Commercial", "Cotton(lint)": "Commercial", "Jute": "Commercial",
    "Mesta": "Commercial", "Tobacco": "Commercial", "Jute & mesta": "Commercial",
    "Sanhemp": "Commercial",
    # Horticulture, Vegetables & Fruits
    "Potato": "Horticulture", "Onion": "Horticulture", "Banana": "Horticulture",
    "Tapioca": "Horticulture", "Sweet potato": "Horticulture", "Garlic": "Horticulture",
    "Ginger": "Horticulture", "Turmeric": "Horticulture", "Dry chillies": "Horticulture",
    "Coriander": "Horticulture", "Black pepper": "Horticulture", "Cardamom": "Horticulture",
    "Other Fresh Fruits": "Horticulture", "Other Vegetables": "Horticulture",
    "Papaya": "Horticulture", "Mango": "Horticulture", "Orange": "Horticulture",
    "Tomato": "Horticulture", "Brinjal": "Horticulture", "Cabbage": "Horticulture",
    "Cauliflower": "Horticulture", "Grapes": "Horticulture", "Pineapple": "Horticulture",
    "Apple": "Horticulture", "Cashewnut": "Horticulture", "Arecanut": "Plantation",
    "Coconut": "Plantation", "Tea": "Plantation", "Coffee": "Plantation", "Rubber": "Plantation"
}


def get_crop_category(crop_name: str) -> str:
    """Classify crop into high-level agronomic category."""
    crop_clean = str(crop_name).strip()
    return CROP_CATEGORY_MAP.get(crop_clean, "Other / Mixed")


def assign_farm_scale(area_hectares: float) -> str:
    """
    Classify cultivated area into operational holding scale based on
    standard Ministry of Agriculture & Farmers Welfare brackets:
    - Marginal: < 1.0 ha
    - Small: 1.0 to < 2.0 ha
    - Semi-Medium: 2.0 to < 4.0 ha
    - Medium: 4.0 to < 10.0 ha
    - Large: >= 10.0 ha
    """
    if area_hectares < 1.0:
        return "Marginal (<1ha)"
    elif area_hectares < 2.0:
        return "Small (1-2ha)"
    elif area_hectares < 4.0:
        return "Semi-Medium (2-4ha)"
    elif area_hectares < 10.0:
        return "Medium (4-10ha)"
    else:
        return "Large (>=10ha)"


def compute_historical_yield_benchmarks(
    df: pd.DataFrame,
    split_year: int = 2011
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], float], float]:
    """
    Compute historical crop-state average yields strictly up to split_year to avoid
    leakage into the test set.

    Returns:
        Tuple: (df with engineered feature, crop_state_benchmark_dict, global_mean_yield)
    """
    df = df.copy()
    train_mask = df["Crop_Year"] <= split_year
    train_df = df[train_mask]

    # Global mean yield for fallback
    global_mean = float(train_df["Yield"].mean())

    # State-Crop benchmark
    state_crop_means = (
        train_df.groupby(["State_Name", "Crop"])["Yield"]
        .mean()
        .to_dict()
    )

    # Crop-only benchmark for fallback
    crop_means = train_df.groupby("Crop")["Yield"].mean().to_dict()

    # Map back to entire dataset
    def map_benchmark(row):
        key = (row["State_Name"], row["Crop"])
        if key in state_crop_means:
            return state_crop_means[key]
        elif row["Crop"] in crop_means:
            return crop_means[row["Crop"]]
        return global_mean

    df["Hist_Crop_State_Yield"] = df.apply(map_benchmark, axis=1)

    return df, state_crop_means, global_mean


def engineer_features(
    df: pd.DataFrame,
    split_year: int = 2011,
    filter_outliers: bool = True,
    exclude_non_tonne_crops: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Primary feature engineering pipeline.

    Args:
        df: Cleaned crop dataset.
        split_year: Cutoff year for chronological train/test split.
        filter_outliers: Whether to exclude records flagged as extreme outliers.
        exclude_non_tonne_crops: Whether to exclude crops with different units (Coconut)
                                 for metric tonnes/ha yield modeling.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Engineered DataFrame and metadata.
    """
    logger.info("Executing feature engineering pipeline...")
    data = df.copy()

    # Filter non-tonne crops if modeling tonnes/ha
    if exclude_non_tonne_crops and "Unit_Type" in data.columns:
        before_len = len(data)
        data = data[data["Unit_Type"] == "Metric Tonnes"].copy()
        logger.info("Filtered non-tonne units (e.g. Coconut nuts): %d -> %d rows", before_len, len(data))

    # Filter extreme outliers if specified
    if filter_outliers and "is_outlier_yield" in data.columns:
        before_len = len(data)
        data = data[~data["is_outlier_yield"]].copy()
        logger.info("Excluded flagged biological outlier yields: %d -> %d rows", before_len, len(data))

    # 1. Agronomic Crop Category
    data["Crop_Category"] = data["Crop"].apply(get_crop_category)

    # 2. Log-Transformed Cultivated Area
    # Stabilizes exponential variance across farm sizes
    data["Log_Area"] = np.log1p(data["Area"])

    # 3. Operational Farm Scale Indicator
    data["Farm_Scale"] = data["Area"].apply(assign_farm_scale)

    # 4. Temporal Trend Index (Base year 1997 = 0)
    data["Year_Index"] = data["Crop_Year"] - 1997

    # 5. Monsoon Season Indicator (Kharif is monsoon rainfed)
    data["Is_Kharif_Monsoon"] = (data["Season"] == "Kharif").astype(int)

    # 6. Historical Benchmark Feature (strictly leakage-free)
    data, state_crop_benchmarks, global_mean = compute_historical_yield_benchmarks(
        data, split_year=split_year
    )

    metadata = {
        "split_year": split_year,
        "train_years": [int(data[data["Crop_Year"] <= split_year]["Crop_Year"].min()), split_year],
        "test_years": [split_year + 1, int(data[data["Crop_Year"] > split_year]["Crop_Year"].max())],
        "total_records": len(data),
        "train_records": int((data["Crop_Year"] <= split_year).sum()),
        "test_records": int((data["Crop_Year"] > split_year).sum()),
        "global_mean_yield": round(global_mean, 4),
        "feature_columns": [
            "State_Name", "District_Name", "Season", "Crop", "Crop_Category",
            "Farm_Scale", "Log_Area", "Year_Index", "Is_Kharif_Monsoon",
            "Hist_Crop_State_Yield"
        ],
        "target_column": "Yield"
    }

    logger.info("Feature engineering complete: %d train, %d test records.",
                metadata["train_records"], metadata["test_records"])
    return data, metadata


def get_chronological_splits(
    df: pd.DataFrame,
    features: List[str],
    target: str = "Yield",
    split_year: int = 2011
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data chronologically into train and test sets to simulate real-world forecasting.
    """
    train_mask = df["Crop_Year"] <= split_year
    test_mask = df["Crop_Year"] > split_year

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data.load_data import get_default_raw_path
    from src.data.preprocess import preprocess_crop_data

    df_cleaned, _ = preprocess_crop_data(save=False)
    df_feat, meta = engineer_features(df_cleaned)

    print("\n=== FEATURE ENGINEERING METADATA ===")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\nEngineered Feature Sample:")
    print(df_feat[meta["feature_columns"] + ["Yield"]].head())
