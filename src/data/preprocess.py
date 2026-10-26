"""
src/data/preprocess.py
----------------------
Data preprocessing, validation, unit standardization, and outlier auditing
for AgriYield AI. Transforms raw official Government of India agricultural
records into a clean, audited, and analysis-ready dataset.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

from .load_data import load_raw_data, get_default_raw_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Agricultural domain upper yield bounds (Tonnes / Hectare) based on ICAR / FAO agronomic benchmarks
MAX_PLAUSIBLE_YIELDS: Dict[str, float] = {
    "Rice": 15.0,
    "Wheat": 12.0,
    "Maize": 15.0,
    "Sugarcane": 250.0,
    "Potato": 60.0,
    "Onion": 50.0,
    "Banana": 120.0,
    "Gram": 6.0,
    "Arhar/Tur": 6.0,
    "Moong(Green Gram)": 5.0,
    "Urad": 5.0,
    "Groundnut": 10.0,
    "Cotton(lint)": 8.0,
    "Soyabean": 8.0,
    "Rapeseed &Mustard": 6.0,
    "Bajra": 8.0,
    "Jowar": 8.0,
    "Default": 80.0
}


def get_default_processed_path() -> Path:
    """Resolve default processed data path relative to project root."""
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "processed" / "crop_production_cleaned.csv"


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize whitespace, casing, and category labels.
    """
    df = df.copy()
    str_cols = ["State_Name", "District_Name", "Season", "Crop"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Standardize State Names (Title Case for readability)
    df["State_Name"] = df["State_Name"].str.title()
    # Districts are kept in uppercase standard as per census/ministry convention
    df["District_Name"] = df["District_Name"].str.upper()

    # Standardize Season labels
    season_mapping = {
        "Kharif": "Kharif",
        "Kharif ": "Kharif",
        "Rabi": "Rabi",
        "Rabi ": "Rabi",
        "Summer": "Summer",
        "Summer ": "Summer",
        "Whole Year": "Whole Year",
        "Whole Year ": "Whole Year",
        "Autumn": "Autumn",
        "Autumn ": "Autumn",
        "Winter": "Winter",
        "Winter ": "Winter",
    }
    df["Season"] = df["Season"].map(lambda x: season_mapping.get(x, x))

    return df


def audit_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs comprehensive data auditing, cleaning, and quality reporting.

    Steps:
    1. Schema and type validation.
    2. Missing value quantification and documented treatment.
    3. Duplicate record detection.
    4. Unit standardization (distinguishing Coconut nuts vs metric tonnes).
    5. Yield computation: Yield = Production / Area.
    6. Agronomic and statistical outlier flagging.
    """
    logger.info("Starting data auditing and preprocessing pipeline...")
    initial_rows = len(df)

    # 1. Clean string values
    df = clean_text_columns(df)

    # 2. Type conversions
    df["Crop_Year"] = pd.to_numeric(df["Crop_Year"], errors="coerce").fillna(0).astype(int)
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")

    # 3. Quality audit stats
    missing_production_count = int(df["Production"].isnull().sum())
    missing_production_pct = round(missing_production_count / initial_rows * 100, 3)
    missing_area_count = int(df["Area"].isnull().sum())
    zero_area_count = int((df["Area"] <= 0).sum())
    zero_production_count = int((df["Production"] == 0).sum())
    exact_duplicates = int(df.duplicated().sum())

    key_cols = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]
    key_duplicates = int(df.duplicated(subset=key_cols).sum())

    logger.info("Missing Production records: %d (%.2f%%)", missing_production_count, missing_production_pct)
    logger.info("Exact duplicate rows: %d | Key duplicates: %d", exact_duplicates, key_duplicates)

    # 4. Documented Handling of Missing Production:
    # Records missing Production cannot compute ground truth Yield. We filter them for yield analytics,
    # preserving only valid rows for yield modeling.
    df_valid = df.dropna(subset=["Production"]).copy()
    # Filter non-positive area if any
    df_valid = df_valid[df_valid["Area"] > 0].copy()

    # 5. Unit standardisation
    # Official DES note: Coconut is reported in nuts / thousand nuts in many states.
    df_valid["Unit_Type"] = np.where(
        df_valid["Crop"].str.contains("Coconut", case=False, na=False),
        "Thousand Nuts",
        "Metric Tonnes"
    )

    # 6. Compute Target: Yield
    # Formula: Yield = Production / Area
    # Standard field crops: Metric Tonnes per Hectare
    df_valid["Yield"] = df_valid["Production"] / df_valid["Area"]

    # 7. Agronomic and Statistical Outlier Flagging
    # We do NOT arbitrarily delete points; we flag them using domain benchmarks and IQR.
    def get_max_yield(crop_name: str) -> float:
        return MAX_PLAUSIBLE_YIELDS.get(crop_name, MAX_PLAUSIBLE_YIELDS["Default"])

    df_valid["Max_Plausible_Yield"] = df_valid["Crop"].map(get_max_yield)

    # Outlier criteria for tonne crops:
    # Yield > 0, Yield <= Max_Plausible_Yield
    is_coconut = df_valid["Unit_Type"] == "Thousand Nuts"
    is_tonne_crop = ~is_coconut

    # Plausibility flag
    df_valid["is_outlier_yield"] = False
    # For tonne crops, if yield exceeds agronomic threshold, flag as outlier
    tonne_outlier_mask = is_tonne_crop & (df_valid["Yield"] > df_valid["Max_Plausible_Yield"])
    df_valid.loc[tonne_outlier_mask, "is_outlier_yield"] = True

    # Also calculate IQR bounds for non-outliers
    crop_iqr_stats = {}
    for crop, group in df_valid[is_tonne_crop].groupby("Crop"):
        q1 = group["Yield"].quantile(0.25)
        q3 = group["Yield"].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = q3 + 3.0 * iqr
        crop_iqr_stats[crop] = {
            "Q1": round(float(q1), 4),
            "Q3": round(float(q3), 4),
            "IQR": round(float(iqr), 4),
            "Upper_Whisker": round(float(upper_whisker), 4)
        }

    outlier_count = int(df_valid["is_outlier_yield"].sum())
    valid_records_count = int((~df_valid["is_outlier_yield"]).sum())

    logger.info(
        "Outlier screening complete. Flagged %d records (%.2f%%) exceeding biological bounds.",
        outlier_count,
        outlier_count / len(df_valid) * 100
    )

    quality_summary = {
        "raw_record_count": initial_rows,
        "raw_column_count": len(df.columns),
        "exact_duplicates": exact_duplicates,
        "key_duplicates": key_duplicates,
        "missing_production_count": missing_production_count,
        "missing_production_pct": missing_production_pct,
        "missing_area_count": missing_area_count,
        "zero_area_count": zero_area_count,
        "zero_production_count": zero_production_count,
        "cleaned_record_count": len(df_valid),
        "outlier_yield_records_flagged": outlier_count,
        "valid_yield_records": valid_records_count,
        "states_covered": int(df_valid["State_Name"].nunique()),
        "districts_covered": int(df_valid["District_Name"].nunique()),
        "crops_covered": int(df_valid["Crop"].nunique()),
        "years_covered": [int(df_valid["Crop_Year"].min()), int(df_valid["Crop_Year"].max())],
        "seasons_covered": sorted(df_valid["Season"].unique().tolist())
    }

    return df_valid, quality_summary


def preprocess_crop_data(
    raw_path: Optional[str or Path] = None,
    output_path: Optional[str or Path] = None,
    save: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end preprocessing pipeline. Loads raw data, audits, cleans, and saves.
    """
    df_raw = load_raw_data(raw_path)
    df_cleaned, quality_summary = audit_and_clean_data(df_raw)

    if save:
        out_path = Path(output_path) if output_path else get_default_processed_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving cleaned dataset to: %s", out_path)
        df_cleaned.to_csv(out_path, index=False)
        logger.info("Saved %d records successfully.", len(df_cleaned))

    return df_cleaned, quality_summary


if __name__ == "__main__":
    df, summary = preprocess_crop_data()
    print("\n=== DATA QUALITY AUDIT REPORT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nFirst 5 cleaned records:")
    print(df[["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production", "Yield", "is_outlier_yield"]].head())
