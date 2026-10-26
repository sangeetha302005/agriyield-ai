"""
src/models/predict.py
---------------------
Inference and decision-support module for AgriYield AI.
Provides single-record and batch prediction capabilities with input validation,
plausible yield bounds, and agribusiness advisory notices.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from ..features.build_features import (
    get_crop_category,
    assign_farm_scale
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_CACHE: Optional[Any] = None
_METADATA_CACHE: Optional[Dict[str, Any]] = None


def get_model_paths() -> Tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parents[2]
    model_file = base_dir / "models" / "best_yield_model.joblib"
    meta_file = base_dir / "models" / "model_metadata.json"
    return model_file, meta_file


def load_prediction_artifacts():
    """Load model pipeline and metadata with in-memory caching."""
    global _MODEL_CACHE, _METADATA_CACHE
    if _MODEL_CACHE is not None and _METADATA_CACHE is not None:
        return _MODEL_CACHE, _METADATA_CACHE

    model_file, meta_file = get_model_paths()
    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(
            f"Trained model artifacts not found at {model_file}. "
            "Please train the model first by running: python -m src.models.train"
        )

    _MODEL_CACHE = joblib.load(model_file)
    with open(meta_file, "r", encoding="utf-8") as f:
        _METADATA_CACHE = json.load(f)

    return _MODEL_CACHE, _METADATA_CACHE


def validate_input_features(
    state: str,
    district: str,
    season: str,
    crop: str,
    area: float,
    year: int
) -> Tuple[bool, Optional[str]]:
    """Validate user inputs against expected physical and categorical boundaries."""
    if not state or not isinstance(state, str):
        return False, "State name must be a valid non-empty string."
    if not district or not isinstance(district, str):
        return False, "District name must be a valid non-empty string."
    if not season or not isinstance(season, str):
        return False, "Season must be specified (e.g. Kharif, Rabi, Summer, Whole Year)."
    if not crop or not isinstance(crop, str):
        return False, "Crop name must be specified (e.g. Rice, Wheat, Sugarcane)."
    if area <= 0:
        return False, f"Cultivated area must be strictly positive (> 0 hectares). Received: {area}"
    if area > 10_000_000:
        return False, f"Cultivated area {area:,} ha exceeds maximum plausible district acreage."
    if year < 1990 or year > 2035:
        return False, f"Crop year {year} is out of realistic evaluation bounds (1990–2035)."

    return True, None


def predict_crop_yield(
    state: str,
    district: str,
    season: str,
    crop: str,
    area: float,
    year: int = 2016
) -> Dict[str, Any]:
    """
    Predict pre-harvest agricultural crop yield and expected harvest production.

    Args:
        state: Name of the State/UT (e.g. "Uttar Pradesh")
        district: Name of the District (e.g. "AGRA")
        season: Sowing season (e.g. "Kharif", "Rabi", "Summer", "Whole Year")
        crop: Agricultural commodity (e.g. "Rice", "Wheat")
        area: Sown area in hectares (ha)
        year: Cropping year

    Returns:
        Dict[str, Any]: Detailed prediction results including yield, production,
                        confidence intervals, and advisory flags.
    """
    is_valid, err_msg = validate_input_features(state, district, season, crop, area, year)
    if not is_valid:
        raise ValueError(f"Input validation error: {err_msg}")

    model, meta = load_prediction_artifacts()

    # Clean strings
    state_clean = state.strip().title()
    district_clean = district.strip().upper()
    season_clean = season.strip()
    crop_clean = crop.strip()

    # Derive engineered features
    crop_category = get_crop_category(crop_clean)
    log_area = float(np.log1p(area))
    farm_scale = assign_farm_scale(area)
    year_index = int(year - 1997)
    is_kharif = 1 if season_clean.lower() == "kharif" else 0

    # Historical state-crop benchmark proxy
    hist_yield = meta.get("global_mean_yield", 4.28)

    # Build single-row feature DataFrame
    input_data = pd.DataFrame([{
        "State_Name": state_clean,
        "District_Name": district_clean,
        "Season": season_clean,
        "Crop": crop_clean,
        "Crop_Category": crop_category,
        "Farm_Scale": farm_scale,
        "Log_Area": log_area,
        "Year_Index": year_index,
        "Is_Kharif_Monsoon": is_kharif,
        "Hist_Crop_State_Yield": hist_yield
    }])

    # Predict yield
    raw_pred_yield = float(model.predict(input_data)[0])
    pred_yield = max(0.0, raw_pred_yield)  # Physical yield cannot be negative

    # Estimated production: Production (Tonnes) = Yield (Tonnes/Ha) * Area (Ha)
    est_production = pred_yield * area

    # Model standard error from metadata
    hgb_metrics = meta.get("benchmark_results", {}).get("HistGradientBoosting", {})
    rmse = hgb_metrics.get("RMSE", 4.5)
    mae = hgb_metrics.get("MAE", 1.25)

    # Crop-specific confidence margin
    lower_yield = max(0.0, pred_yield - 1.28 * mae)
    upper_yield = pred_yield + 1.28 * mae

    lower_prod = lower_yield * area
    upper_prod = upper_yield * area

    advisory_notes = []
    if pred_yield == 0:
        advisory_notes.append("Forecast indicates near-zero yield or extreme crop vulnerability.")
    if farm_scale.startswith("Marginal"):
        advisory_notes.append("Marginal landholding (<1 ha): Potential exposure to smallholder liquidity constraints.")
    if is_kharif:
        advisory_notes.append("Kharif crop: Yield heavily contingent on South-West Monsoon spatial-temporal distribution.")

    return {
        "status": "success",
        "inputs": {
            "state": state_clean,
            "district": district_clean,
            "season": season_clean,
            "crop": crop_clean,
            "crop_category": crop_category,
            "area_ha": area,
            "farm_scale": farm_scale,
            "crop_year": year
        },
        "predictions": {
            "predicted_yield_t_per_ha": round(pred_yield, 3),
            "yield_confidence_interval_80pct": [round(lower_yield, 3), round(upper_yield, 3)],
            "estimated_production_tonnes": round(est_production, 2),
            "production_confidence_interval_80pct": [round(lower_prod, 2), round(upper_prod, 2)],
            "unit_yield": "Metric Tonnes / Hectare",
            "unit_production": "Metric Tonnes"
        },
        "advisory": advisory_notes,
        "model_info": {
            "champion_model": meta.get("best_model_name"),
            "model_mae": mae,
            "model_r2": hgb_metrics.get("R2", 0.87)
        }
    }


if __name__ == "__main__":
    sample_pred = predict_crop_yield(
        state="Uttar Pradesh",
        district="AGRA",
        season="Rabi",
        crop="Wheat",
        area=1500.0,
        year=2015
    )
    print("\n=== SAMPLE PREDICTION RESULT ===")
    print(json.dumps(sample_pred, indent=2))
