"""
tests/test_features.py
----------------------
Unit tests for feature engineering, scale assignment, and prediction validation.
"""

import pytest
import pandas as pd
import numpy as np

from src.features.build_features import (
    get_crop_category,
    assign_farm_scale,
    engineer_features,
    get_chronological_splits
)
from src.models.predict import validate_input_features, predict_crop_yield


def test_crop_category_mapping():
    """Verify botanical and agronomic classification."""
    assert get_crop_category("Rice") == "Cereals"
    assert get_crop_category("Wheat") == "Cereals"
    assert get_crop_category("Gram") == "Pulses"
    assert get_crop_category("Groundnut") == "Oilseeds"
    assert get_crop_category("Sugarcane") == "Commercial"
    assert get_crop_category("Potato") == "Horticulture"
    assert get_crop_category("UnknownCropXYZ") == "Other / Mixed"


def test_farm_scale_classification():
    """Verify operational landholding tiers."""
    assert assign_farm_scale(0.5) == "Marginal (<1ha)"
    assert assign_farm_scale(1.5) == "Small (1-2ha)"
    assert assign_farm_scale(3.0) == "Semi-Medium (2-4ha)"
    assert assign_farm_scale(7.5) == "Medium (4-10ha)"
    assert assign_farm_scale(25.0) == "Large (>=10ha)"


def test_feature_engineering_pipeline():
    """Verify engineered dataframe produces expected columns without NaN."""
    mock_clean = pd.DataFrame({
        "State_Name": ["Punjab", "Punjab", "Bihar", "Bihar"],
        "District_Name": ["LUDHIANA", "LUDHIANA", "PATNA", "PATNA"],
        "Crop_Year": [2008, 2010, 2012, 2014],
        "Season": ["Rabi", "Rabi", "Kharif", "Kharif"],
        "Crop": ["Wheat", "Wheat", "Rice", "Rice"],
        "Area": [100.0, 150.0, 50.0, 60.0],
        "Production": [400.0, 630.0, 150.0, 190.0],
        "Yield": [4.0, 4.2, 3.0, 3.167],
        "Unit_Type": ["Metric Tonnes", "Metric Tonnes", "Metric Tonnes", "Metric Tonnes"],
        "is_outlier_yield": [False, False, False, False]
    })

    feat_df, meta = engineer_features(mock_clean, split_year=2010)

    expected_cols = meta["feature_columns"]
    for col in expected_cols:
        assert col in feat_df.columns
        assert feat_df[col].isnull().sum() == 0

    assert "Log_Area" in feat_df.columns
    assert "Year_Index" in feat_df.columns


def test_input_validation():
    """Verify input validation handles invalid domains."""
    # Negative area
    valid, msg = validate_input_features("Punjab", "AMRITSAR", "Rabi", "Wheat", -5.0, 2015)
    assert not valid
    assert "strictly positive" in msg

    # Invalid year
    valid, msg = validate_input_features("Punjab", "AMRITSAR", "Rabi", "Wheat", 10.0, 1850)
    assert not valid
    assert "bounds" in msg

    # Valid input
    valid, msg = validate_input_features("Punjab", "AMRITSAR", "Rabi", "Wheat", 10.0, 2015)
    assert valid
    assert msg is None


def test_prediction_output_structure():
    """Verify prediction returns valid agricultural yield and production."""
    res = predict_crop_yield("Punjab", "LUDHIANA", "Rabi", "Wheat", area=100.0, year=2015)
    assert res["status"] == "success"
    assert res["predictions"]["predicted_yield_t_per_ha"] > 0.0
    assert res["predictions"]["estimated_production_tonnes"] > 0.0
    assert len(res["predictions"]["yield_confidence_interval_80pct"]) == 2
