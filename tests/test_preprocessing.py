"""
tests/test_preprocessing.py
---------------------------
Unit tests for data validation, whitespace cleaning, and outlier auditing.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.load_data import validate_raw_schema, EXPECTED_COLUMNS
from src.data.preprocess import clean_text_columns, audit_and_clean_data


def test_validate_raw_schema_valid():
    """Verify schema validator accepts valid canonical dataframe."""
    sample_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    is_valid, missing = validate_raw_schema(sample_df)
    assert is_valid is True
    assert len(missing) == 0


def test_validate_raw_schema_missing():
    """Verify schema validator detects missing columns."""
    sample_df = pd.DataFrame(columns=["State_Name", "Crop_Year"])
    is_valid, missing = validate_raw_schema(sample_df)
    assert is_valid is False
    assert "Production" in missing
    assert "District_Name" in missing


def test_clean_text_columns():
    """Verify whitespace stripping and season standardization."""
    raw_df = pd.DataFrame({
        "State_Name": ["  kerala  ", "punjab "],
        "District_Name": ["kannur ", " amritsar"],
        "Season": ["Kharif ", "Rabi "],
        "Crop": [" Rice ", "Wheat"]
    })
    cleaned = clean_text_columns(raw_df)

    assert cleaned["State_Name"].tolist() == ["Kerala", "Punjab"]
    assert cleaned["District_Name"].tolist() == ["KANNUR", "AMRITSAR"]
    assert cleaned["Season"].tolist() == ["Kharif", "Rabi"]
    assert cleaned["Crop"].tolist() == ["Rice", "Wheat"]


def test_audit_and_clean_data_pipeline():
    """Verify end-to-end cleaning, yield formula, and outlier flags."""
    raw_df = pd.DataFrame({
        "State_Name": ["Punjab", "Haryana", "Bihar", "Kerala"],
        "District_Name": ["LUDHIANA", "KARNAL", "PATNA", "KOCHI"],
        "Crop_Year": [2010, 2011, 2012, 2013],
        "Season": ["Rabi", "Rabi", "Kharif", "Whole Year"],
        "Crop": ["Wheat", "Wheat", "Rice", "Coconut"],
        "Area": [100.0, 50.0, 200.0, 10.0],
        "Production": [450.0, np.nan, 500.0, 25000.0]  # One NaN, one Coconut
    })

    cleaned, summary = audit_and_clean_data(raw_df)

    # 1 row with NaN production should be excluded from cleaned supervised set
    assert len(cleaned) == 3
    assert summary["missing_production_count"] == 1

    # Check yield calculation: Yield = Production / Area
    wheat_row = cleaned[cleaned["Crop"] == "Wheat"].iloc[0]
    assert wheat_row["Yield"] == 4.5  # 450 / 100

    # Coconut unit check
    coconut_row = cleaned[cleaned["Crop"] == "Coconut"].iloc[0]
    assert coconut_row["Unit_Type"] == "Thousand Nuts"
