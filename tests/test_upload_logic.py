"""
tests/test_upload_logic.py
--------------------------
Unit tests verifying custom CSV/Excel upload, profiling, and yield computation.
"""

import io
import pytest
import pandas as pd
import numpy as np


def test_custom_csv_upload_and_yield_calculation():
    """Verify upload data processing and yield computation."""
    csv_data = """State,District,Crop,Season,Area,Production
Punjab,LUDHIANA,Wheat,Rabi,100,450
Haryana,KARNAL,Rice,Kharif,200,600
"""
    df = pd.read_csv(io.StringIO(csv_data))
    assert len(df) == 2

    # Clean strings
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Compute Yield
    area_col = next((c for c in df.columns if c.strip().lower() == "area"), None)
    prod_col = next((c for c in df.columns if c.strip().lower() == "production"), None)

    assert area_col == "Area"
    assert prod_col == "Production"

    df["Yield"] = df[prod_col] / df[area_col]
    assert df["Yield"].tolist() == [4.5, 3.0]


def test_custom_upload_missing_and_duplicates():
    """Verify profiling handles missing values and duplicate detection."""
    data = {
        "Crop": ["Wheat", "Wheat", "Rice", np.nan],
        "Area": [10.0, 10.0, 20.0, 30.0],
        "Production": [40.0, 40.0, 60.0, 90.0]
    }
    df = pd.DataFrame(data)

    null_count = int(df.isnull().sum().sum())
    assert null_count == 1

    dup_count = int(df.duplicated().sum())
    assert dup_count == 1  # 2nd Wheat row is identical

    df_cleaned = df.drop_duplicates().dropna(subset=["Crop"])
    assert len(df_cleaned) == 2
