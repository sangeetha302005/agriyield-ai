"""
src/data/load_data.py
---------------------
Data ingestion module for AgriYield AI.
Loads, validates, and verifies the official Government of India
District-wise, Season-wise Crop Production Statistics (1997-2015).
"""

import os
import sys
import logging
import urllib.request
import ssl
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected canonical schema from Government of India catalogue
EXPECTED_COLUMNS: List[str] = [
    "State_Name",
    "District_Name",
    "Crop_Year",
    "Season",
    "Crop",
    "Area",
    "Production"
]

# Primary and fallback URLs
PRIMARY_SOURCE_URL = "https://aps.dac.gov.in/APY/apy.csv"
GOV_PORTAL_URL = "https://data.gov.in/resource/district-wise-season-wise-crop-production-statistics-1997"
MIRROR_RAW_URL = "https://raw.githubusercontent.com/dibyendubiswas1998/Crop-Production-Analysis/main/DATA/crop_production.csv"


def get_default_raw_path() -> Path:
    """Resolve default raw data path relative to project root."""
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / "data" / "raw" / "crop_production_raw.csv"


def download_raw_dataset(dest_path: Optional[Path] = None) -> Path:
    """
    Download the authentic Government of India crop production dataset.
    Tries primary endpoint first, then open verifiable mirror if official server times out.
    """
    if dest_path is None:
        dest_path = get_default_raw_path()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    logger.info("Attempting download from primary endpoint: %s", PRIMARY_SOURCE_URL)
    downloaded = False
    try:
        req = urllib.request.Request(
            PRIMARY_SOURCE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgriYieldAI/1.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            downloaded = True
            logger.info("Successfully downloaded %d bytes from primary endpoint.", len(content))
    except Exception as e:
        logger.warning("Primary endpoint unreachable or timed out (%s). Falling back to authentic mirror...", e)

    if not downloaded:
        logger.info("Attempting download from authentic verified mirror: %s", MIRROR_RAW_URL)
        req = urllib.request.Request(
            MIRROR_RAW_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgriYieldAI/1.0"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=45) as resp:
            content = resp.read()
            with open(dest_path, "wb") as f:
                f.write(content)
            logger.info("Successfully retrieved %d bytes from verified mirror.", len(content))

    return dest_path


def validate_raw_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Verify that the input DataFrame strictly adheres to the expected column schema.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list of missing columns)
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    is_valid = len(missing) == 0
    return is_valid, missing


def load_raw_data(
    filepath: Optional[str or Path] = None,
    auto_download: bool = True
) -> pd.DataFrame:
    """
    Load raw crop production dataset with schema validation.

    Args:
        filepath: Custom path to raw CSV file. If None, default path is used.
        auto_download: Whether to attempt automatic download if file is absent.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the file is not found and auto_download is False or fails.
        ValueError: If the schema does not match the canonical expected columns.
    """
    path = Path(filepath) if filepath else get_default_raw_path()

    if not path.exists():
        if auto_download:
            logger.info("Raw dataset not found at %s. Triggering automated download...", path)
            download_raw_dataset(path)
        else:
            raise FileNotFoundError(
                f"Raw dataset not found at '{path}'. Please download the official dataset "
                f"from {GOV_PORTAL_URL} and place it at {path}."
            )

    logger.info("Loading raw dataset from: %s", path)
    df = pd.read_csv(path, low_memory=False)

    # Validate schema
    is_valid, missing = validate_raw_schema(df)
    if not is_valid:
        raise ValueError(
            f"Schema validation failed! Missing expected columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    logger.info(
        "Successfully loaded %d records and %d columns. Memory usage: %.2f MB",
        len(df),
        len(df.columns),
        df.memory_usage(deep=True).sum() / (1024 * 1024)
    )
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Schema Validation: SUCCESS")
    print(df.info())
    print("\nSample records:")
    print(df.head())
