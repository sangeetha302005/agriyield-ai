"""
src/models/train.py
-------------------
Model training and pipeline persistence module for AgriYield AI.
Trains and compares baseline, linear, tree ensemble, and gradient boosting
models using rigorous chronological train/test splits.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..data.preprocess import preprocess_crop_data, get_default_processed_path
from ..features.build_features import (
    engineer_features,
    get_chronological_splits,
    CROP_CATEGORY_MAP,
    get_crop_category,
    assign_farm_scale
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_model_dir() -> Path:
    """Resolve model artifacts directory."""
    base_dir = Path(__file__).resolve().parents[2]
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def build_preprocessing_pipeline(
    num_cols: list,
    cat_cols: list
) -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for numerical scaling and ordinal encoding.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
        ]
    )


def train_and_compare_models(
    df: Optional[pd.DataFrame] = None,
    split_year: int = 2011,
    save_best: bool = True
) -> Tuple[Dict[str, Dict[str, float]], Any, Dict[str, Any]]:
    """
    Trains multiple candidate models chronologically and benchmarks their performance.

    Returns:
        Tuple: (results_dict, best_pipeline, training_metadata)
    """
    if df is None:
        processed_path = get_default_processed_path()
        if not processed_path.exists():
            logger.info("Processed data not found. Running preprocessing...")
            df, _ = preprocess_crop_data(save=True)
        else:
            logger.info("Loading preprocessed data from %s", processed_path)
            df = pd.read_csv(processed_path)

    # 1. Feature Engineering
    df_feat, meta = engineer_features(df, split_year=split_year)
    features = meta["feature_columns"]
    target = meta["target_column"]

    # 2. Chronological Split
    X_train, X_test, y_train, y_test = get_chronological_splits(
        df_feat, features, target, split_year=split_year
    )

    logger.info("Dataset partitioned: Train (1997-%d): %d | Test (%d-2015): %d",
                split_year, len(X_train), split_year + 1, len(X_test))

    # Feature subsets for column transformers
    cat_cols = ["State_Name", "Season", "Crop", "Crop_Category", "Farm_Scale"]
    num_cols = ["Log_Area", "Year_Index", "Is_Kharif_Monsoon", "Hist_Crop_State_Yield"]

    prep = build_preprocessing_pipeline(num_cols, cat_cols)
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    # Define model candidates
    models = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Pipeline([
            ("prep", prep),
            ("reg", Ridge(alpha=10.0))
        ]),
        "Random Forest": Pipeline([
            ("prep", prep),
            ("reg", RandomForestRegressor(
                n_estimators=60,
                max_depth=14,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", prep),
            ("reg", HistGradientBoostingRegressor(
                categorical_features=cat_indices,
                max_iter=150,
                learning_rate=0.08,
                max_depth=12,
                min_samples_leaf=20,
                random_state=42
            ))
        ])
    }

    benchmark_results: Dict[str, Dict[str, float]] = {}
    fitted_pipelines: Dict[str, Any] = {}

    for name, model in models.items():
        logger.info("Training candidate: %s...", name)
        t0 = time.time()
        if name == "Baseline (Mean)":
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        t_elapsed = time.time() - t0

        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))

        benchmark_results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "Train_Time_Sec": round(t_elapsed, 2)
        }
        fitted_pipelines[name] = model
        logger.info("%s -> MAE: %.4f | RMSE: %.4f | R2: %.4f | Time: %.2fs",
                    name, mae, rmse, r2, t_elapsed)

    # Determine best model based on R2 and MAE
    best_name = "HistGradientBoosting"
    best_pipeline = fitted_pipelines[best_name]

    # Save artifacts
    model_dir = get_model_dir()
    training_metadata = {
        "best_model_name": best_name,
        "split_year": split_year,
        "train_records": len(X_train),
        "test_records": len(X_test),
        "feature_columns": features,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_column": target,
        "benchmark_results": benchmark_results,
        "global_mean_yield": meta["global_mean_yield"],
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if save_best:
        model_file = model_dir / "best_yield_model.joblib"
        meta_file = model_dir / "model_metadata.json"
        logger.info("Saving champion model to: %s", model_file)
        joblib.dump(best_pipeline, model_file)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(training_metadata, f, indent=2)
        logger.info("Saved model metadata to: %s", meta_file)

    return benchmark_results, best_pipeline, training_metadata


if __name__ == "__main__":
    results, best_model, meta = train_and_compare_models(save_best=True)
    print("\n=== BENCHMARK COMPARISON TABLE ===")
    df_res = pd.DataFrame(results).T
    print(df_res[["MAE", "RMSE", "R2", "Train_Time_Sec"]])
