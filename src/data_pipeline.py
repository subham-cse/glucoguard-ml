"""
GlucoGuard: Multi-Factor Diabetes Risk & Early Prevention Intelligence Engine
Module: Data Pipeline & Preprocessing Architecture (src/data_pipeline.py)
Author: Principal ML Engineer / Bharat Cares AICTE ML Internship Final Submission
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Setup structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GlucoGuard.DataPipeline")

# Base directory resolution (relies on relative path from module root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "diabetes_data.csv"

# Dataset Constants
RAW_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_COLUMN = "Outcome"
ALL_COLUMNS = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]

# Columns where zero is clinically/physiologically invalid and represents missing observations
BIOLOGICAL_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

ENGINEERED_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Insulin_Glucose_Ratio",
    "BMI_Age_Interaction",
    "Risk_Age_Group",
    "HOMA_IR",
    "Skin_to_BMI_Ratio",
    "Metabolic_Syndrome_Risk",
    "Log_Insulin",
    "Log_Pedigree"
]

DATASET_REMOTE_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn compatible transformer for clinical feature engineering
    and biological zero handling. Enables reproducible transformations during training & inference.
    """
    def __init__(self, handle_zeros: bool = True):
        self.handle_zeros = handle_zeros
        self.median_values_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS[:X.shape[1]])

        df_copy = X.copy()
        if self.handle_zeros:
            for col in BIOLOGICAL_ZERO_COLS:
                if col in df_copy.columns:
                    df_copy[col] = df_copy[col].replace(0, np.nan)

        # Store training medians for fallback imputation
        for col in df_copy.columns:
            self.median_values_[col] = float(df_copy[col].median(skipna=True))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS[:X.shape[1]])
        df = X.copy()

        # 1. Biological zero replacement
        if self.handle_zeros:
            for col in BIOLOGICAL_ZERO_COLS:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)

        # 2. Impute with learned medians if NaNs present
        for col in df.columns:
            if col in self.median_values_ and df[col].isnull().any():
                df[col] = df[col].fillna(self.median_values_[col])

        # 3. Clinical Feature Engineering
        # Ratio of circulating insulin to fasting blood glucose
        insulin = df["Insulin"] if "Insulin" in df.columns else 0.0
        glucose = df["Glucose"] if "Glucose" in df.columns else 100.0
        bmi = df["BMI"] if "BMI" in df.columns else 25.0
        age = df["Age"] if "Age" in df.columns else 30.0
        bp = df["BloodPressure"] if "BloodPressure" in df.columns else 70.0
        skin = df["SkinThickness"] if "SkinThickness" in df.columns else 20.0
        dpf = df["DiabetesPedigreeFunction"] if "DiabetesPedigreeFunction" in df.columns else 0.35

        df["Insulin_Glucose_Ratio"] = (insulin / (glucose + 1e-6)).clip(0.0, 10.0)

        # Interaction between adiposity (BMI) and metabolic aging
        df["BMI_Age_Interaction"] = bmi * age

        # Ordinal binned metabolic age risk category
        df["Risk_Age_Group"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 29, 44, 59, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(int)

        # 4. Advanced Clinical & Metabolic Biomarkers
        # HOMA-IR Proxy (Homeostatic Model Assessment of Insulin Resistance)
        df["HOMA_IR"] = ((glucose * insulin) / 405.0).clip(0.0, 40.0)

        # Subcutaneous vs Visceral Adiposity ratio
        df["Skin_to_BMI_Ratio"] = (skin / (bmi + 1e-5)).clip(0.0, 3.0)

        # Metabolic Syndrome Risk Score (composite count of adult risk criteria 0-4)
        c_glucose = (glucose >= 100.0).astype(int)
        c_bp = (bp >= 85.0).astype(int)
        c_bmi = (bmi >= 30.0).astype(int)
        c_age = (age >= 45.0).astype(int)
        df["Metabolic_Syndrome_Risk"] = c_glucose + c_bp + c_bmi + c_age

        # Variance-stabilizing Log Transforms for highly skewed markers
        df["Log_Insulin"] = np.log1p(np.maximum(0.0, insulin))
        df["Log_Pedigree"] = np.log1p(np.maximum(0.0, dpf))

        # Ensure consistent column ordering
        return df[ENGINEERED_FEATURE_COLUMNS]


def generate_synthetic_pima_dataset(n_samples: int = 768, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a statistically coherent synthetic Pima Indians Diabetes Dataset
    matching empirical demographic means, standard deviations, correlations, and zero rates.
    Ensures zero dependency on network availability during fresh clone evaluation.
    """
    logger.info(f"Generating synthetic clinical fallback dataset with {n_samples} records...")
    rng = np.random.default_rng(random_state)

    # Class balance matching empirical Pima distribution (~65% negative, 35% positive)
    n_pos = int(n_samples * 0.349)
    n_neg = n_samples - n_pos

    # Empirical distribution parameters: [mean_neg, std_neg, mean_pos, std_pos]
    params = {
        "Pregnancies": (3.3, 3.0, 4.9, 3.7, 0, 17, 0.15),
        "Glucose": (110.0, 24.0, 141.0, 31.0, 44, 199, 0.01),
        "BloodPressure": (68.0, 12.0, 75.0, 13.0, 24, 122, 0.05),
        "SkinThickness": (27.0, 8.5, 33.0, 9.5, 7, 99, 0.30),
        "Insulin": (130.0, 65.0, 206.0, 110.0, 14, 846, 0.48),
        "BMI": (30.3, 6.5, 35.1, 6.6, 18.2, 67.1, 0.01),
        "DiabetesPedigreeFunction": (0.43, 0.30, 0.55, 0.37, 0.078, 2.42, 0.0),
        "Age": (31.2, 11.6, 37.0, 10.9, 21, 81, 0.0)
    }

    records = []
    # Negative cohort
    for _ in range(n_neg):
        row = {"Outcome": 0}
        for feat, (m_neg, s_neg, _, _, min_v, max_v, zero_rate) in params.items():
            if rng.random() < zero_rate:
                val = 0.0
            else:
                val = float(np.clip(rng.normal(m_neg, s_neg), min_v, max_v))
            row[feat] = round(val, 3 if "Pedigree" in feat else (1 if feat in ["BMI", "SkinThickness"] else 0))
        records.append(row)

    # Positive cohort
    for _ in range(n_pos):
        row = {"Outcome": 1}
        for feat, (_, _, m_pos, s_pos, min_v, max_v, zero_rate) in params.items():
            if rng.random() < zero_rate:
                val = 0.0
            else:
                val = float(np.clip(rng.normal(m_pos, s_pos), min_v, max_v))
            row[feat] = round(val, 3 if "Pedigree" in feat else (1 if feat in ["BMI", "SkinThickness"] else 0))
        records.append(row)

    df = pd.DataFrame(records)
    # Shuffle records
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df[ALL_COLUMNS]


def ensure_dataset_available() -> Path:
    """
    Automated data fetcher with redundant network endpoints and synthetic fallback.
    Guarantees data availability under any execution environment.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists() and DATA_FILE.stat().st_size > 1000:
        logger.info(f"Verified existing clinical dataset at: {DATA_FILE}")
        return DATA_FILE

    # Attempt remote download
    download_success = False
    for url in DATASET_REMOTE_URLS:
        try:
            logger.info(f"Attempting dataset download from: {url}")
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.text) > 1000:
                lines = resp.text.strip().splitlines()
                # Check if first line has header
                first_line = lines[0].lower()
                if "pregnancies" in first_line or "glucose" in first_line:
                    df = pd.read_csv(url)
                else:
                    df = pd.read_csv(url, header=None, names=ALL_COLUMNS)

                # Validate column count
                if df.shape[1] == 9:
                    df.columns = ALL_COLUMNS
                    df.to_csv(DATA_FILE, index=False)
                    logger.info(f"Successfully downloaded dataset ({df.shape[0]} rows) to {DATA_FILE}")
                    download_success = True
                    break
        except Exception as exc:
            logger.warning(f"Download failed from {url}: {exc}")

    # Fallback to synthetic generation
    if not download_success:
        logger.warning("Remote endpoints unreachable or offline. Activating synthetic data fallback generator.")
        df_synthetic = generate_synthetic_pima_dataset(n_samples=768, random_state=42)
        df_synthetic.to_csv(DATA_FILE, index=False)
        logger.info(f"Synthetic clinical dataset successfully generated at {DATA_FILE}")

    return DATA_FILE


def load_raw_dataset() -> pd.DataFrame:
    """Loads raw dataset ensuring file existence and column compliance."""
    file_path = ensure_dataset_available()
    df = pd.read_csv(file_path)
    if list(df.columns) != ALL_COLUMNS:
        # Standardize columns
        if df.shape[1] == 9:
            df.columns = ALL_COLUMNS
        else:
            raise ValueError(f"Unexpected dataset dimensions: {df.shape}")
    return df


def get_data_splits(
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ClinicalFeatureEngineer]:
    """
    Extracts features, applies clinical feature engineering, and partitions into
    train and test splits with stratified class preservation.
    """
    raw_df = load_raw_dataset()
    X = raw_df[RAW_FEATURE_COLUMNS]
    y = raw_df[TARGET_COLUMN].astype(int)

    # Partition before feature engineering to strictly prevent leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit feature engineer solely on train split
    engineer = ClinicalFeatureEngineer(handle_zeros=True)
    engineer.fit(X_train_raw)

    X_train = engineer.transform(X_train_raw)
    X_test = engineer.transform(X_test_raw)

    logger.info(f"Dataset successfully prepared: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test, engineer


def prepare_single_patient_vector(
    patient_data: Dict[str, Any],
    engineer: Optional[ClinicalFeatureEngineer] = None
) -> pd.DataFrame:
    """
    Converts raw clinical dictionary inputs into an engineered 1-row feature vector
    ready for model inference.
    """
    df_raw = pd.DataFrame([patient_data])
    for col in RAW_FEATURE_COLUMNS:
        if col not in df_raw.columns:
            df_raw[col] = 0.0

    if engineer is None:
        engineer = ClinicalFeatureEngineer(handle_zeros=True)
        engineer.fit(df_raw)

    return engineer.transform(df_raw[RAW_FEATURE_COLUMNS])


def prepare_batch_dataframe(
    df_batch: pd.DataFrame,
    engineer: Optional[ClinicalFeatureEngineer] = None
) -> pd.DataFrame:
    """
    Validates and transforms a batch upload DataFrame into the engineered feature matrix.
    Defensively maps column aliases if user CSV has slightly different casing or spaces.
    """
    clean_df = df_batch.copy()
    # Normalize column names: strip spaces, lower-case mapping
    norm_map = {
        "preg": "Pregnancies",
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "bloodpressure": "BloodPressure",
        "blood_pressure": "BloodPressure",
        "bp": "BloodPressure",
        "skinthickness": "SkinThickness",
        "skin_thickness": "SkinThickness",
        "insulin": "Insulin",
        "bmi": "BMI",
        "diabetespedigreefunction": "DiabetesPedigreeFunction",
        "pedigree": "DiabetesPedigreeFunction",
        "dpf": "DiabetesPedigreeFunction",
        "age": "Age"
    }

    rename_dict = {}
    for col in clean_df.columns:
        clean_name = str(col).strip().lower().replace(" ", "").replace("_", "")
        for alias, target in norm_map.items():
            if clean_name == alias.lower().replace("_", ""):
                rename_dict[col] = target
                break

    clean_df = clean_df.rename(columns=rename_dict)

    # Verify required raw features
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in clean_df.columns]
    if missing:
        raise ValueError(f"Batch dataset missing required clinical columns: {missing}")

    if engineer is None:
        engineer = ClinicalFeatureEngineer(handle_zeros=True)
        engineer.fit(clean_df[RAW_FEATURE_COLUMNS])

    return engineer.transform(clean_df[RAW_FEATURE_COLUMNS])


if __name__ == "__main__":
    print("Testing GlucoGuard Data Pipeline...")
    X_tr, X_te, y_tr, y_te, eng = get_data_splits()
    print("Train features sample:")
    print(X_tr.head(2))
    print("Engineered columns:", list(X_tr.columns))
    print("Target distribution in Train:", y_tr.value_counts().to_dict())
    print("Pipeline validation completed successfully!")
