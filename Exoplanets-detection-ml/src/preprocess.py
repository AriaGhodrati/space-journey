# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess dataset.
    """

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove leakage / identifier columns
    leakage_columns = [
        "rowid",
        "kepid",
        "kepoi_name",
        "kepler_name",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_fpflag_nt",
        "koi_fpflag_ss"
    ]

    existing_cols = [col for col in leakage_columns if col in df.columns]

    df = df.drop(columns=existing_cols)

    return df


def encode_target(df: pd.DataFrame, target_column: str):
    """
    Encode categorical target labels.
    """

    encoder = LabelEncoder()

    y = encoder.fit_transform(df[target_column])

    X = df.drop(columns=[target_column])
    
    # Keeping only numerical columns
    X = X.select_dtypes(include=["number"])

    return X, y, encoder
