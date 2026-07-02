# src/train.py

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from data_loader import load_data
from preprocess import clean_data, encode_target
from features import prepare_features
from evaluate import evaluate_model



# Paths


DATA_PATH = "../data/raw/cumulative.csv"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)



# Load Data

print("Loading dataset...")

df = load_data(DATA_PATH)

print(f"Dataset shape: {df.shape}")



# Preprocessing

print("Cleaning dataset...")

df = clean_data(df)

TARGET_COLUMN = "koi_disposition"

X, y, encoder = encode_target(df, TARGET_COLUMN)



# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)



# Feature Preparation


X_train, X_test = prepare_features(X_train, X_test)



# Models


models = {
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    ),

    "LightGBM": LGBMClassifier(
        random_state=42,
    ),

    "CatBoost": CatBoostClassifier(
        verbose=0,
        random_state=42,
        allow_writing_files=False,
    ),
}


# Training Loop


results = []

best_model = None
best_score = 0


for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # y_proba = model.predict_proba(X_test)[:, 1]
    y_proba = model.predict_proba(X_test)

    metrics = evaluate_model(y_test, y_pred, y_proba)

    metrics["model"] = model_name

    results.append(metrics)

    print(metrics)

    # Save best model
    if metrics["f1_score"] > best_score:

        best_score = metrics["f1_score"]

        best_model = model

        model_path = os.path.join(
            MODEL_DIR,
            f"best_model_{model_name}.pkl"
        )

        joblib.dump(model, model_path)



# Final Results


results_df = pd.DataFrame(results)

print("\nFinal Results:")
print(results_df.sort_values(by="f1_score", ascending=False))
