import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.data_loader import load_data
from src.preprocess import clean_data, encode_target
from src.features import prepare_features
from src.evaluate import evaluate_model
from src.explain import compute_shap_values, plot_shap_summary
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "cumulative.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_training():
    # 1. Load & Preprocess
    df = load_data(DATA_PATH)
    df = clean_data(df)
    # encode target + build X (numeric-only) inside preprocess.encode_target
    X, y, label_encoder = encode_target(df, "koi_disposition")
    
    # Keep column names before converting to Numpy
    feature_names = X.columns.tolist()

    #2. Cross-Validation (LGBM as an example because it was the best)
    model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # 3. Final Train/Test Split for Evaluation & SHAP
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_scaled, X_test_scaled = prepare_features(X_train, X_test)
    
    # Convert back to DataFrame to preserve column names in SHAP
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate
    # 4.1. First predict the model on the test data
    y_pred = model.predict(X_test_scaled)

    # 4.2. If the Probability model also outputs (necessary for ROC-AUC)
    y_prob = model.predict_proba(X_test_scaled)

    # 4.3. Now pass the actual values ​​and predicted values ​​to the function
    metrics = evaluate_model(y_test, y_pred, y_prob)
    print("Evaluation Metrics:", metrics)

    # 5. SHAP Explanation
    print("Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X_test_df)
    plot_shap_summary(explainer, shap_values, X_test_df, save_path=RESULTS_DIR / "shap_beeswarm.png")

    # 6. Save Model & Encoder
    joblib.dump({'model': model, 'encoder': label_encoder, 'features': feature_names}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    run_training()
