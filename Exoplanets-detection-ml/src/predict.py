import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

def predict_new_data(csv_path):
    # Load the model and attachments
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    encoder = artifacts['encoder']
    feature_names = artifacts['features']

    # Load new data
    new_data = pd.read_csv(csv_path)
    
    # We only keep the columns the model was trained on
    # Note: In a real system, the exact same preprocessing should be repeated here.
    X_new = new_data[feature_names] 
    
    # Prediction
    preds = model.predict(X_new)
    decoded_preds = encoder.inverse_transform(preds)
    
    return decoded_preds

if __name__ == "__main__":
    # Test with existing data (for example):
    # result = predict_new_data(BASE_DIR / "data/raw/cumulative.csv")
    # print(result[:5])
    pass
