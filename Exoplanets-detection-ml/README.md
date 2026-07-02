# 🪐 Exoplanet Detection using Machine Learning (Kepler Data)

This project aims to classify Kepler Objects of Interest (KOI) into categories: **Confirmed**, **Candidate**, or **False Positive** using advanced Gradient Boosting algorithms.

## 🚀 Key Highlights
- **Leakage Prevention:** Unlike many basic notebooks, this project identifies and removes "leakage columns" (like `koi_fpflag_*`) that contain information about the target labels before analysis. This ensures the model learns actual physics, not just metadata artifacts.
- **Explainable AI (XAI):** Integrated **SHAP** to interpret model decisions and visualize feature importance and interactions.
- **Modular Architecture:** Refactored from a messy notebook into a professional Python package structure (`src/` module).
- **Model Comparison:** Evaluates XGBoost, LightGBM, and CatBoost.

## 📊 Results & Interpretability
After removing data leakage, the **LightGBM** model achieved:
- **Accuracy:** ~86%
- **ROC-AUC:** ~0.97 (Stable across classes)

### SHAP Interaction Analysis
![SHAP Interaction](results/shap_beeswarm.png)
The SHAP interaction plot reveals how physical features like `koi_period` and `koi_score` work together. For instance, the model's sensitivity to the `koi_score` changes significantly based on the planetary orbital period.

## Project Structure
```text
├── data/               # Raw dataset (cumulative.csv)
├── results/            # Saved SHAP plots and metrics
├── src/                # Source code
│   ├── data_loader.py  # Data ingestion
│   ├── preprocess.py   # Cleaning and Leakage removal
│   ├── features.py     # Imputation and Scaling
│   ├── train.py        # Training pipeline (K-Fold CV)
│   ├── evaluate.py     # Evaluation metrics
│   ├── explain.py      # SHAP interpretability
│   └── predict.py      # Inference for new data
└── main.py             # Entry point (Optional)


## How to Run:

1.Clone the repo

2.Install dependencies:
   pip install -r requirements.txt
   
3.Train and Evaluate::
   python -m src.train
   
   
## Skills Demonstrated
- Machine Learning: Gradient Boosting (LGBM, XGB, CatBoost), Stratified K-Fold CV.
- Data Engineering: Feature Engineering, Data Cleaning, Leakage Analysis.
- Tools: Python, Scikit-learn, SHAP, Pandas, Matplotlib.
