import shap
import matplotlib.pyplot as plt
from pathlib import Path

def compute_shap_values(model, X_sample):
    # Using TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values

def plot_shap_summary(explainer, shap_values, X_sample, save_path=None):
    plt.figure(figsize=(10, 6))
    
    # In multiclass models, SHAP returns a list. 
    # We usually plot the importance of features for the whole or a specific class (e.g. confirmed planets).
    # Here, for simplicity and clarity, we draw a general summary.
    shap.summary_plot(shap_values, X_sample, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
