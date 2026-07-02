import shap
import matplotlib.pyplot as plt
from pathlib import Path


def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values


def plot_shap_summary(shap_values, X_sample, save_path=None):
    shap.summary_plot(shap_values, X_sample, show=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close()
