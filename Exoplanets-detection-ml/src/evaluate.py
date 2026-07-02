# src/evaluate.py

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba):

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted"
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted"
        ),

        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted"
        ),
    }

    # ROC-AUC for multiclass
    try:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="weighted"
        )
    except Exception:
        metrics["roc_auc"] = None

    return metrics
