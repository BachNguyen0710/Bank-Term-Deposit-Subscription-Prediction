# src/metrics_utils.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, classification_report, confusion_matrix
)
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def evaluate_proba(y_true, y_proba, outdir: Path | None = None, prefix="baseline"):
    y_pred = (y_proba >= 0.5).astype(int)
    roc = roc_auc_score(y_true, y_proba)
    pr  = average_precision_score(y_true, y_proba)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
    }
    df_metrics = pd.DataFrame([metrics])

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(outdir / f"{prefix}_metrics.csv", index=False)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC={roc:.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.legend()
        plt.savefig(outdir / f"{prefix}_roc.png", bbox_inches="tight")
        plt.close()

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        plt.figure()
        plt.plot(rec, prec, label=f"PR AUC={pr:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
        plt.legend()
        plt.savefig(outdir / f"{prefix}_pr.png", bbox_inches="tight")
        plt.close()

    return df_metrics, cm
