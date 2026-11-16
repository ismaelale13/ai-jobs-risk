"""Probability calibration and decision threshold.

The models use class_weight="balanced", which pushes predicted probabilities up (mean ~0.51 vs. a true
decline share of ~0.39). We fit Platt scaling on the out-of-fold predictions of the TRAINING occupations
only and apply it to all rows; the held-out test set is untouched by the fit. The decision threshold is the
one that maximizes F1 on the training out-of-fold predictions.
"""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve, roc_auc_score

from src import config as C


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def run():
    scored = pd.read_parquet(C.PROCESSED / "scored.parquet")
    if "risk_raw" not in scored:
        scored["risk_raw"] = scored["risk"]
    tr, te = scored[scored["in_test_set"] == 0], scored[scored["in_test_set"] == 1]

    platt = LogisticRegression(C=1e6).fit(_logit(tr["risk_raw"].values), tr[C.TARGET])
    scored["risk"] = platt.predict_proba(_logit(scored["risk_raw"].values))[:, 1]
    joblib.dump(platt, C.ROOT / "models" / "calibrator.joblib")

    tr_cal = scored.loc[scored["in_test_set"] == 0, "risk"]
    prec, rec, thr = precision_recall_curve(tr[C.TARGET], tr_cal)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    threshold_f1 = float(thr[np.argmax(f1[:-1])])
    # F1-optimal threshold flags most rows when the signal is weak; for the dashboard we flag the same
    # share of cells as actually declined in training (top ~39 % of risk) — easier to interpret.
    threshold = float(np.quantile(tr_cal, 1 - tr[C.TARGET].mean()))

    te_cal = scored.loc[scored["in_test_set"] == 1, "risk"]
    y = te[C.TARGET]
    cal = {
        "threshold_from_train_oof": threshold,
        "threshold_max_f1_train_oof": threshold_f1,
        "test_f1_at_max_f1_threshold": float(f1_score(y, te_cal >= threshold_f1)),
        "test_accuracy_at_threshold": float(((te_cal >= threshold) == y).mean()),
        "test_mean_predicted_raw": float(te["risk_raw"].mean()),
        "test_mean_predicted_calibrated": float(te_cal.mean()),
        "test_decline_share": float(y.mean()),
        "test_brier_raw": float(brier_score_loss(y, te["risk_raw"])),
        "test_brier_calibrated": float(brier_score_loss(y, te_cal)),
        "test_roc_auc": float(roc_auc_score(y, te_cal)),  # unchanged: calibration is monotone
        "test_f1_at_threshold": float(f1_score(y, te_cal >= threshold)),
        "test_share_flagged_at_threshold": float((te_cal >= threshold).mean()),
    }
    metrics = json.loads((C.REPORTS / "metrics.json").read_text())
    metrics["calibration"] = cal
    (C.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    scored.to_parquet(C.PROCESSED / "scored.parquet", index=False)

    # reliability diagram on the held-out test set
    fig, ax = plt.subplots(figsize=(5, 5))
    for col, lab in [("risk_raw", "before calibration"), ("risk", "after calibration")]:
        pt, pp = calibration_curve(y, scored.loc[scored["in_test_set"] == 1, col], n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.set_xlabel("Predicted probability of decline")
    ax.set_ylabel("Observed share declined")
    ax.set_title("Calibration on held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.REPORTS / "calibration.png", dpi=150)
    plt.close(fig)
    return cal


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
