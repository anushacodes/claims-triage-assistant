import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from training.features import build_features

DATASET_PATH     = 'notebooks/data/insurance_claims.csv'
MODEL_PATH       = 'training/artifacts/model.joblib'
EVAL_RESULTS_PATH = 'training/artifacts/eval_results.json'

# Thresholds are calibrated to the isotonic-calibrated model's output range (~0.0–0.70).
# Derived from the training-set probability percentile distribution:
#   >= 0.60 → top ~5% of scores — high-confidence fraud
#   >= 0.35 → above-median probability — uncertain, needs review
#   <  0.35 → low fraud probability — safe to auto-approve
AUTO_REJECT_THRESHOLD    = 0.60
MANUAL_REVIEW_THRESHOLD  = 0.35


def triage_decision(prob: float) -> str:
    """
    Routes a claim based on its fraud probability.

    Thresholds are tuned to the isotonic-calibrated model output range (~0.0–0.70):
    - >= 0.60 : AUTO_REJECT   — top ~5% of scores, high-confidence fraud
    - >= 0.35 : MANUAL_REVIEW — above-median probability, needs human judgment
    - <  0.35 : AUTO_APPROVE  — low fraud risk, safe to process automatically
    """
    if prob >= AUTO_REJECT_THRESHOLD:
        return "AUTO_REJECT"
    if prob >= MANUAL_REVIEW_THRESHOLD:
        return "MANUAL_REVIEW"
    return "AUTO_APPROVE"


def main():
    print("Loading test data...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    df_processed = build_features(df)

    X = df_processed.drop(columns=['fraud_reported'])
    y = df_processed['fraud_reported']

    # Same split as train.py — reproduces identical holdout set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found. Run train.py first.")

    model = joblib.load(MODEL_PATH)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Core metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    auc    = roc_auc_score(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='uniform')

    # Confidence histogram
    hist_counts, bin_edges = np.histogram(y_pred_prob, bins=10, range=(0, 1))

    # Triage band coverage + per-band precision
    decisions = pd.Series([triage_decision(p) for p in y_pred_prob])
    y_test_reset = y_test.reset_index(drop=True)

    band_stats: dict = {}
    for band in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_REJECT"]:
        mask = decisions == band
        n = int(mask.sum())
        if n > 0:
            fraud_rate = float(y_test_reset[mask].mean())
        else:
            fraud_rate = 0.0
        band_stats[band] = {
            "count":      n,
            "pct":        round(n / len(decisions), 4),
            "fraud_rate": round(fraud_rate, 4),
        }

    print("\nTriage Band Coverage:")
    for band, s in band_stats.items():
        print(f"  {band}: {s['count']} ({s['pct']*100:.1f}%)  fraud_rate={s['fraud_rate']:.2%}")

    results = {
        "thresholds": {
            "auto_reject":   AUTO_REJECT_THRESHOLD,
            "manual_review": MANUAL_REVIEW_THRESHOLD,
        },
        "metrics": {
            "accuracy":  float(report["accuracy"]),
            "precision": float(report["1"]["precision"]),
            "recall":    float(report["1"]["recall"]),
            "f1_score":  float(report["1"]["f1-score"]),
            "roc_auc":   float(auc),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "calibration": {
            "true_probabilities": [float(x) for x in prob_true],
            "pred_probabilities": [float(x) for x in prob_pred],
        },
        "confidence_histogram": {
            "counts":    [int(x) for x in hist_counts],
            "bin_edges": [float(x) for x in bin_edges],
        },
        "triage_coverage": band_stats,
    }

    with open(EVAL_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved evaluation results to {EVAL_RESULTS_PATH}")
    print("Evaluation complete.")


if __name__ == '__main__':
    main()
