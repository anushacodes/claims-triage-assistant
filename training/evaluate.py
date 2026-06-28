import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve
import joblib

from training.features import build_features

def triage_decision(prob: float) -> str:
    """
    Determines the decision category based on the predicted fraud probability.
    - >= 0.90: Auto-reject (high risk of fraud)
    - >= 0.55: Manual review (moderate risk / uncertain)
    - < 0.55: Auto-approve (low risk of fraud)
    """
    if prob >= 0.90:
        return "AUTO_REJECT"
    elif prob >= 0.55:
        return "MANUAL_REVIEW"
    else:
        return "AUTO_APPROVE"

def main():
    dataset_path = 'notebooks/data/insurance_claims.csv'
    model_path = 'training/artifacts/model.joblib'
    eval_results_path = 'training/artifacts/eval_results.json'
    
    print("Loading test data...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    df_processed = build_features(df)
    
    X = df_processed.drop(columns=['fraud_reported'])
    y = df_processed['fraud_reported']
    
    # Split using the exact same random state and stratify
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Please run train.py first.")
        
    model = joblib.load(model_path)
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate classification metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_pred_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    
    # Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='uniform')
    
    # Confidence Histogram
    counts, bin_edges = np.histogram(y_pred_prob, bins=10, range=(0, 1))
    
    # Prepare results for JSON export
    results = {
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "precision": float(report["1"]["precision"]),
            "recall": float(report["1"]["recall"]),
            "f1_score": float(report["1"]["f1-score"]),
            "roc_auc": float(auc)
        },
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "calibration": {
            "true_probabilities": [float(x) for x in prob_true],
            "pred_probabilities": [float(x) for x in prob_pred]
        },
        "confidence_histogram": {
            "counts": [int(x) for x in counts],
            "bin_edges": [float(x) for x in bin_edges]
        }
    }
    
    # Calculate triage decisions coverage
    decisions = [triage_decision(p) for p in y_pred_prob]
    decisions_series = pd.Series(decisions)
    coverage = decisions_series.value_counts(normalize=True).to_dict()
    counts_dict = decisions_series.value_counts().to_dict()
    
    # Fill in missing options just in case
    for choice in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_REJECT"]:
        if choice not in coverage:
            coverage[choice] = 0.0
        if choice not in counts_dict:
            counts_dict[choice] = 0
            
    print("\nTriage Decision Coverage on Test Set:")
    for choice in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_REJECT"]:
        print(f"  {choice}: {counts_dict[choice]} ({coverage[choice]*100:.2f}%)")
        
    results["triage_coverage"] = {
        "percentages": {k: float(v) for k, v in coverage.items()},
        "counts": {k: int(v) for k, v in counts_dict.items()}
    }
    
    print(f"Saving evaluation results to {eval_results_path}...")
    with open(eval_results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation completed successfully.")

if __name__ == '__main__':
    main()
