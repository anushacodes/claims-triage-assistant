import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from training.features import build_features, FEATURE_COLS

ARTIFACTS_DIR = 'training/artifacts'
DATASET_PATH  = 'notebooks/data/insurance_claims.csv'
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, 'model.joblib')
STATS_PATH    = os.path.join(ARTIFACTS_DIR, 'training_stats.json')
PROCESSED_CSV = os.path.join(ARTIFACTS_DIR, 'processed_claims.csv')


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print(f"Loading dataset from: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Place it there first.")

    df = pd.read_csv(DATASET_PATH)

    print("Building features...")
    df_processed = build_features(df)

    # Save processed CSV so SageMaker upload script has a ready artifact
    df_processed.to_csv(PROCESSED_CSV, index=False)
    print(f"Saved processed dataset to {PROCESSED_CSV}")

    X = df_processed.drop(columns=['fraud_reported'])
    y = df_processed['fraud_reported']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # Train base XGBoost then wrap with isotonic calibration.
    # Calibration aligns raw probabilities with empirical frequencies,
    # making confidence thresholds in triage_decision() more reliable.
    print("Training XGBoost + isotonic calibration...")
    base = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss',
    )
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X_train, y_train)

    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    # Quick summary metrics
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test)[:, 1]
    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    test_auc  = roc_auc_score(y_test, y_prob)

    print("\nTraining Metrics Summary:")
    print(f"  Train Accuracy : {train_acc:.4f}")
    print(f"  Test Accuracy  : {test_acc:.4f}")
    print(f"  Test ROC-AUC   : {test_auc:.4f}")

    # Save training feature stats for Phase 6 drift monitoring
    stats = {
        col: {
            "mean": float(X_train[col].mean()),
            "std":  float(X_train[col].std()),
            "min":  float(X_train[col].min()),
            "max":  float(X_train[col].max()),
        }
        for col in FEATURE_COLS
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Saved training feature stats to {STATS_PATH}")
    print("Training complete.")


if __name__ == '__main__':
    main()
