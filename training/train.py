import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

from training.features import build_features

def main():
    # Define paths
    dataset_path = 'notebooks/data/insurance_claims.csv'
    artifacts_dir = 'training/artifacts'
    model_path = os.path.join(artifacts_dir, 'model.joblib')
    
    # Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please place it there first.")
        
    df = pd.read_csv(dataset_path)
    
    # Process features
    print("Processing features...")
    df_processed = build_features(df)
    
    # Split features and target
    X = df_processed.drop(columns=['fraud_reported'])
    y = df_processed['fraud_reported']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # Train the XGBoost model
    print("Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    
    # Simple evaluations
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    # Predict probabilities for ROC-AUC
    from sklearn.metrics import roc_auc_score
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_prob)
    
    print("\nTraining Metrics Summary:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    print("Training complete.")

if __name__ == '__main__':
    main()
