import argparse
import io
import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model/'))
    parser.add_argument('--train',     type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data/'))
    # parse_known_args so SageMaker's injected flags don't cause errors
    args, _ = parser.parse_known_args()

    data_path = os.path.join(args.train, 'processed_claims.csv')
    print(f"Loading dataset from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    X  = df.drop(columns=['fraud_reported'])
    y  = df['fraud_reported']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    print("Training XGBoost + isotonic calibration...")
    base  = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          random_state=42, eval_metric='logloss')
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X_train, y_train)

    model_path = os.path.join(args.model_dir, 'model.joblib')
    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)

    train_acc = model.score(X_train, y_train)
    test_acc  = model.score(X_test,  y_test)
    test_auc  = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"train_accuracy={train_acc:.4f}; test_accuracy={test_acc:.4f}; test_auc={test_auc:.4f}")

    # Write metrics so SageMaker Model Registry can surface them
    metrics = {"train_accuracy": train_acc, "test_accuracy": test_acc, "test_auc": test_auc}
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    print("SageMaker training complete.")



def model_fn(model_dir):
    """
    Loads the model from the model directory.
    Called automatically by the SageMaker hosting container.
    """
    model_path = os.path.join(model_dir, 'model.joblib')
    print(f"Loading model from path: {model_path}")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """
    Deserializes the incoming request body into a pandas DataFrame.
    Supports application/json and text/csv.
    """
    print(f"Received request body. Content Type: {request_content_type}")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        # Ensure single-row dictionary inputs are wrapped in a list
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
        return df
    elif request_content_type == 'text/csv':
        df = pd.read_csv(io.StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Applies the model to predict fraud probability and hard labels.
    """
    print("Executing model prediction...")
    # Get probabilities of class 1 (fraud)
    probabilities = model.predict_proba(input_data)[:, 1]
    # Get class predictions (0 or 1)
    predictions = model.predict(input_data)
    
    result = {
        "probabilities": [float(p) for p in probabilities],
        "predictions": [int(p) for p in predictions]
    }
    return result

def output_fn(prediction, content_type):
    """
    Serializes the prediction result to the client.
    """
    print("Formatting prediction output...")
    if content_type == 'application/json':
        return json.dumps(prediction), content_type
    raise ValueError(f"Unsupported content type: {content_type}")
