import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv
from training.features import build_features

# Load environment variables
load_dotenv()

def main():
    endpoint_name = os.getenv("ENDPOINT_NAME", "claims-triage-endpoint")
    region = os.getenv("AWS_REGION", "us-east-1")
    dataset_path = 'notebooks/data/insurance_claims.csv'
    
    print(f"Testing SageMaker Endpoint: {endpoint_name}")
    print(f"AWS Region: {region}")
    
    # Load dataset and prepare a single record
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    df_processed = build_features(df)
    
    # Drop target column to simulate inference
    X = df_processed.drop(columns=['fraud_reported'])
    
    # Take the first row as a dictionary
    sample_record = X.iloc[0].to_dict()
    print("\nSample input feature record:")
    print(json.dumps(sample_record, indent=2))
    
    # Invoke SageMaker Endpoint
    print(f"\nInvoking endpoint '{endpoint_name}' via boto3...")
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(sample_record)
        )
        
        # Read and decode response
        result_payload = response['Body'].read().decode('utf-8')
        result = json.loads(result_payload)
        
        print("\nPrediction Response:")
        print(json.dumps(result, indent=2))
        
        # Verify the format
        if "probabilities" in result and "predictions" in result:
            prob = result["probabilities"][0]
            pred = result["predictions"][0]
            print(f"\nTest Success! Probability of fraud: {prob:.4f}, Prediction label: {pred}")
        else:
            print("\nWarning: Response payload format differs from expected.")
            
    except Exception as e:
        print(f"\nError invoking endpoint: {e}")
        print("Make sure the endpoint is fully deployed and active before running this script.")

if __name__ == '__main__':
    main()
