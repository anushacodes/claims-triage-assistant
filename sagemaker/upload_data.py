import os
import boto3
import pandas as pd
from dotenv import load_dotenv
from training.features import build_features

# Load environment variables
load_dotenv()

def main():
    bucket_name = os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET not set in environment or .env file")
        
    raw_data_path = 'notebooks/data/insurance_claims.csv'
    processed_local_path = 'training/artifacts/processed_claims.csv'
    s3_key = 'data/processed_claims.csv'
    
    print(f"Loading raw data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at {raw_data_path}")
        
    df = pd.read_csv(raw_data_path)
    
    print("Preprocessing data with build_features...")
    df_processed = build_features(df)
    
    # Ensure artifacts directory exists
    os.makedirs('training/artifacts', exist_ok=True)
    
    print(f"Saving processed data locally to: {processed_local_path}")
    df_processed.to_csv(processed_local_path, index=False)
    
    # Upload to S3
    print(f"Uploading to S3 bucket '{bucket_name}' at key '{s3_key}'...")
    s3_client = boto3.client('s3')
    
    try:
        s3_client.upload_file(processed_local_path, bucket_name, s3_key)
        print("Upload successful!")
        print(f"S3 URI: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e

if __name__ == '__main__':
    main()
