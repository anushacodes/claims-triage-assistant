import os
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_sagemaker_config():
    config_path = 'training/artifacts/sagemaker_config.env'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found. Did you run train_job.py?")
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                config[k] = v
    return config

def main():
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not role_arn:
        raise ValueError("SAGEMAKER_ROLE_ARN must be set in .env")
        
    # Read the model S3 location from config
    config = read_sagemaker_config()
    model_s3_uri = config.get("MODEL_ARTIFACT_S3")
    if not model_s3_uri:
        raise ValueError("MODEL_ARTIFACT_S3 not found in config")
        
    print(f"Registering model from S3 location: {model_s3_uri}")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Create SKLearnModel to register
    model = SKLearnModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point='sagemaker_train.py',
        source_dir='training',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    
    model_package_group_name = "claims-triage-model-group"
    print(f"Registering model in Group: {model_package_group_name}...")
    
    # Register the model in the SageMaker Model Registry with approval status
    model_package = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status="Approved",
        description="XGBoost model trained on insurance claims data for fraud classification"
    )
    
    model_package_arn = model_package.model_package_arn
    print(f"\nModel registered successfully!")
    print(f"Model Package Group: {model_package_group_name}")
    print(f"Model Package ARN: {model_package_arn}")
    
    # Append the package ARN to sagemaker_config.env
    config_path = 'training/artifacts/sagemaker_config.env'
    with open(config_path, 'a') as f:
        f.write(f"MODEL_PACKAGE_ARN={model_package_arn}\n")
    print(f"Updated config file with MODEL_PACKAGE_ARN")

if __name__ == '__main__':
    main()
