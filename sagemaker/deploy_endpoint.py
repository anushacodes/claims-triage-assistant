import os
import boto3
import sagemaker
from sagemaker import ModelPackage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_sagemaker_config():
    config_path = 'training/artifacts/sagemaker_config.env'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found. Did you run register_model.py?")
    
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                config[k] = v
    return config

def update_dotenv_file(endpoint_name):
    dotenv_path = '.env'
    lines = []
    updated = False
    
    if os.path.exists(dotenv_path):
        with open(dotenv_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if line.startswith('ENDPOINT_NAME='):
                lines[i] = f"ENDPOINT_NAME={endpoint_name}\n"
                updated = True
                break
                
    if not updated:
        lines.append(f"\nENDPOINT_NAME={endpoint_name}\n")
        
    with open(dotenv_path, 'w') as f:
        f.writelines(lines)
    print(f"Updated {dotenv_path} with ENDPOINT_NAME={endpoint_name}")

def main():
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    endpoint_name = os.getenv("ENDPOINT_NAME", "claims-triage-endpoint")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not role_arn:
        raise ValueError("SAGEMAKER_ROLE_ARN must be set in .env")
        
    config = read_sagemaker_config()
    model_package_arn = config.get("MODEL_PACKAGE_ARN")
    if not model_package_arn:
        raise ValueError("MODEL_PACKAGE_ARN not found in config")
        
    print(f"Deploying model from package: {model_package_arn}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Region: {region}")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Create ModelPackage object
    model_package = ModelPackage(
        role=role_arn,
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session
    )
    
    print("Launching deployment (spinning up EC2 instance ml.t2.medium)...")
    print("Note: This can take 5-10 minutes. Please wait...")
    
    # Deploy to ml.t2.medium (cost efficient)
    predictor = model_package.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name=endpoint_name
    )
    
    print(f"\nEndpoint deployed successfully!")
    print(f"Endpoint Name: {predictor.endpoint_name}")
    
    # Update local .env file
    update_dotenv_file(predictor.endpoint_name)

if __name__ == '__main__':
    main()
