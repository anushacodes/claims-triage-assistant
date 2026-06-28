import os
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN")
    bucket_name = os.getenv("S3_BUCKET")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    if not role_arn or not bucket_name:
        raise ValueError("SAGEMAKER_ROLE_ARN and S3_BUCKET must be set in .env")
        
    print(f"Using SageMaker Role: {role_arn}")
    print(f"Using S3 Bucket: {bucket_name}")
    print(f"AWS Region: {region}")
    
    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    
    # Configure SageMaker SKLearn Estimator
    # The source_dir points to training/, which contains sagemaker_train.py.
    # SageMaker will package the whole training/ folder and send it to the training container.
    # Note: We need xgboost inside the container, so we'll make sure training/requirements.txt is present.
    print("Configuring SKLearn Estimator...")
    estimator = SKLearn(
        entry_point='sagemaker_train.py',
        source_dir='training',
        role=role_arn,
        instance_count=1,
        instance_type='ml.m5.large', # Cost-efficient standard instance
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    
    # Define input channel
    input_s3_uri = f"s3://{bucket_name}/data/"
    print(f"Starting training job. Input S3 location: {input_s3_uri}")
    
    # Launch training job (synchronous by default, printing logs)
    estimator.fit({'train': input_s3_uri})
    
    # Get training job name
    job_name = estimator.latest_training_job.name
    print(f"\nTraining job completed successfully!")
    print(f"Job Name: {job_name}")
    print(f"Model Artifact S3 URI: {estimator.model_data}")
    
    # Write the job name and model artifact S3 URI to a temp config file
    # so register_model.py and deploy_endpoint.py can read it
    config_path = 'training/artifacts/sagemaker_config.env'
    os.makedirs('training/artifacts', exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(f"TRAINING_JOB_NAME={job_name}\n")
        f.write(f"MODEL_ARTIFACT_S3={estimator.model_data}\n")
    print(f"Saved job config to {config_path}")

if __name__ == '__main__':
    main()
