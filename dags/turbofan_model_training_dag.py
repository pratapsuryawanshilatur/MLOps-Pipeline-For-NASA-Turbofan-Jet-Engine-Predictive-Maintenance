from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

sys.path.insert(0, '/opt/airflow/scripts')

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_data_exists():
    """Check if processed data exists"""
    import os
    data_path = '/opt/airflow/data/processed/turbofan_train_FD001.parquet'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    print(f"Data found: {data_path}")
    print(f"  Size: {os.path.getsize(data_path) / 1024 / 1024:.2f} MB")

def train_model_locally():
    """Train model using the script"""
    import sys
    sys.path.insert(0, '/opt/airflow/scripts')

    # Create models directory if it doesn't exist
    import os
    models_dir = '/opt/airflow/models'
    os.makedirs(models_dir, exist_ok=True)

    # Try to fix permissions (optional)
    try:
        os.chmod(models_dir, 0o777)
    except:
        pass # Ignore permission erros
    
    from train_model import main
    
    print("Starting model training...")
    main()
    print("Model training completed!")

def validate_model():
    """Validate trained model"""
    import joblib
    import json
    import os
    
    model_path = '/opt/airflow/models/turbofan_model_v1.joblib'
    metrics_path = '/opt/airflow/models/training_metrics_v1.json'
    
    # Check files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    
    # Load and validate model
    model = joblib.load(model_path)
    print(f"Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Model parameters: {model.get_params()}")
    
    # Load and validate metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"Metrics loaded successfully")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Check if metrics meet minimum requirements
    if metrics['f1_score'] < 0.6:
        print(f"Warning: F1 score is low: {metrics['f1_score']:.4f}")
    
    print("Model validation passed!")

with DAG(
    'turbofan_model_training',
    default_args=default_args,
    description='Train predictive maintenance model',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'training'],
) as dag:
    
    start = DummyOperator(task_id='start')
    
    check_data = PythonOperator(
        task_id='check_data_exists',
        python_callable=check_data_exists
    )
    
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model_locally
    )
    
    validate_trained_model = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model
    )
    
    end = DummyOperator(task_id='end')
    
    # Set dependencies
    start >> check_data >> train_model >> validate_trained_model >> end