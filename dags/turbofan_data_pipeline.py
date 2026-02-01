from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
import sys
import os

# Add scripts directory to Python path
sys.path.insert(0, '/opt/airflow/scripts')

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'email': ['pratapsuryawanshi98@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def process_turbofan_data():
    """
    Task function to process NASA Turbofan data
    """
    # Import here to avoid Airflow serialization issues
    from data_processor import main
    
    print("Starting Turbofan data processing via Airflow...")
    main()
    print("Processing completed!")

def validate_processed_data():
    """
    Task to validate the processed data
    """
    import pandas as pd
    import os
    
    processed_file = "/opt/airflow/data/processed/turbofan_train_FD001.parquet"
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed file not found: {processed_file}")
    
    # Load and validate
    df = pd.read_parquet(processed_file)
    
    # Basic validations
    required_columns = ['engine_id', 'time_cycles', 'RUL', 'failure_soon']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Data quality checks
    if df['RUL'].min() < 0:
        raise ValueError("RUL has negative values!")
    
    if df['failure_soon'].notnull().sum() != len(df):
        raise ValueError("Missing values in target column!")
    
    print(f"✅ Validation passed!")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Engines: {df['engine_id'].nunique()}")
    print(f"   Failure rate: {df['failure_soon'].mean():.2%}")

# Define the DAG
with DAG(
    'turbofan_data_pipeline',
    default_args=default_args,
    description='NASA Turbofan Predictive Maintenance Data Pipeline',
    schedule_interval='@daily',  # Run daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'predictive-maintenance'],
) as dag:
    
    # Define tasks
    start_task = DummyOperator(
        task_id='start_pipeline',
        dag=dag,
    )
    
    process_data_task = PythonOperator(
        task_id='process_turbofan_data',
        python_callable=process_turbofan_data,
        dag=dag,
    )
    
    validate_data_task = PythonOperator(
        task_id='validate_processed_data',
        python_callable=validate_processed_data,
        dag=dag,
    )
    
    end_task = DummyOperator(
        task_id='end_pipeline',
        dag=dag,
    )
    
    # Define task dependencies
    start_task >> process_data_task >> validate_data_task >> end_task