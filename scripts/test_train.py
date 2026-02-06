# Quick test to verify training works
import sys
import os

# Add scripts to path
sys.path.append('.')

try:
    from train_model import TurbofanModelTrainer
    
    # Test with small data if it exists
    sample_path = 'data/processed/turbofan_train_FD001_sample.parquet'
    
    if os.path.exists(sample_path):
        print(f"Testing with sample data: {sample_path}")
        trainer = TurbofanModelTrainer(
            data_path=sample_path,
            model_save_path='models/test_model_ci.joblib',
            metrics_save_path='models/test_metrics_ci.json'
        )
        
        # Run training
        trainer.run_training()
        print("CI test training completed successfully!")
    else:
        print("Sample data not found, skipping training test")
        
except Exception as e:
    print(f"Test failed: {e}")
    sys.exit(1)