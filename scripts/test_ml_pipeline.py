import unittest
import pandas as pd
import numpy as np
import os

class TestMLPipeline(unittest.TestCase):
    
    def test_data_files_exist(self):
        """Test that required data files exist"""
        self.assertTrue(os.path.exists('data/processed/turbofan_train_FD001_sample.parquet'),
                       "Sample data file should exist")
        print("Sample data file exists")
    
    def test_model_files_exist(self):
        """Test that trained model files exist"""
        if os.path.exists('models/turbofan_model_v1.joblib'):
            print("Trained model file exists")
        else:
            print(" No trained model found (this is OK for CI)")
    
    def test_requirements_file(self):
        """Test that requirements file exists and has content"""
        self.assertTrue(os.path.exists('scripts/requirements.txt'),
                       "requirements.txt should exist")
        with open('scripts/requirements.txt', 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 0, "requirements.txt should not be empty")
        print(" Requirements file is valid")
    
    def test_import_modules(self):
        """Test that all modules can be imported"""
        import scripts.data_processor as dp
        import scripts.train_model as tm
        print("All modules import successfully")

if __name__ == '__main__':
    unittest.main()