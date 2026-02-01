import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib
import json
import os
import sys
from datetime import datetime

class TurbofanModelTrainer:
    def __init__(self, data_path, model_save_path, metrics_save_path):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.metrics_save_path = metrics_save_path
        self.model = None
        self.metrics = {}
        
    def load_data(self):
        """Load processed data from Parquet file"""
        print(f"Loading data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Separate features and target
        # Exclude non-feature columns
        exclude_cols = ['engine_id', 'time_cycles', 'RUL', 'failure_soon']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['failure_soon']
        
        print(f"Data shape: {X.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y, feature_cols
    
    def prepare_features(self, X):
        """Prepare features for training"""
        # Handle any missing values
        X = X.fillna(X.mean())
        return X
    
    def train_model(self, X_train, y_train):
        """Train Random Forest model with hyperparameter tuning"""
        print("Training Random Forest model with hyperparameter tuning...")
        
        # Define base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Define hyperparameter grid (small for quick training)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='f1',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'precision': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'],
            'recall': classification_report(y_test, y_pred, output_dict=True)['weighted avg']['recall']
        }
        
        # Print metrics
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        self.metrics = metrics
        return metrics
    
    def save_model(self):
        """Save trained model and metrics"""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_save_path), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Save metrics
        with open(self.metrics_save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"Metrics saved to {self.metrics_save_path}")
        
        # Also save feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = self.model_save_path.replace('.joblib', '_feature_importance.csv')
            feature_importance.to_csv(importance_path, index=False)
            print(f"Feature importance saved to {importance_path}")
    
    def run_training(self):
        """Main training pipeline"""
        print("\n" + "="*50)
        print("STARTING TURBOFAN MODEL TRAINING")
        print("="*50)
        
        # Step 1: Load data
        X, y, feature_cols = self.load_data()
        
        # Step 2: Prepare features
        X = self.prepare_features(X)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        
        # Step 4: Train model
        self.train_model(X_train, y_train)
        
        # Step 5: Evaluate model
        self.evaluate_model(X_test, y_test)
        
        # Step 6: Save model and metrics
        self.save_model()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)

def main():
    """Main function with configurable paths"""
    # Configurable paths (can be overridden by environment variables)
    data_path = os.getenv('DATA_PATH', '/opt/airflow/data/processed/turbofan_train_FD001.parquet')
    model_save_path = os.getenv('MODEL_SAVE_PATH', '/opt/airflow/models/turbofan_model_v1.joblib')
    metrics_save_path = os.getenv('METRICS_SAVE_PATH', '/opt/airflow/models/training_metrics_v1.json')
    
    # Create trainer and run
    trainer = TurbofanModelTrainer(data_path, model_save_path, metrics_save_path)
    trainer.run_training()

if __name__ == "__main__":
    main()