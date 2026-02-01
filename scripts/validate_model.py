import joblib
import json
import pandas as pd

# Load the trained model
model_path = 'models/turbofan_model_v1.joblib'
metrics_path = 'models/training_metrics_v1.json'

print("Loading trained model...")
model = joblib.load(model_path)
print(f"Model loaded: {type(model).__name__}")
print(f"Model parameters: {model.get_params()}")

# Load metrics
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

print("\nModel Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# Check if metrics meet minimum requirements
print("\nValidation Check:")
if metrics['f1_score'] > 0.7:
    print("✅ F1 score meets minimum requirement (> 0.7)")
else:
    print("❌ F1 score too low")

if metrics['roc_auc'] > 0.8:
    print("✅ ROC AUC meets minimum requirement (> 0.8)")
else:
    print("❌ ROC AUC too low")

# Load feature importance
feature_importance_path = 'models/turbofan_model_v1_feature_importance.csv'
try:
    feature_importance = pd.read_csv(feature_importance_path)
    print(f"\nTop 5 important features:")
    print(feature_importance.head(5).to_string(index=False))
except:
    print("\nFeature importance file not found")

print("\n✅ Model validation complete!")