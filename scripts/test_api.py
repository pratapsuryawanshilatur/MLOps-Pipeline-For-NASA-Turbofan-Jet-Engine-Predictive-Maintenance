import requests
import json
import numpy as np

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting model info endpoint...")
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Model type: {data.get('model_type')}")
    print(f"Features count: {data.get('features_count')}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\nTesting single prediction endpoint...")
    
    # Create sample features (24 values as expected by model)
    sample_features = [100.0, 0.25, 0.0] + list(np.random.rand(21))
    
    payload = {
        "features": sample_features
    }
    
    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result.get('prediction')}")
        print(f"Probability: {result.get('probability'):.4f}")
        print(f"Engine Status: {result.get('engine_status')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\nTesting batch prediction endpoint...")
    
    # Create 3 sample engine readings
    engine_readings = []
    for i in range(3):
        features = [100.0 + i, 0.25, 0.0] + list(np.random.rand(21))
        engine_readings.append(features)
    
    payload = {
        "engine_readings": engine_readings
    }
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total predictions: {result.get('total_predictions')}")
        print(f"Critical count: {result.get('critical_count')}")
        for pred in result.get('predictions', [])[:2]:  # Show first 2
            print(f"  Engine {pred['engine_id']}: {pred['engine_status']}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    print("=== NASA Turbofan Predictive Maintenance API Tests ===\n")
    
    # Wait for API to start
    import time
    print("Waiting for API to start...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        try:
            print(f"{test_name}: {status}")
        except UnicodeError:
            # Windows encoding fallback
            status_text = "PASS" if success else "FAIL"
            prin(f"{test_name}: {status_text}")
    
    all_passed = all(success for _, success in results)
    try:
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    except UnicodeEncodeError:
        print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)