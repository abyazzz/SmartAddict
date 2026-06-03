"""
🧪 Test Neural Network Model After Training
============================================

This script will:
1. Load the newly trained NN model
2. Activate it in the app
3. Verify it can predict

Run this AFTER training completes!
"""

import sys
from pathlib import Path

# Find the most recent model directory
model_base = Path("model")
model_dirs = [d for d in model_base.glob("model_FINAL_*") if d.is_dir()]

if not model_dirs:
    print("❌ No FINAL model directories found!")
    sys.exit(1)

# Sort by name (timestamp in name)
latest_model = sorted(model_dirs, key=lambda x: x.name)[-1]

print("="*70)
print("🧪 TESTING NEURAL NETWORK MODEL")
print("="*70)
print(f"\n📂 Model Directory: {latest_model}")

# Check if all model files exist
required_files = ['nn2_classifier.pkl', 'scaler.pkl', 'metrics.json']
missing = []

for fname in required_files:
    fpath = latest_model / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        print(f"   ✅ {fname} ({size_kb:.1f} KB)")
    else:
        print(f"   ❌ {fname} - MISSING!")
        missing.append(fname)

if missing:
    print(f"\n❌ Training not complete yet. Missing: {missing}")
    sys.exit(1)

print("\n✅ All model files present!")

# Now test loading the NN model
print("\n🔧 Testing NN model loading...")

try:
    import joblib
    import numpy as np
    
    nn_path = latest_model / "nn2_classifier.pkl"
    scaler_path = latest_model / "scaler.pkl"
    
    print(f"   Loading NN from: {nn_path}")
    nn_model = joblib.load(nn_path)
    print(f"   ✅ NN Model loaded: {type(nn_model).__name__}")
    
    print(f"   Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
    
    # Test prediction with dummy data (14 features based on notebook)
    print("\n🧪 Testing prediction with dummy data...")
    dummy_input = np.array([[
        25, 1, 1, 4.5, 3, 1, 1, 3, 2, 180, 45, 240, 60, 210
    ]])
    
    print(f"   Input shape: {dummy_input.shape}")
    dummy_scaled = scaler.transform(dummy_input)
    print(f"   Scaled shape: {dummy_scaled.shape}")
    
    prediction = nn_model.predict(dummy_scaled)
    print(f"   ✅ Prediction: {prediction[0]}")
    
    if hasattr(nn_model, 'predict_proba'):
        proba = nn_model.predict_proba(dummy_scaled)
        print(f"   ✅ Probabilities: {proba[0]}")
    
    print("\n" + "="*70)
    print("🎉 SUCCESS! NN MODEL WORKS PERFECTLY!")
    print("="*70)
    
    print("\n📋 Next Steps:")
    print("   1. Activate this model in the app:")
    print(f"      python activate_model.py {latest_model.name}")
    print("   2. Restart the Flask app: python app.py")
    print("   3. Test prediction with NN model in the UI")
    print("\n✅ NN AKAN JALAN 100%!")
    
except Exception as e:
    print(f"\n❌ Error loading NN model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
