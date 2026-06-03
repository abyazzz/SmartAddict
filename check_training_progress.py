"""
Quick script to check training progress
"""
import os
from pathlib import Path
import time

model_dir = Path("model/model_FINAL_20260604_030811")

print("🔍 Checking training progress...\n")

if not model_dir.exists():
    print("❌ Model directory doesn't exist yet")
else:
    files = list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json"))
    
    if not files:
        print("⏳ Training in progress... No model files created yet")
    else:
        print(f"✅ Found {len(files)} file(s):")
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f} KB)")
        
        # Check if NN model exists
        nn_model = model_dir / "nn2_classifier.pkl"
        if nn_model.exists():
            print("\n🎉 NN MODEL CREATED! Training likely complete!")
        else:
            print("\n⏳ Still waiting for nn2_classifier.pkl...")

print("\n💡 Tip: Training takes 10-15 minutes. Be patient!")
