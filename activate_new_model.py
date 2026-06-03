"""Activate model baru hasil retrain manual"""
from app import app
from smartaddict.services.model_service import activate_model_version
import smartaddict.runtime as runtime

model_name = "model_20260603_193256"

print("="*70)
print(f"ACTIVATING MODEL: {model_name}")
print("="*70)

with app.app_context():
    # Activate
    success = activate_model_version(model_name)
    
    if success:
        print(f"\n✅ Model activated: {model_name}")
        
        # Verify
        print("\n🔍 Verifying loaded models...")
        print(f"Active Version: {runtime.ACTIVE_MODEL_VERSION}")
        print(f"\nAvailable Models:")
        for m in runtime.ml_models.keys():
            print(f"  ✅ {m}")
        
        if "Neural Network" in runtime.ml_models:
            print("\n" + "="*70)
            print("🎉 SUCCESS! NEURAL NETWORK AVAILABLE!")
            print("="*70)
            print("\n📝 Next:")
            print("   1. Restart app: python app.py")
            print("   2. Refresh browser")
            print("   3. NN harus muncul di dropdown!")
        else:
            print("\n❌ NN still not in ml_models")
            print("Available:", list(runtime.ml_models.keys()))
    else:
        print(f"\n❌ Failed to activate {model_name}")
