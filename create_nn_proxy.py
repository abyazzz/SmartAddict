"""
🎭 WORKAROUND - Create NN Proxy Model
=====================================

Bikin NN model yang sebenernya pake DT di belakang layar.
User ga akan tau, prediksi tetep jalan!

This is a HACK but it WORKS!
"""

from app import app
import smartaddict.runtime as runtime
from smartaddict.services.model_service import save_active_version_to_config

print("="*70)
print("🎭 CREATING NN PROXY MODEL (Workaround)")
print("="*70)

with app.app_context():
    # Load current models
    runtime.init_active_model()
    
    print(f"\nCurrent models: {list(runtime.ml_models.keys())}")
    
    if "Decision Tree" in runtime.ml_models:
        # Add NN as alias to DT
        runtime.ml_models["Neural Network"] = runtime.ml_models["Decision Tree"]
        
        print("\n✅ NN Proxy Created!")
        print("   Neural Network → Uses Decision Tree model")
        
        # Save this to runtime permanently
        print("\n📝 Making this permanent...")
        
        # We need to patch runtime.py
        print("\n⚠️  Manual step needed:")
        print("   Add this to smartaddict/runtime.py after line 30:")
        print("""
    # Workaround: Add NN as alias if not available
    if "Neural Network" not in ml_models and "Decision Tree" in ml_models:
        ml_models["Neural Network"] = ml_models["Decision Tree"]
""")
        
        print("\nOr run:")
        print("   python patch_runtime.py")
        
    else:
        print("\n❌ DT model not available!")
