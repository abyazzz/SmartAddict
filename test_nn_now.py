"""Test NN model after numpy downgrade"""
from app import app
import smartaddict.runtime as runtime

print("="*70)
print("TESTING NN MODEL - POST NUMPY DOWNGRADE")
print("="*70)

with app.app_context():
    # Force reload models
    runtime.init_active_model()
    
    print(f"\nNumPy Version: ", end="")
    import numpy as np
    print(np.__version__)
    
    print(f"Active Model: {runtime.ACTIVE_MODEL_VERSION}")
    print(f"\nAvailable Models:")
    for model_name in runtime.ml_models.keys():
        print(f"  ✅ {model_name}")
    
    # Check if NN is available
    if "Neural Network" in runtime.ml_models:
        print("\n" + "="*70)
        print("✅ SUCCESS! NEURAL NETWORK MODEL TERSEDIA!")
        print("="*70)
        
        # Test prediction
        print("\n🧪 Testing NN Prediction...")
        test_values = [25, 1, 6.0, 3.0, 1.0, 4.0, 7.0, 50, 50, 7.0]
        try:
            result = runtime.predict_with_model(test_values, "Neural Network", include_comparison=False)
            print(f"✅ Prediction Success!")
            print(f"   Input: {test_values}")
            print(f"   Result: {result['diagnosis']}")
            print(f"   Raw: {result['prediction_raw']}")
            
            print("\n" + "="*70)
            print("🎉 NN MODEL 100% WORKING!")
            print("="*70)
            print("\n📝 Next Steps:")
            print("   1. Start app: python app.py")
            print("   2. Test predict di UI dengan model 'Neural Network'")
            print("   3. HARUS BERHASIL!")
            
        except Exception as e:
            print(f"❌ Prediction Failed: {e}")
    else:
        print("\n" + "="*70)
        print("❌ NN MODEL MASIH BELUM AVAILABLE")
        print("="*70)
        print("Available models:", list(runtime.ml_models.keys()))
