"""
Quick test script untuk verify fixes:
1. NN Model loading
2. Retrain trigger logic
"""

from app import app
import smartaddict.runtime as runtime
from smartaddict.models.predict_user_session import PredictUserSession
from smartaddict.services.retrain_service import should_trigger_retrain

def test_model_loading():
    print("\n" + "="*60)
    print("TEST 1: Model Loading & NN Availability")
    print("="*60)
    
    with app.app_context():
        print(f"Active Model Version: {runtime.ACTIVE_MODEL_VERSION}")
        print(f"Available Models: {list(runtime.ml_models.keys())}")
        print(f"Scaler Available: {runtime.scaler is not None}")
        
        # Check if NN is available
        if "Neural Network" in runtime.ml_models:
            print("✅ Neural Network model TERSEDIA!")
            nn_model = runtime.ml_models["Neural Network"]
            print(f"   Type: {type(nn_model)}")
        else:
            print("❌ Neural Network model TIDAK TERSEDIA!")
            print(f"   Available: {list(runtime.ml_models.keys())}")
        
        # Test prediction with NN
        try:
            test_values = [25, 1, 6.0, 3.0, 1.0, 4.0, 7.0, 50, 50, 7.0]
            result = runtime.predict_with_model(test_values, "Neural Network", include_comparison=False)
            print(f"✅ Prediksi dengan NN berhasil!")
            print(f"   Diagnosis: {result['diagnosis']}")
            print(f"   Prediction Raw: {result['prediction_raw']}")
        except Exception as e:
            print(f"❌ Prediksi dengan NN gagal: {e}")


def test_retrain_trigger():
    print("\n" + "="*60)
    print("TEST 2: Auto-Retrain Trigger Logic")
    print("="*60)
    
    with app.app_context():
        current_count = PredictUserSession.query.count()
        print(f"Current predict_user_session count: {current_count}")
        
        # Test cases
        test_cases = [
            (45, 3, False, "45 + 3 = 48 (belum 50)"),
            (48, 2, True, "48 + 2 = 50 (tepat threshold!)"),
            (49, 1, True, "49 + 1 = 50 (tepat threshold!)"),
            (49, 5, True, "49 + 5 = 54 (melewati threshold!)"),
            (50, 10, False, "50 + 10 = 60 (sudah lewat)"),
            (0, 50, True, "0 + 50 = 50 (tepat threshold!)"),
        ]
        
        print("\nTesting trigger logic:")
        all_passed = True
        for before, added, should_trigger, desc in test_cases:
            result = should_trigger_retrain(before, added, threshold=50)
            status = "✅" if result == should_trigger else "❌"
            if result != should_trigger:
                all_passed = False
            print(f"{status} {desc} -> Trigger: {result} (expected: {should_trigger})")
        
        if all_passed:
            print("\n✅ Semua test trigger logic PASSED!")
        else:
            print("\n❌ Ada test yang FAILED!")


def test_reset_logic():
    print("\n" + "="*60)
    print("TEST 3: Reset Table Logic")
    print("="*60)
    
    with app.app_context():
        count = PredictUserSession.query.count()
        print(f"Current rows in predict_user_session: {count}")
        print(f"Logic: Tabel akan di-reset setelah retrain berhasil")
        print(f"       Cek function: _reset_predict_user_sessions() di retrain_service.py")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# TESTING FIXES - NN Model & Auto-Retrain")
    print("#"*60)
    
    test_model_loading()
    test_retrain_trigger()
    test_reset_logic()
    
    print("\n" + "#"*60)
    print("# TESTING SELESAI")
    print("#"*60 + "\n")
