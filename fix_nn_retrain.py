"""
Helper script untuk trigger manual retrain dan fix NN model compatibility
"""

from app import app
from smartaddict.services.retrain_service import run_retrain_pipeline
import time

print("\n" + "="*70)
print("RETRAIN MODEL BARU - FIX NN COMPATIBILITY")
print("="*70)

print("\nModel NN yang ada di-save dengan numpy version lama dan tidak")
print("compatible dengan numpy 1.26.2 yang sekarang terinstall.")
print("\nSolusi: Retrain model baru yang akan compatible dengan numpy sekarang.")

input("\nPress ENTER untuk mulai retrain (atau Ctrl+C untuk cancel)...")

with app.app_context():
    job_id = run_retrain_pipeline(app._get_current_object())
    
    if job_id:
        print(f"\n✅ Retrain dimulai! Job ID: {job_id}")
        print("\nProses retraining berjalan di background.")
        print("Untuk monitoring progress, buka browser:")
        print("  http://localhost:5000/admin/dashboard")
        print("\nAtau check status file di:")
        print(f"  instance/retrain_statuses/{job_id}.json")
        print("\nProses ini akan:")
        print("  1. Train 4 model (DT, KNN, NN, SVM) dengan hyperparameter tuning")
        print("  2. Save model baru ke folder model_TIMESTAMP/")
        print("  3. Aktifkan model baru sebagai active model")
        print("  4. Reset tabel predict_user_session ke 0")
        print("\nWaktu estimasi: 5-15 menit tergantung performa CPU")
    else:
        print("\n❌ Gagal start retrain. Mungkin sudah ada proses yang berjalan.")
        print("Check status di /admin/dashboard")
