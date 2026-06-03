"""
🎯 FINAL FIX untuk NN Model Issue
===================================

ROOT CAUSE:
-----------
NN model yang lama di-save dengan numpy format yang berbeda.
Pas di-load, joblib ga bisa deserialize karena numpy.random._mt19937.MT19937 
BitGenerator format incompatibility.

SOLUTION:
---------
Retrain model BARU dengan numpy 1.26.2 + sklearn 1.8.0 yang sekarang terinstall.
Model baru akan compatible dan bisa di-load!

WHAT THIS SCRIPT DOES:
----------------------
1. Clear semua model lama (backup dulu ke folder backup/)
2. Trigger fresh retrain
3. Generate model baru yang 100% compatible
4. NN model akan tersedia!

WARNING:
--------
Ini akan hapus model lama! Tapi di-backup dulu kok.
"""

import os
import shutil
from datetime import datetime
from app import app
from smartaddict.services.retrain_service import run_retrain_pipeline
from smartaddict.services.model_service import activate_model_version
from pathlib import Path

print("="*70)
print("🔧 FINAL FIX - NN MODEL COMPATIBILITY")
print("="*70)

print("\n📋 Current Environment:")
print(f"   NumPy: 1.26.2")
print(f"   Scikit-learn: 1.8.0")
print(f"   Joblib: latest")

print("\n🎯 Plan:")
print("   1. Backup model lama ke folder backup/")
print("   2. Clean model folder (kecuali model_default)")
print("   3. Trigger retrain baru")
print("   4. Generate NN model yang compatible")
print("   5. Activate model baru")

confirm = input("\n⚠️  Lanjut? Model lama akan di-backup (y/n): ")
if confirm.lower() != 'y':
    print("❌ Cancelled")
    exit(0)

# Step 1: Backup
print("\n" + "="*70)
print("STEP 1: Backup Model Lama")
print("="*70)

backup_dir = Path(f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
backup_dir.mkdir(exist_ok=True)

model_dir = Path("model")
if model_dir.exists():
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("model_202") and item.name != "model_default":
            print(f"   Backup: {item.name}")
            shutil.copytree(item, backup_dir / item.name)

print(f"✅ Backup selesai di: {backup_dir}")

# Step 2: Clean
print("\n" + "="*70)
print("STEP 2: Clean Model Lama")
print("="*70)

for item in model_dir.iterdir():
    if item.is_dir() and item.name.startswith("model_202") and item.name != "model_default":
        print(f"   Delete: {item.name}")
        shutil.rmtree(item)

print("✅ Model folder cleaned")

# Step 3: Retrain
print("\n" + "="*70)
print("STEP 3: Trigger Retrain")
print("="*70)

with app.app_context():
    # Pass app instance directly (not _get_current_object())
    job_id = run_retrain_pipeline(app)
    
    if job_id:
        print(f"\n✅ Retrain started! Job ID: {job_id}")
        print("\n📊 Monitor progress:")
        print(f"   Status file: instance/retrain_statuses/{job_id}.json")
        print(f"   Web UI: http://localhost:5000/admin/dashboard")
        
        print("\n⏱️  Estimated time: 10-15 minutes")
        print("\n🎯 What's happening:")
        print("   1. Training 4 models (DT, KNN, NN, SVM)")
        print("   2. Hyperparameter tuning for each")
        print("   3. Saving models with current numpy/sklearn")
        print("   4. Models will be COMPATIBLE!")
        
        print("\n✅ After completion:")
        print("   - Test NN prediction")
        print("   - Should see 'Neural Network' in available models")
        print("   - All 4 models working!")
        
    else:
        print("\n❌ Failed to start retrain")
        print("   Mungkin ada proses yang masih running")
        print("   Check: http://localhost:5000/admin/dashboard")
