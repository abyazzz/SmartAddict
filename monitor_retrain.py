"""Monitor retrain progress"""
import json
import time
from pathlib import Path

job_id = "ed03c401-f66d-4e87-b093-7be7b264dff5"
status_file = Path(f"instance/retrain_statuses/{job_id}.json")

print("="*70)
print(f"MONITORING RETRAIN JOB: {job_id}")
print("="*70)
print("\nPress Ctrl+C to stop monitoring\n")

last_progress = 0
last_step = None

try:
    while True:
        if not status_file.exists():
            print("⏳ Waiting for job to start...")
            time.sleep(5)
            continue
        
        with open(status_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        status = data.get('status', 'unknown')
        progress = data.get('progress', 0)
        current_step = data.get('current_step')
        
        # Update display if changed
        if progress != last_progress or current_step != last_step:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status:8} | Progress: {progress:5.1f}% | Step: {current_step or 'Idle'}")
            last_progress = progress
            last_step = current_step
        
        # Check if done
        if status in ['success', 'failed']:
            print("\n" + "="*70)
            if status == 'success':
                print("✅ RETRAIN SELESAI!")
                print("\nModel baru berhasil di-generate!")
                print("\n📝 Next Steps:")
                print("   1. Start app: python app.py")
                print("   2. Login ke http://localhost:5000")
                print("   3. Test predict dengan model 'Neural Network'")
                print("   4. NN harus available sekarang!")
            else:
                print("❌ RETRAIN GAGAL!")
                print("\nCheck logs di status file untuk detail error.")
            print("="*70)
            break
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\n\n⚠️  Monitoring stopped (retrain still running in background)")
    print(f"   Check status: {status_file}")
