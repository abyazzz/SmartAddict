"""Quick check retrain status"""
import json
from pathlib import Path

job_id = "ed03c401-f66d-4e87-b093-7be7b264dff5"
status_file = Path(f"instance/retrain_statuses/{job_id}.json")

if not status_file.exists():
    print("❌ Status file not found!")
    exit(1)

with open(status_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

status = data.get('status', 'unknown')
progress = data.get('progress', 0)
current_step = data.get('current_step', 'None')

print("="*70)
print("RETRAIN STATUS CHECK")
print("="*70)
print(f"Job ID: {job_id}")
print(f"Status: {status}")
print(f"Progress: {progress}%")
print(f"Current Step: {current_step}")
print("="*70)

if status == 'idle':
    print("⏳ Retrain sedang initialize... tunggu 1-2 menit lalu cek lagi")
    print("   Run: python check_status.py")
elif status == 'running':
    print(f"🔄 Retrain sedang berjalan! Progress: {progress}%")
    print("   Refresh popup di web UI atau run script ini lagi")
elif status == 'success':
    print("✅ RETRAIN SELESAI!")
    print("\n📝 Next:")
    print("   1. Start app: python app.py")
    print("   2. Test predict dengan NN model")
elif status == 'failed':
    print("❌ RETRAIN GAGAL!")
    logs = data.get('logs', [])
    if logs:
        print("\nLast log:")
        print(f"   {logs[-1]}")
