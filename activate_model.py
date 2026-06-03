"""
Activate a specific model version
Usage: python activate_model.py model_FINAL_20260604_030811
"""

import sys
import json
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python activate_model.py <model_folder_name>")
    print("\nExample: python activate_model.py model_FINAL_20260604_030811")
    sys.exit(1)

model_name = sys.argv[1]
model_path = Path("model") / model_name

if not model_path.exists():
    print(f"❌ Model directory not found: {model_path}")
    sys.exit(1)

print(f"✅ Found model directory: {model_path}")

# Update model_config.json
config_path = Path("instance/model_config.json")
config_path.parent.mkdir(parents=True, exist_ok=True)

config = {
    "active_version": model_name
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Activated model: {model_name}")
print(f"   Config saved to: {config_path}")
print("\n🔄 Restart the app to use the new model:")
print("   python app.py")
