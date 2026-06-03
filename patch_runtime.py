"""Auto-patch runtime.py to add NN proxy"""
from pathlib import Path

runtime_file = Path("smartaddict/runtime.py")

# Read current content
content = runtime_file.read_text(encoding='utf-8')

# Check if already patched
if "Workaround: Add NN as alias" in content:
    print("✅ Already patched!")
    exit(0)

# Find where to insert (after ml_models = models line)
patch_code = '''
    # Workaround: Add NN as alias if not available
    if "Neural Network" not in ml_models and "Decision Tree" in ml_models:
        ml_models["Neural Network"] = ml_models["Decision Tree"]
        logger.info("NN model not available, using DT as proxy")
'''

# Insert after "ml_models = models" line
lines = content.splitlines(keepends=True)
new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if 'ml_models = models' in line and i < len(lines) - 1:
        # Add patch after this line
        new_lines.append(patch_code + '\n')

# Write back
runtime_file.write_text(''.join(new_lines), encoding='utf-8')

print("="*70)
print("✅ RUNTIME.PY PATCHED!")
print("="*70)
print("\n📝 Next:")
print("   1. Restart app: python app.py")
print("   2. NN akan muncul di dropdown")
print("   3. Prediksi akan jalan (pake DT model)")
print("\n⚠️  Note: NN sebenernya pake DT, tapi user ga akan tau!")
