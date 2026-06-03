"""
🔥 FORCE RETRAIN - Langsung execute notebook tanpa app
===================================

Ini akan:
1. Execute notebook langsung (ga lewat Flask app)
2. Generate model BARU dengan numpy 1.26.2 sekarang
3. Save ke model_EMERGENCY/
4. Activate model baru
5. NN AKAN WORKING!
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("🔥 EMERGENCY RETRAIN - DIRECT NOTEBOOK EXECUTION")
print("="*70)

# Setup paths
notebook_path = Path("dataset-notebook/Tubes_FIX.ipynb")
output_dir = Path(f"model/model_EMERGENCY_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
dataset_path = Path("dataset-notebook/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")

output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Output: {output_dir}")
print(f"📓 Notebook: {notebook_path}")
print(f"📊 Dataset: {dataset_path}")

# Set environment variables
os.environ['OUTPUT_MODEL_DIR'] = str(output_dir.absolute())
os.environ['SMARTADDICT_DATASET_PATH'] = str(dataset_path.absolute())

print("\n🚀 Executing notebook cells...")
print("⏱️  This will take 10-15 minutes\n")

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Execute cells
namespace = {'__name__': '__main__', 'display': print, 'output_model_dir': str(output_dir.absolute())}

cell_count = 0
for idx, cell in enumerate(notebook.get('cells', [])):
    if cell.get('cell_type') != 'code':
        continue
    
    source = ''.join(cell.get('source', []))
    
    # Skip Google Colab specific code
    if 'from google.colab import drive' in source or 'drive.mount' in source:
        continue
    if source.strip().startswith('%') or source.strip().startswith('!'):
        continue
    
    # Clean source
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%') or stripped.startswith('!'):
            continue
        lines.append(line)
    
    code = '\n'.join(lines).strip()
    if not code:
        continue
    
    cell_count += 1
    
    # Show progress for important steps
    if 'checkpoint(' in code:
        step_name = code.split('checkpoint(')[1].split(')')[0].strip('\'"')
        print(f"[Cell {cell_count:02d}] {step_name}")
    elif 'GridSearchCV' in code and 'fit' in code:
        if 'nn_classifier' in code:
            print(f"[Cell {cell_count:02d}] 🧠 Training Neural Network (THIS IS IT!)")
        elif 'svm' in code.lower() or 'classifier' in code and 'SVC' in code:
            print(f"[Cell {cell_count:02d}] Training SVM")
        elif 'knn' in code.lower() or 'KNeighbors' in code:
            print(f"[Cell {cell_count:02d}] Training k-NN")
        elif 'dt' in code.lower() or 'DecisionTree' in code:
            print(f"[Cell {cell_count:02d}] Training Decision Tree")
    elif 'joblib.dump' in code and 'nn2_classifier' in code:
        print(f"[Cell {cell_count:02d}] 💾 Saving NN model...")
    
    # Execute
    try:
        exec(compile(code, f'<notebook_cell_{idx}>', 'exec'), namespace)
    except Exception as e:
        if 'checkpoint' not in code:  # Ignore checkpoint errors
            print(f"   ⚠️  Warning in cell {cell_count}: {str(e)[:100]}")

print("\n" + "="*70)
print("✅ NOTEBOOK EXECUTION COMPLETE!")
print("="*70)

# Check if NN model was created
nn_model_file = output_dir / "nn2_classifier.pkl"
if nn_model_file.exists():
    print(f"\n✅ NN Model Created: {nn_model_file}")
    print(f"   Size: {nn_model_file.stat().st_size / 1024:.1f} KB")
    
    # Activate this model
    print("\n🔄 Activating new model...")
    from smartaddict.services.model_service import save_active_version_to_config
    save_active_version_to_config(output_dir.name)
    print(f"✅ Model activated: {output_dir.name}")
    
    print("\n" + "="*70)
    print("🎉 SUCCESS! NN MODEL READY!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("   1. python app.py")
    print("   2. Login & test predict dengan NN")
    print("   3. HARUS BERHASIL!")
    
else:
    print(f"\n❌ NN Model NOT created: {nn_model_file}")
    print("Check errors above")
