"""
🎯 REAL RETRAIN - Di environment yang BENAR
=============================================

Ini akan execute notebook pake Python YANG SAMA dengan app.
Jadi numpy version PASTI match!
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("🎯 REAL RETRAIN - CORRECT ENVIRONMENT")
print("="*70)

# Check current environment
print(f"\n📍 Using Python: {sys.executable}")

result = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], 
                       capture_output=True, text=True)
print(f"   NumPy: {result.stdout.strip()}")

result = subprocess.run([sys.executable, "-c", "import sklearn; print(sklearn.__version__)"], 
                       capture_output=True, text=True)
print(f"   Sklearn: {result.stdout.strip()}")

# Setup
notebook_path = Path("dataset-notebook/Tubes_FIX.ipynb")
output_dir = Path(f"model/model_CORRECT_ENV_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n📂 Output will be saved to: {output_dir}")
print(f"📓 Notebook: {notebook_path}")

# Check if nbconvert installed
result = subprocess.run([sys.executable, "-m", "pip", "show", "nbconvert"], 
                       capture_output=True, text=True)

if "Name: nbconvert" not in result.stdout:
    print("\n⚠️  nbconvert not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "nbconvert"], check=True)
    print("✅ nbconvert installed")

# Prepare notebook with output parameter and fix dataset path
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix dataset path and output_model_dir
for cell in notebook['cells']:
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        
        # Update output_model_dir parameter
        if 'output_model_dir' in source and 'parameters' in cell.get('metadata', {}).get('tags', []):
            cell['source'] = [f"output_model_dir = '{output_dir.as_posix()}'\n"]
        
        # Fix dataset path in load dataset cell
        if 'pd.read_csv(file_path)' in source and 'dataset-notebook' in source:
            # Replace the cell with corrected path
            new_source = f"""# Load data dari file clean_dataset.csv
import os
file_path = r'{Path('dataset-notebook/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv').absolute().as_posix()}'
df = pd.read_csv(file_path)

# Convert non-numeric columns to object dtype to ensure compatibility with pandas 3.x
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].astype('object')

print(f"Bentuk dataset: {{df.shape}}")
"""
            cell['source'] = new_source.split('\n')

# Save modified notebook
temp_notebook = Path("dataset-notebook/Tubes_FIX_TEMP.ipynb")
with open(temp_notebook, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("\n🚀 Executing notebook...")
print("⏱️  This will take 10-15 minutes. Please wait...\n")

# Execute using nbconvert
cmd = [
    sys.executable, "-m", "nbconvert",
    "--to", "notebook",
    "--execute",
    str(temp_notebook),
    "--output", str(temp_notebook.stem + "_executed.ipynb"),
    "--ExecutePreprocessor.timeout=-1",
]

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("✅ Notebook executed successfully!")
    
    # Clean up temp file
    temp_notebook.unlink()
    executed_notebook = temp_notebook.parent / (temp_notebook.stem + "_executed.ipynb")
    if executed_notebook.exists():
        executed_notebook.unlink()
    
    # Check if NN model was created
    nn_model = output_dir / "nn2_classifier.pkl"
    if nn_model.exists():
        print(f"\n✅ NN Model created: {nn_model}")
        print(f"   Size: {nn_model.stat().st_size / 1024:.1f} KB")
        
        # Activate this model
        print("\n🔄 Activating model...")
        from smartaddict.services.model_service import save_active_version_to_config
        save_active_version_to_config(output_dir.name)
        
        print("\n" + "="*70)
        print("🎉 SUCCESS!")
        print("="*70)
        print(f"\n✅ Model trained & activated: {output_dir.name}")
        print("\n📝 Next:")
        print("   1. Restart app: python app.py")
        print("   2. Test predict dengan NN")
        print("   3. HARUS BERHASIL karena numpy version SAMA!")
    else:
        print(f"\n❌ NN model not found: {nn_model}")
        print("Check notebook execution errors above")
        
except subprocess.CalledProcessError as e:
    print(f"\n❌ Notebook execution failed!")
    print(f"Error: {e}")
    print(f"\nStdout:\n{e.stdout}")
    print(f"\nStderr:\n{e.stderr}")
    temp_notebook.unlink()
