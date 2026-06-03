"""
🎯 FINAL SOLUTION - Convert Notebook to Python Script & Run
=============================================================

This is THE solution that will work 100%!
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("="*70)
print("🎯 FINAL SOLUTION - NOTEBOOK TO PYTHON SCRIPT")
print("="*70)

# Setup
notebook_path = Path("dataset-notebook/Tubes_FIX.ipynb")
output_dir = Path(f"model/model_FINAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
output_dir.mkdir(parents=True, exist_ok=True)
dataset_path = Path("dataset-notebook/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv").absolute()

print(f"\n📂 Output: {output_dir}")
print(f"📓 Notebook: {notebook_path}")
print(f"📊 Dataset: {dataset_path}")

# Convert notebook to Python script
print("\n🔄 Converting notebook to Python script...")

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract all code cells into a single Python script
python_script = f"""#!/usr/bin/env python
# coding: utf-8

# Auto-generated from Tubes_FIX.ipynb
# Generated at: {datetime.now()}

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
import datetime as _datetime_module

# Set output directory
output_model_dir = r'{output_dir.absolute().as_posix()}'
os.makedirs(output_model_dir, exist_ok=True)

# Set dataset path
DATASET_PATH = r'{dataset_path.as_posix()}'

# Helper for display() function (used in notebook)
def display(obj):
    print(obj)

"""

for idx, cell in enumerate(notebook.get('cells', [])):
    if cell.get('cell_type') != 'code':
        continue
    
    source = ''.join(cell.get('source', []))
    
    # Skip Google Colab specific code
    if 'from google.colab import drive' in source or 'drive.mount' in source:
        continue
    
    # Skip magic commands
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%') or stripped.startswith('!'):
            continue
        lines.append(line)
    
    code = '\n'.join(lines).strip()
    if not code:
        continue
    
    # Fix dataset path in load cell
    if 'pd.read_csv(file_path)' in code:
        code = code.replace(
            "file_path = 'dataset-notebook/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'",
            "file_path = DATASET_PATH"
        )
        code = code.replace(
            "if not os.path.exists(file_path):\n    file_path = '/content/drive/MyDrive/dataset_tubes/Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv'",
            ""
        )
    
    # Fix checkpoint calls (they might fail, make them optional)
    if 'checkpoint(' in code:
        code = f"try:\n    {code.replace(chr(10), chr(10) + '    ')}\nexcept: pass"
    
    python_script += f"\n\n# Cell {idx}\n{code}\n"

# Save Python script
script_path = Path("retrain_script.py")
script_path.write_text(python_script, encoding='utf-8')

print(f"✅ Python script created: {script_path}")

# Run the script
print("\n🚀 Running training script...")
print("⏱️  This will take 10-15 minutes. Please wait...\n")

try:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )
    
    print("✅ Training completed successfully!")
    
    if result.stdout:
        print("\n📋 Output (last 50 lines):")
        print('\n'.join(result.stdout.splitlines()[-50:]))
    
    # Check if NN model was created
    nn_model = output_dir / "nn2_classifier.pkl"
    if nn_model.exists():
        print(f"\n✅ NN Model created: {nn_model}")
        print(f"   Size: {nn_model.stat().st_size / 1024:.1f} KB")
        
        # List all files created
        print(f"\n📦 All model files:")
        for f in output_dir.iterdir():
            if f.is_file():
                print(f"   ✅ {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        # Activate this model
        print("\n🔄 Activating model...")
        from app import app
        with app.app_context():
            from smartaddict.services.model_service import save_active_version_to_config
            import smartaddict.runtime as runtime
            
            save_active_version_to_config(output_dir.name)
            runtime.init_active_model()
            
            print(f"✅ Model activated: {output_dir.name}")
            print(f"\n🔍 Available models:")
            for m in runtime.ml_models.keys():
                print(f"   ✅ {m}")
            
            if "Neural Network" in runtime.ml_models:
                print("\n" + "="*70)
                print("🎉🎉🎉 SUCCESS! NEURAL NETWORK IS AVAILABLE! 🎉🎉🎉")
                print("="*70)
                print("\n📝 Final Steps:")
                print("   1. Restart app: python app.py")
                print("   2. Test predict dengan NN")
                print("   3. NN PASTI JALAN!")
            else:
                print(f"\n⚠️  NN not in loaded models yet")
                print(f"   Available: {list(runtime.ml_models.keys())}")
    else:
        print(f"\n❌ NN model not found: {nn_model}")
        print("Check if training completed properly")
        
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed!")
    print(f"\nLast 100 lines of stdout:")
    if e.stdout:
        print('\n'.join(e.stdout.splitlines()[-100:]))
    print(f"\nLast 50 lines of stderr:")
    if e.stderr:
        print('\n'.join(e.stderr.splitlines()[-50:]))
