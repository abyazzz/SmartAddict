"""
Diagnose kenapa Jupyter pake environment berbeda
"""
import subprocess
import sys

print("="*70)
print("🔍 DIAGNOSE JUPYTER ENVIRONMENT")
print("="*70)

print("\n1️⃣ Current Python (yang app pake):")
print(f"   Path: {sys.executable}")

result = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], 
                       capture_output=True, text=True)
print(f"   NumPy: {result.stdout.strip()}")

result = subprocess.run([sys.executable, "-c", "import sklearn; print(sklearn.__version__)"], 
                       capture_output=True, text=True)
print(f"   Sklearn: {result.stdout.strip()}")

print("\n2️⃣ Jupyter Python (yang notebook pake):")
result = subprocess.run(["jupyter", "--version"], capture_output=True, text=True)
if result.returncode == 0:
    print(f"   Jupyter installed: Yes")
    print(f"   {result.stdout.strip()}")
    
    # Check which python jupyter uses
    result = subprocess.run(["jupyter", "kernelspec", "list"], capture_output=True, text=True)
    print(f"\n   Available kernels:")
    print(f"   {result.stdout}")
else:
    print(f"   Jupyter: Not installed atau ga di PATH")

print("\n3️⃣ Solution:")
print("   Jupyter harus pake Python yang sama dengan app!")
print("\n   Option A - Install jupyter di environment ini:")
print("   pip install jupyter")
print("\n   Option B - Run notebook via nbconvert:")
print("   pip install nbconvert")
print("   jupyter nbconvert --execute dataset-notebook/Tubes_FIX.ipynb")
print("\n   Option C - Run notebook via Python directly:")
print("   python run_notebook_directly.py")
