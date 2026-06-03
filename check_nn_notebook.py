"""Check if NN model is properly trained in the notebook"""
import json

with open('dataset-notebook/Tubes_FIX.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("="*70)
print("CHECKING NN CLASSIFIER TRAINING IN NOTEBOOK")
print("="*70)

# Find cells with nn_classifier
nn_cells = []
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'nn_classifier' in source:
            nn_cells.append((idx, source))

print(f"\nFound {len(nn_cells)} cells mentioning 'nn_classifier'\n")

# Check for key operations
has_gridsearch = False
has_fit = False
has_save = False

for idx, source in nn_cells:
    print(f"\n{'='*60}")
    print(f"Cell #{idx}:")
    print("="*60)
    print(source[:800])
    
    if 'GridSearchCV' in source:
        has_gridsearch = True
        print("\n✅ Found GridSearchCV definition")
    if 'nn_classifier.fit' in source or 'nn_classifier.fit(' in source:
        has_fit = True
        print("\n✅ Found nn_classifier.fit() call")
    if 'nn_artifact' in source or 'nn2_classifier.pkl' in source:
        has_save = True
        print("\n✅ Found NN model save")

print("\n" + "="*70)
print("SUMMARY:")
print("="*70)
print(f"GridSearchCV defined: {'✅ YES' if has_gridsearch else '❌ NO'}")
print(f"nn_classifier.fit() called: {'✅ YES' if has_fit else '❌ NO - INI MASALAHNYA!'}")
print(f"NN model saved: {'✅ YES' if has_save else '❌ NO'}")

if not has_fit:
    print("\n" + "="*70)
    print("⚠️  CRITICAL BUG FOUND!")
    print("="*70)
    print("nn_classifier di-define tapi TIDAK DI-TRAIN (.fit() missing)!")
    print("Makanya nn2_classifier.pkl ga pernah ke-generate dengan benar.")
    print("\nSolusi: Tambah cell training untuk nn_classifier!")
