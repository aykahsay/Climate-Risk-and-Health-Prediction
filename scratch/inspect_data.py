import json
import pandas as pd
import numpy as np

# Load data files
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')
sample_sub = pd.read_csv('data/SampleSubmission.csv')

print("--- DATA SUMMARY ---")
print(f"Train shape: {train.shape}, Test shape: {test.shape}, Climate shape: {climate.shape}")
print(f"Target balance: {train['is_climate_sensitive'].value_counts(normalize=True).to_dict()}")

# Read starter notebook to extract metric and starter logic
nb_path = 'data/climate_health_starter_notebook_.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("\n--- STARTER NOTEBOOK SUMMARY ---")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"\n--- Code Cell {i} ---")
        lines = [line for line in source.split('\n') if not line.startswith('%') and not line.startswith('!')]
        clean_code = "\n".join(lines)
        # remove emojis / non-ascii for safe printing
        safe_code = clean_code.encode('ascii', 'ignore').decode('ascii')
        print(safe_code[:400])
