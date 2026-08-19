import json

nb_path = 'data/climate_health_starter_notebook_.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if 'metric' in source.lower() or 'f1' in source.lower() or 'roc' in source.lower() or 'score' in source.lower() or 'weight' in source.lower() or '0.6' in source:
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        safe_str = source.encode('ascii', 'ignore').decode('ascii')
        print(safe_str)
        print("="*50)
