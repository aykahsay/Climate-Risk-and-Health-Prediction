import glob
import pandas as pd

print("Inspecting all CSV submission files in the directory:")
for f in sorted(glob.glob('*.csv') + glob.glob('scratch/*.csv')):
    try:
        df = pd.read_csv(f)
        if 'TargetF1' in df.columns:
            ones = (df['TargetF1'] == 1).sum()
            print(f"{f:35s}: shape={df.shape}, TargetF1 ones={ones} ({ones/len(df):.1%}), mean_prob={df['TargetRAUC'].mean():.4f}")
    except Exception as e:
        pass
