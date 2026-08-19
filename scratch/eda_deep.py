import pandas as pd
import numpy as np

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

# Merge
train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

print("--- DEPORTATION OF DATES & SPLIT ANALYSIS ---")
train_df['deathdate'] = pd.to_datetime(train_df['deathdate'])
test_df['deathdate'] = pd.to_datetime(test_df['deathdate'])

print(f"Train dates: {train_df['deathdate'].min()} to {train_df['deathdate'].max()}")
print(f"Test dates:  {test_df['deathdate'].min()} to {test_df['deathdate'].max()}")

print("\n--- LOCATIONS & GEOGRAPHY ---")
print("Train unique locations:", train_df['location'].nunique())
print("Test unique locations:", test_df['location'].nunique())
overlap_locations = set(train_df['location']).intersection(set(test_df['location']))
print(f"Location overlap: {len(overlap_locations)} / {test_df['location'].nunique()} test locations exist in train")

print("\nTrain zones:", train_df['zone'].value_counts().to_dict())
print("Test zones:", test_df['zone'].value_counts().to_dict())

print("\nTrain gender:", train_df['gender'].value_counts().to_dict())
print("Test gender:", test_df['gender'].value_counts().to_dict())

print("\n--- AGE STATS ---")
print("Train age describe:\n", train_df['age'].describe())
print("Test age describe:\n", test_df['age'].describe())

print("\n--- CORRELATIONS WITH TARGET ---")
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corrs = train_df[numeric_cols].corr()['is_climate_sensitive'].sort_values(ascending=False)
print("Top positive correlations:")
print(corrs.head(10))
print("\nTop negative correlations:")
print(corrs.tail(10))
