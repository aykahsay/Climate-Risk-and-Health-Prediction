import pandas as pd
import numpy as np

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

print("--- HOT DAYS ---")
print("Train hot_days_30d unique:", train_df['hot_days_30d'].value_counts())
print("Test hot_days_30d unique:", test_df['hot_days_30d'].value_counts())

print("\n--- AGE vs IS_CLIMATE_SENSITIVE ---")
train_df['age_group'] = pd.cut(train_df['age'], bins=[-1, 1, 5, 12, 18, 50, 70, 120], labels=['<1', '1-5', '5-12', '12-18', '18-50', '50-70', '70+'])
print(train_df.groupby('age_group')['is_climate_sensitive'].agg(['count', 'mean']))

print("\n--- LOCATION STRINGS & DISTRICTS ---")
print("Sample train locations:", train_df['location'].head(10).tolist())
print("Sample test locations:", test_df['location'].head(10).tolist())

train_df['district'] = train_df['location'].apply(lambda x: x.split(',')[-2].strip() if len(x.split(','))>=2 else x)
test_df['district'] = test_df['location'].apply(lambda x: x.split(',')[-2].strip() if len(x.split(','))>=2 else x)
print("\nTrain districts:", train_df['district'].value_counts().to_dict())
print("Test districts:", test_df['district'].value_counts().to_dict())

print("\n--- LAT/LONG BOUNDS ---")
print(f"Train Lat: {train_df['latitude'].min():.4f} to {train_df['latitude'].max():.4f}, Long: {train_df['longitude'].min():.4f} to {train_df['longitude'].max():.4f}")
print(f"Test Lat:  {test_df['latitude'].min():.4f} to {test_df['latitude'].max():.4f}, Long: {test_df['longitude'].min():.4f} to {test_df['longitude'].max():.4f}")

print("\n--- SEASONALITY / MONTHS ---")
train_df['month'] = pd.to_datetime(train_df['deathdate']).dt.month
test_df['month'] = pd.to_datetime(test_df['deathdate']).dt.month
print(train_df.groupby('month')['is_climate_sensitive'].agg(['count', 'mean']))
