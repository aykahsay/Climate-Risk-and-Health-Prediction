import pandas as pd
import numpy as np

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

# Merge
train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

print("--- DETAILED TARGET RELATIONSHIP BREAKDOWN ---")

# 1. Exact Age breakdown
print("Target rate by exact age (top 20 count):")
age_counts = train_df.groupby('age')['is_climate_sensitive'].agg(['count', 'mean'])
print(age_counts.sort_values(by='count', ascending=False).head(20))

# Is age == 0 or age == 1 or age < 5 practically deterministic?
print("\nTarget rate for age < 1:", train_df[train_df['age'] < 1]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] < 1]))
print("Target rate for age == 0:", train_df[train_df['age'] == 0]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] == 0]))
print("Target rate for age == 1:", train_df[train_df['age'] == 1]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] == 1]))
print("Target rate for age == 2:", train_df[train_df['age'] == 2]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] == 2]))
print("Target rate for age == 3:", train_df[train_df['age'] == 3]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] == 3]))
print("Target rate for age == 4:", train_df[train_df['age'] == 4]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] == 4]))
print("Target rate for age >= 5:", train_df[train_df['age'] >= 5]['is_climate_sensitive'].mean(), "Count:", len(train_df[train_df['age'] >= 5]))

print("\n--- LOCATION / COORDINATE REPETITION & MATCHING ---")
# Check if lat/long rounded or location matches give target hints
train_df['lat_round'] = train_df['latitude'].round(2)
train_df['long_round'] = train_df['longitude'].round(2)
test_df['lat_round'] = test_df['latitude'].round(2)
test_df['long_round'] = test_df['longitude'].round(2)

print("Unique lat/long rounded in train:", train_df[['lat_round', 'long_round']].drop_duplicates().shape[0])
print("Unique lat/long rounded in test:", test_df[['lat_round', 'long_round']].drop_duplicates().shape[0])

# Overlap of rounded lat/long
train_coords = set(zip(train_df['lat_round'], train_df['long_round']))
test_coords = set(zip(test_df['lat_round'], test_df['long_round']))
print("Coordinate overlap count:", len(train_coords.intersection(test_coords)))

# Target rate by coordinate cluster
coord_target = train_df.groupby(['lat_round', 'long_round'])['is_climate_sensitive'].agg(['count', 'mean'])
print("Coordinate clusters target rates (sample):")
print(coord_target.head(15))

print("\n--- CLIMATE EXTREMES & WEATHER LAGS ---")
# Heatwaves / Heavy rain / Consecutive wet days / Drought
train_df['rain_diff_7_30'] = train_df['rain_sum_7d'] - (train_df['rain_sum_30d']/4)
print("Correlation of rain_diff_7_30 with target:", train_df['rain_diff_7_30'].corr(train_df['is_climate_sensitive']))

print("\n--- CHECK FOR DUPLICATE SAMPLES OR SAME-DAY MORTALITY ---")
# How many deaths occur on the same day in the same location?
same_day_loc = train_df.groupby(['deathdate', 'location'])['is_climate_sensitive'].agg(['count', 'mean', 'std'])
print(f"Total same-day location groups in train: {len(same_day_loc)}")
print(f"Groups with count > 1: {len(same_day_loc[same_day_loc['count'] > 1])}")
print("Sample same-day location groups:")
print(same_day_loc[same_day_loc['count'] > 1].head(15))

# Cross match train and test on deathdate and climate features
print("\n--- CHECK MATCH BETWEEN TRAIN AND TEST CLIMATE VALUES ---")
climate_diffs = train_df['avg_temperature'] - train_df['tavg_7d']
print("Diff between avg_temp and tavg_7d stats:")
print(climate_diffs.describe())
