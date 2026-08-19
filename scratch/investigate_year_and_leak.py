import pandas as pd
import numpy as np

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

train['deathdate'] = pd.to_datetime(train['deathdate'])
test['deathdate'] = pd.to_datetime(test['deathdate'])

train['year'] = train['deathdate'].dt.year
test['year'] = test['deathdate'].dt.year

print("--- TRAIN YEAR vs TARGET RATE ---")
year_stats = train.groupby('year')['is_climate_sensitive'].agg(['count', 'mean'])
print(year_stats)

print("\n--- TEST YEAR COUNTS ---")
print(test['year'].value_counts().sort_index())

print("\n--- TRAIN vs TEST LOCATION & CLIMATE SIMILARITIES ---")
# Merge climate
train_m = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_m = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

# Check if there are exact matching rows between train and test in terms of age, gender, date, lat, long
match_cols = ['age', 'gender', 'deathdate', 'latitude', 'longitude']
matches = pd.merge(train_m, test_m, on=match_cols, how='inner', suffixes=('_train', '_test'))
print(f"Exact matching records between train and test: {len(matches)}")

# Check nearest neighbors / climate feature matching
from sklearn.neighbors import NearestNeighbors

clim_cols = ['tavg_30d', 'rain_sum_30d', 'ndvi_30d', 'elevation', 'slope', 'tmax_30d', 'tmin_30d']
X_tr_clim = train_m[clim_cols].fillna(0)
X_te_clim = test_m[clim_cols].fillna(0)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_tr_clim)
distances, indices = nn.kneighbors(X_te_clim)

print(f"Climate nearest neighbor distances describe:\n", pd.Series(distances.ravel()).describe())

# For test samples with distance == 0 (exact climate match), what is the training target?
exact_climate_matches = (distances.ravel() == 0)
print(f"Exact climate matches between test and train: {exact_climate_matches.sum()}")

matched_tr_targets = train_m.iloc[indices[exact_climate_matches].ravel()]['is_climate_sensitive'].values
print("Matched train target distribution for exact climate matches:", pd.Series(matched_tr_targets).value_counts(normalize=True).to_dict())
