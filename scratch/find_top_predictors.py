import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import lightgbm as lgb

train = pd.read_csv('data/Train.csv')
climate = pd.read_csv('data/climate_features.csv')

df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

# Dates
df['deathdate_dt'] = pd.to_datetime(df['deathdate'])
df['year'] = df['deathdate_dt'].dt.year
df['month'] = df['deathdate_dt'].dt.month
df['day'] = df['deathdate_dt'].dt.day
df['dayofyear'] = df['deathdate_dt'].dt.dayofyear
df['quarter'] = df['deathdate_dt'].dt.quarter

df['log_age'] = np.log1p(df['age'])
df['is_age_0'] = (df['age'] == 0).astype(int)
df['is_age_1'] = (df['age'] == 1).astype(int)
df['is_age_2'] = (df['age'] == 2).astype(int)
df['is_under_5'] = (df['age'] < 5).astype(int)
df['is_elderly'] = (df['age'] >= 65).astype(int)

df['gender_code'] = (df['gender'] == 'Female').astype(int)
df['zone_code'] = (df['zone'] == 'Rural').astype(int)

df['temp_range_day'] = df['max_temperature'] - df['min_temperature']
df['temp_range_30d'] = df['tmax_30d'] - df['tmin_30d']
df['temp_range_anomaly'] = df['temp_range_day'] - df['temp_range_mean_30d']

df['tavg_anomaly_30d'] = df['avg_temperature'] - df['tavg_30d']
df['tavg_anomaly_7d'] = df['avg_temperature'] - df['tavg_7d']
df['tavg_trend_7_30'] = df['tavg_7d'] - df['tavg_30d']

df['rain_daily_avg_30d'] = df['rain_sum_30d'] / 30.0
df['rain_ratio_7_30'] = df['rain_sum_7d'] / (df['rain_sum_30d'] + 1e-5)
df['precip_anomaly_30d'] = df['precipitation'] - df['rain_daily_avg_30d']

df['ndvi_trend_30_90'] = df['ndvi_30d'] - df['ndvi_90d']
df['ndvi_ratio_30_90'] = df['ndvi_30d'] / (df['ndvi_90d'] + 1e-5)

df['under5_x_rain30d'] = df['is_under_5'] * df['rain_sum_30d']
df['under5_x_tavg30d'] = df['is_under_5'] * df['tavg_30d']
df['under5_x_precip'] = df['is_under_5'] * df['precipitation']
df['age_x_tavg30d'] = df['age'] * df['tavg_30d']
df['age_x_rain30d'] = df['age'] * df['rain_sum_30d']

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'location', 'zone', 'gender', 'hot_days_30d']
features = [c for c in df.columns if c not in drop_cols]

X = df[features].fillna(df[features].median())
y = df['is_climate_sensitive'].astype(int)

# Mutual information
mi = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi, index=features).sort_values(ascending=False)

print("--- TOP MUTUAL INFORMATION WITH TARGET ---")
print(mi_series.head(25))

# LightGBM Feature Importance
model = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, verbose=-1)
model.fit(X, y)
lgb_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

print("\n--- TOP LIGHTGBM FEATURE IMPORTANCES ---")
print(lgb_imp.head(25))
