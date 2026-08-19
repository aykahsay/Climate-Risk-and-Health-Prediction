import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

# Merge
train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

# Combine train and test to compute time-series / spatial lag features
df_all = pd.concat([train_df.assign(is_train=1), test_df.assign(is_train=0, is_climate_sensitive=np.nan)], ignore_index=True)
df_all['deathdate_dt'] = pd.to_datetime(df_all['deathdate'])
df_all = df_all.sort_values(by=['deathdate_dt', 'latitude', 'longitude']).reset_index(drop=True)

# 1. Advanced Temporal & Calendar Features
df_all['year'] = df_all['deathdate_dt'].dt.year
df_all['month'] = df_all['deathdate_dt'].dt.month
df_all['day'] = df_all['deathdate_dt'].dt.day
df_all['dayofweek'] = df_all['deathdate_dt'].dt.dayofweek
df_all['dayofyear'] = df_all['deathdate_dt'].dt.dayofyear
df_all['weekofyear'] = df_all['deathdate_dt'].dt.isocalendar().week.astype(int)
df_all['quarter'] = df_all['deathdate_dt'].dt.quarter

# Cyclical calendar features
df_all['sin_month'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
df_all['cos_month'] = np.cos(2 * np.pi * df_all['month'] / 12.0)
df_all['sin_doy'] = np.sin(2 * np.pi * df_all['dayofyear'] / 365.25)
df_all['cos_doy'] = np.cos(2 * np.pi * df_all['dayofyear'] / 365.25)

# 2. Detailed Age Features
df_all['log_age'] = np.log1p(df_all['age'])
df_all['is_age_0'] = (df_all['age'] == 0).astype(int)
df_all['is_age_1'] = (df_all['age'] == 1).astype(int)
df_all['is_age_2'] = (df_all['age'] == 2).astype(int)
df_all['is_age_1_to_3'] = (df_all['age'].isin([1, 2, 3])).astype(int)
df_all['is_under_5'] = (df_all['age'] < 5).astype(int)
df_all['is_elderly'] = (df_all['age'] >= 65).astype(int)
df_all['age_bucket'] = pd.cut(df_all['age'], bins=[-1, 0, 1, 3, 5, 12, 18, 40, 65, 120], labels=False)

# 3. Categorical Encodings
df_all['gender_code'] = (df_all['gender'] == 'Female').astype(int)
df_all['zone_code'] = (df_all['zone'] == 'Rural').astype(int)

# Cleaned location / district
df_all['district'] = df_all['location'].apply(lambda x: x.split(',')[-2].strip() if len(x.split(','))>=2 else x)
df_all['district_cat'] = df_all['district'].astype('category').cat.codes

# 4. Weather & Climate Interactions
df_all['temp_range_day'] = df_all['max_temperature'] - df_all['min_temperature']
df_all['temp_range_30d'] = df_all['tmax_30d'] - df_all['tmin_30d']
df_all['temp_range_anomaly'] = df_all['temp_range_day'] - df_all['temp_range_mean_30d']

df_all['tavg_anomaly_30d'] = df_all['avg_temperature'] - df_all['tavg_30d']
df_all['tavg_anomaly_7d'] = df_all['avg_temperature'] - df_all['tavg_7d']
df_all['tavg_diff_7_30'] = df_all['tavg_7d'] - df_all['tavg_30d']

df_all['rain_daily_avg_30d'] = df_all['rain_sum_30d'] / 30.0
df_all['rain_daily_avg_7d'] = df_all['rain_sum_7d'] / 7.0
df_all['rain_ratio_7_30'] = df_all['rain_sum_7d'] / (df_all['rain_sum_30d'] + 1e-5)
df_all['rain_ratio_30_90'] = df_all['rain_sum_30d'] / (df_all['rain_sum_90d'] + 1e-5)

df_all['precip_anomaly_30d'] = df_all['precipitation'] - df_all['rain_daily_avg_30d']
df_all['precip_anomaly_7d'] = df_all['precipitation'] - df_all['rain_daily_avg_7d']

df_all['ndvi_diff_30_90'] = df_all['ndvi_30d'] - df_all['ndvi_90d']
df_all['ndvi_ratio_30_90'] = df_all['ndvi_30d'] / (df_all['ndvi_90d'] + 1e-5)

# Terrain / Spatial features
df_all['slope_elev_ratio'] = df_all['slope'] / (df_all['elevation'] + 1e-5)
df_all['lat_long_ratio'] = df_all['latitude'] / (df_all['longitude'] + 1e-5)
df_all['lat_long_sum'] = df_all['latitude'] + df_all['longitude']

# 5. Grouped Climate Summary Aggregations
for group_col in ['zone', 'district', 'month', 'age_bucket']:
    for feat in ['avg_temperature', 'precipitation', 'rain_sum_30d', 'tavg_30d', 'ndvi_30d']:
        grp_mean = df_all.groupby(group_col)[feat].transform('mean')
        grp_std = df_all.groupby(group_col)[feat].transform('std')
        df_all[f'{feat}_mean_by_{group_col}'] = grp_mean
        df_all[f'{feat}_diff_from_{group_col}_mean'] = df_all[feat] - grp_mean

# 6. Age x Climate Interactions
df_all['age_x_precip'] = df_all['age'] * df_all['precipitation']
df_all['age_x_tavg30d'] = df_all['age'] * df_all['tavg_30d']
df_all['age_x_rain30d'] = df_all['age'] * df_all['rain_sum_30d']
df_all['under5_x_rain30d'] = df_all['is_under_5'] * df_all['rain_sum_30d']
df_all['under5_x_tavg30d'] = df_all['is_under_5'] * df_all['tavg_30d']

# Split back into train and test
train_clean = df_all[df_all['is_train'] == 1].sort_values(by='ID').reset_index(drop=True)
test_clean = df_all[df_all['is_train'] == 0].sort_values(by='ID').reset_index(drop=True)

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'is_train', 'location', 'district', 'zone', 'gender', 'hot_days_30d']
features = [c for c in train_clean.columns if c not in drop_cols]

X = train_clean[features]
y = train_clean['is_climate_sensitive'].astype(int)
X_test = test_clean[features]

print(f"Total features created: {len(features)}")

# Evaluate 5-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_multi_metric(y_true, y_pred_proba):
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score

# CatBoost Tuning
oof_cat = np.zeros(len(X))
for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model = CatBoostClassifier(iterations=1500, learning_rate=0.03, depth=6, random_seed=42, early_stopping_rounds=100, verbose=0)
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[val_idx] = model.predict_proba(X_va)[:, 1]

f1_c, auc_c, score_c = evaluate_multi_metric(y, oof_cat)
print(f"[Enhanced CatBoost] OOF F1: {f1_c:.5f} | ROC-AUC: {auc_c:.5f} | Final Score: {score_c:.5f}")

# LightGBM Tuning
oof_lgb = np.zeros(len(X))
for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.02, num_leaves=31, max_depth=6, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[val_idx] = model.predict_proba(X_va)[:, 1]

f1_l, auc_l, score_l = evaluate_multi_metric(y, oof_lgb)
print(f"[Enhanced LightGBM] OOF F1: {f1_l:.5f} | ROC-AUC: {auc_l:.5f} | Final Score: {score_l:.5f}")

# XGBoost Tuning
oof_xgb = np.zeros(len(X))
for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    model = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.7, random_state=42, early_stopping_rounds=100)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model.predict_proba(X_va)[:, 1]

f1_x, auc_x, score_x = evaluate_multi_metric(y, oof_xgb)
print(f"[Enhanced XGBoost]  OOF F1: {f1_x:.5f} | ROC-AUC: {auc_x:.5f} | Final Score: {score_x:.5f}")

# Combined Ensemble
oof_ensemble = 0.5 * oof_cat + 0.25 * oof_lgb + 0.25 * oof_xgb
f1_e, auc_e, score_e = evaluate_multi_metric(y, oof_ensemble)
print(f"[Enhanced Ensemble] OOF F1: {f1_e:.5f} | ROC-AUC: {auc_e:.5f} | Final Score: {score_e:.5f}")
