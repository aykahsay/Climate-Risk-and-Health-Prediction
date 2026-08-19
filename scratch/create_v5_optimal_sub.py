import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

df_all = pd.concat([train_df.assign(is_train=1), test_df.assign(is_train=0, is_climate_sensitive=np.nan)], ignore_index=True)
df_all['deathdate_dt'] = pd.to_datetime(df_all['deathdate'])

# 1. Calendar & Time Features
df_all['year'] = df_all['deathdate_dt'].dt.year
df_all['month'] = df_all['deathdate_dt'].dt.month
df_all['day'] = df_all['deathdate_dt'].dt.day
df_all['dayofweek'] = df_all['deathdate_dt'].dt.dayofweek
df_all['dayofyear'] = df_all['deathdate_dt'].dt.dayofyear
df_all['weekofyear'] = df_all['deathdate_dt'].dt.isocalendar().week.astype(int)
df_all['quarter'] = df_all['deathdate_dt'].dt.quarter

df_all['sin_month'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
df_all['cos_month'] = np.cos(2 * np.pi * df_all['month'] / 12.0)
df_all['sin_doy'] = np.sin(2 * np.pi * df_all['dayofyear'] / 365.25)
df_all['cos_doy'] = np.cos(2 * np.pi * df_all['dayofyear'] / 365.25)

# Uganda rainy seasons: Mar-May (season 1) & Sep-Nov (season 2)
df_all['is_wet_season_1'] = df_all['month'].isin([3, 4, 5]).astype(int)
df_all['is_wet_season_2'] = df_all['month'].isin([9, 10, 11]).astype(int)
df_all['is_dry_season'] = df_all['month'].isin([1, 2, 6, 7, 8, 12]).astype(int)

# 2. Demographic Vulnerability
df_all['log_age'] = np.log1p(df_all['age'])
df_all['sqrt_age'] = np.sqrt(df_all['age'])
df_all['is_age_0'] = (df_all['age'] == 0).astype(int)
df_all['is_age_1'] = (df_all['age'] == 1).astype(int)
df_all['is_age_2'] = (df_all['age'] == 2).astype(int)
df_all['is_infant'] = (df_all['age'] < 1.0).astype(int)
df_all['is_toddler'] = ((df_all['age'] >= 1.0) & (df_all['age'] < 5.0)).astype(int)
df_all['is_under_5'] = (df_all['age'] < 5.0).astype(int)
df_all['is_school_age'] = ((df_all['age'] >= 5.0) & (df_all['age'] < 18.0)).astype(int)
df_all['is_adult'] = ((df_all['age'] >= 18.0) & (df_all['age'] < 60.0)).astype(int)
df_all['is_senior'] = (df_all['age'] >= 60.0).astype(int)
df_all['age_group'] = pd.cut(df_all['age'], bins=[-1, 0, 1, 3, 5, 12, 18, 40, 65, 120], labels=False)

df_all['gender_female'] = (df_all['gender'] == 'Female').astype(int)
df_all['zone_rural'] = (df_all['zone'] == 'Rural').astype(int)

# 3. Spatial Coordinates & Terrain
df_all['lat_poly2'] = df_all['latitude'] ** 2
df_all['long_poly2'] = df_all['longitude'] ** 2
df_all['lat_x_long'] = df_all['latitude'] * df_all['longitude']
df_all['slope_elev_ratio'] = df_all['slope'] / (df_all['elevation'] + 1e-5)
df_all['slope_elev_prod'] = df_all['slope'] * df_all['elevation']

# Malaria risk elevation threshold in Uganda (highlands > 1300m have lower malaria risk)
df_all['is_highland'] = (df_all['elevation'] > 1300).astype(int)
df_all['is_lowland'] = (df_all['elevation'] < 1100).astype(int)

# 4. Weather Lags & Acceleration Features
df_all['temp_range_day'] = df_all['max_temperature'] - df_all['min_temperature']
df_all['temp_range_30d'] = df_all['tmax_30d'] - df_all['tmin_30d']
df_all['temp_range_anomaly'] = df_all['temp_range_day'] - df_all['temp_range_mean_30d']

df_all['tmax_diff'] = df_all['max_temperature'] - df_all['tmax_30d']
df_all['tmin_diff'] = df_all['min_temperature'] - df_all['tmin_30d']
df_all['tavg_anomaly_30d'] = df_all['avg_temperature'] - df_all['tavg_30d']
df_all['tavg_anomaly_7d'] = df_all['avg_temperature'] - df_all['tavg_7d']
df_all['tavg_trend_7_30'] = df_all['tavg_7d'] - df_all['tavg_30d']
df_all['tavg_trend_30_90'] = df_all['tavg_30d'] - df_all['tavg_90d']

# Rainfall acceleration & ratios
df_all['rain_rate_7d'] = df_all['rain_sum_7d'] / 7.0
df_all['rain_rate_30d'] = df_all['rain_sum_30d'] / 30.0
df_all['rain_rate_90d'] = df_all['rain_sum_90d'] / 90.0
df_all['rain_accel_7_30'] = df_all['rain_rate_7d'] - df_all['rain_rate_30d']
df_all['rain_accel_30_90'] = df_all['rain_rate_30d'] - df_all['rain_rate_90d']

df_all['rain_ratio_7_30'] = df_all['rain_sum_7d'] / (df_all['rain_sum_30d'] + 1e-5)
df_all['rain_ratio_30_90'] = df_all['rain_sum_30d'] / (df_all['rain_sum_90d'] + 1e-5)
df_all['rain_intensity_30d'] = df_all['max_daily_rain_30d'] / (df_all['rain_sum_30d'] + 1e-5)
df_all['rain_day_prop_30d'] = df_all['rain_days_30d'] / 30.0

df_all['precip_anomaly_30d'] = df_all['precipitation'] - df_all['rain_rate_30d']
df_all['precip_anomaly_7d'] = df_all['precipitation'] - df_all['rain_rate_7d']

# Vegetation & Drought Indices
df_all['ndvi_diff_30_90'] = df_all['ndvi_30d'] - df_all['ndvi_90d']
df_all['ndvi_ratio_30_90'] = df_all['ndvi_30d'] / (df_all['ndvi_90d'] + 1e-5)

# 5. Non-linear Vulnerability Interactions
df_all['under5_x_rain30d'] = df_all['is_under_5'] * df_all['rain_sum_30d']
df_all['under5_x_tavg30d'] = df_all['is_under_5'] * df_all['tavg_30d']
df_all['under5_x_precip'] = df_all['is_under_5'] * df_all['precipitation']
df_all['under5_x_ndvi30d'] = df_all['is_under_5'] * df_all['ndvi_30d']
df_all['under5_x_rain_accel'] = df_all['is_under_5'] * df_all['rain_accel_7_30']

df_all['senior_x_tmax30d'] = df_all['is_senior'] * df_all['tmax_30d']
df_all['senior_x_tavg_anomaly'] = df_all['is_senior'] * df_all['tavg_anomaly_30d']

df_all['age_x_tavg30d'] = df_all['age'] * df_all['tavg_30d']
df_all['age_x_rain30d'] = df_all['age'] * df_all['rain_sum_30d']
df_all['age_x_ndvi30d'] = df_all['age'] * df_all['ndvi_30d']

train_clean = df_all[df_all['is_train'] == 1].sort_values(by='ID').reset_index(drop=True)
test_clean = df_all[df_all['is_train'] == 0].sort_values(by='ID').reset_index(drop=True)

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'is_train', 'location', 'zone', 'gender', 'hot_days_30d']
features = [c for c in train_clean.columns if c not in drop_cols]

X = train_clean[features].copy()
y = train_clean['is_climate_sensitive'].astype(int).values
X_test = test_clean[features].copy()

# Add Target Encodings
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()

for col in ['age_group', 'month', 'year', 'quarter']:
    X[f'{col}_te'] = np.nan
    X_test[f'{col}_te'] = np.nan
    for train_idx, val_idx in skf.split(X, y):
        tr_y = y[train_idx]
        tr_col = X.iloc[train_idx][col]
        te_map = pd.Series(tr_y).groupby(tr_col).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
        X.iloc[val_idx, X.columns.get_loc(f'{col}_te')] = X.iloc[val_idx][col].map(te_map).fillna(global_mean)
    full_map = pd.Series(y).groupby(X[col]).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
    X_test[f'{col}_te'] = X_test[col].map(full_map).fillna(global_mean)

features = list(X.columns)

# 10-Fold Stratified CV across 4 Diverse Models for Maximum Generalization
skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

oof_rf = np.zeros(len(X))
test_rf = np.zeros(len(X_test))

X_imp = X.fillna(X.median())
X_test_imp = X_test.fillna(X.median())

print("--- TRAINING 10-FOLD GBDT + RF ENSEMBLE ---")

for fold, (train_idx, val_idx) in enumerate(skf10.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # CatBoost
    cb = CatBoostClassifier(iterations=1200, learning_rate=0.03, depth=6, random_seed=42+fold, early_stopping_rounds=60, verbose=0)
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = cb.predict_proba(X_va)[:, 1]
    test_cat += cb.predict_proba(X_test)[:, 1] / 10.0
    
    # LightGBM
    lgbm = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.02, max_depth=6, num_leaves=31, random_state=42+fold, verbose=-1)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_lgb[val_idx] = lgbm.predict_proba(X_va)[:, 1]
    test_lgb += lgbm.predict_proba(X_test)[:, 1] / 10.0
    
    # XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, early_stopping_rounds=60)
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_m.predict_proba(X_va)[:, 1]
    test_xgb += xgb_m.predict_proba(X_test)[:, 1] / 10.0
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42+fold, n_jobs=-1)
    rf.fit(X_imp.iloc[train_idx], y_tr)
    oof_rf[val_idx] = rf.predict_proba(X_imp.iloc[val_idx])[:, 1]
    test_rf += rf.predict_proba(X_test_imp)[:, 1] / 10.0

def eval_score(y_true, y_pred_proba):
    y_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score

print(f"\n10-Fold CatBoost OOF Score: {eval_score(y, oof_cat)[2]:.5f}")
print(f"10-Fold LightGBM OOF Score: {eval_score(y, oof_lgb)[2]:.5f}")
print(f"10-Fold XGBoost  OOF Score: {eval_score(y, oof_xgb)[2]:.5f}")
print(f"10-Fold RandFor   OOF Score: {eval_score(y, oof_rf)[2]:.5f}")

# Weighted Average with Optimal Prior Balance
oof_blend = 0.35 * oof_cat + 0.35 * oof_lgb + 0.20 * oof_xgb + 0.10 * oof_rf
test_blend = 0.35 * test_cat + 0.35 * test_lgb + 0.20 * test_xgb + 0.10 * test_rf

f1_b, auc_b, score_b = eval_score(y, oof_blend)
print(f"\n--- 10-FOLD ENSEMBLE SCORE ---")
print(f"OOF F1 @ 0.5: {f1_b:.5f} | ROC-AUC: {auc_b:.5f} | Multi-Metric Score: {score_b:.5f}")

# Clean submission export
sub = pd.DataFrame({
    'ID': test_clean['ID'],
    'TargetF1': (test_blend >= 0.5).astype(int),
    'TargetRAUC': test_blend
})

sub.to_csv('submission_v5_optimal.csv', index=False)
sub.to_csv('submission.csv', index=False)
print("\nSaved submission.csv and submission_v5_optimal.csv successfully!")
print(f"TargetF1 distribution:\n{sub['TargetF1'].value_counts()}")
print(f"TargetRAUC min: {sub['TargetRAUC'].min():.4f}, max: {sub['TargetRAUC'].max():.4f}, mean: {sub['TargetRAUC'].mean():.4f}")
