import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from scipy.special import logit, expit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

df_all['year'] = df_all['deathdate_dt'].dt.year
df_all['month'] = df_all['deathdate_dt'].dt.month
df_all['day'] = df_all['deathdate_dt'].dt.day
df_all['dayofweek'] = df_all['deathdate_dt'].dt.dayofweek
df_all['dayofyear'] = df_all['deathdate_dt'].dt.dayofyear
df_all['quarter'] = df_all['deathdate_dt'].dt.quarter

df_all['sin_month'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
df_all['cos_month'] = np.cos(2 * np.pi * df_all['month'] / 12.0)
df_all['sin_doy'] = np.sin(2 * np.pi * df_all['dayofyear'] / 365.25)
df_all['cos_doy'] = np.cos(2 * np.pi * df_all['dayofyear'] / 365.25)

df_all['log_age'] = np.log1p(df_all['age'])
df_all['is_age_0'] = (df_all['age'] == 0).astype(int)
df_all['is_age_1'] = (df_all['age'] == 1).astype(int)
df_all['is_age_2'] = (df_all['age'] == 2).astype(int)
df_all['is_under_5'] = (df_all['age'] < 5).astype(int)
df_all['is_elderly'] = (df_all['age'] >= 65).astype(int)
df_all['age_group'] = pd.cut(df_all['age'], bins=[-1, 0, 1, 3, 5, 12, 18, 40, 65, 120], labels=False)

df_all['gender_code'] = (df_all['gender'] == 'Female').astype(int)
df_all['zone_code'] = (df_all['zone'] == 'Rural').astype(int)

df_all['temp_range_day'] = df_all['max_temperature'] - df_all['min_temperature']
df_all['temp_range_30d'] = df_all['tmax_30d'] - df_all['tmin_30d']
df_all['temp_range_anomaly'] = df_all['temp_range_day'] - df_all['temp_range_mean_30d']
df_all['tavg_anomaly_30d'] = df_all['avg_temperature'] - df_all['tavg_30d']
df_all['tavg_anomaly_7d'] = df_all['avg_temperature'] - df_all['tavg_7d']
df_all['rain_daily_avg_30d'] = df_all['rain_sum_30d'] / 30.0
df_all['rain_ratio_7_30'] = df_all['rain_sum_7d'] / (df_all['rain_sum_30d'] + 1e-5)
df_all['precip_anomaly_30d'] = df_all['precipitation'] - df_all['rain_daily_avg_30d']
df_all['ndvi_diff_30_90'] = df_all['ndvi_30d'] - df_all['ndvi_90d']

# Interactions
df_all['under5_x_rain30d'] = df_all['is_under_5'] * df_all['rain_sum_30d']
df_all['under5_x_tavg30d'] = df_all['is_under_5'] * df_all['tavg_30d']
df_all['age_x_tavg30d'] = df_all['age'] * df_all['tavg_30d']

train_clean = df_all[df_all['is_train'] == 1].sort_values(by='ID').reset_index(drop=True)
test_clean = df_all[df_all['is_train'] == 0].sort_values(by='ID').reset_index(drop=True)

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'is_train', 'location', 'zone', 'gender', 'hot_days_30d']
features = [c for c in train_clean.columns if c not in drop_cols]

X = train_clean[features].copy()
y = train_clean['is_climate_sensitive'].astype(int).values
X_test = test_clean[features].copy()

# Add target encodings
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()

for col in ['age_group', 'month', 'year']:
    X[f'{col}_te'] = np.nan
    X_test[f'{col}_te'] = np.nan
    for train_idx, val_idx in skf.split(X, y):
        tr_y = y[train_idx]
        tr_col = X.iloc[train_idx][col]
        te_map = pd.Series(tr_y).groupby(tr_col).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
        X.iloc[val_idx, X.columns.get_loc(f'{col}_te')] = X.iloc[val_idx][col].map(te_map).fillna(global_mean)
    full_map = pd.Series(y).groupby(X[col]).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
    X_test[f'{col}_te'] = X_test[col].map(full_map).fillna(global_mean)

# Train CatBoost and LightGBM
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]
    
    # LGBM
    model_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5.0
    
    # CatBoost
    model_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, early_stopping_rounds=50, verbose=0)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / 5.0
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5.0

oof_raw = 0.40 * oof_cat + 0.35 * oof_lgb + 0.25 * oof_xgb
test_raw = 0.40 * test_cat + 0.35 * test_lgb + 0.25 * test_xgb

def eval_score(y_true, y_pred_proba):
    y_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score

print(f"Raw OOF Multi-Metric Score: {eval_score(y, oof_raw)[2]:.5f}")

# Joint 2D Calibration: Shift (c) and Temperature Scaling (s)
eps = 1e-6
p_clip = np.clip(oof_raw, eps, 1 - eps)
l = logit(p_clip)

best_score = -1.0
best_c = 0.0
best_s = 1.0

for s in np.linspace(0.5, 2.5, 41):
    for c in np.linspace(-2.0, 2.0, 81):
        p_cal = expit(s * l + c)
        f1, auc, score = eval_score(y, p_cal)
        if score > best_score:
            best_score = score
            best_f1 = f1
            best_auc = auc
            best_c = c
            best_s = s

print(f"\n--- OPTIMAL 2D PROBABILITY SCALING ---")
print(f"Scale s={best_s:.2f}, Shift c={best_c:.2f}")
print(f"Calibrated OOF F1 @ 0.5: {best_f1:.5f} | ROC-AUC: {best_auc:.5f} | Multi-Metric Score: {best_score:.5f}")

# Apply 2D Scaling to Test Predictions
test_clip = np.clip(test_raw, eps, 1 - eps)
test_calibrated = expit(best_s * logit(test_clip) + best_c)

sub = pd.DataFrame({
    'ID': test_clean['ID'],
    'TargetF1': (test_calibrated >= 0.5).astype(int),
    'TargetRAUC': test_calibrated
})

sub.to_csv('submission_sharpened.csv', index=False)
print("Saved sharpened submission to submission_sharpened.csv!")
print(f"Class distribution of TargetF1:\n{sub['TargetF1'].value_counts()}")
print(f"TargetRAUC min: {sub['TargetRAUC'].min():.4f}, max: {sub['TargetRAUC'].max():.4f}, mean: {sub['TargetRAUC'].mean():.4f}")
