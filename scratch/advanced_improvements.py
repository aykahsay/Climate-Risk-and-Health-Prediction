import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import logit, expit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

# Merge
train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')
test_df = test.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

df_all = pd.concat([train_df.assign(is_train=1), test_df.assign(is_train=0, is_climate_sensitive=np.nan)], ignore_index=True)
df_all['deathdate_dt'] = pd.to_datetime(df_all['deathdate'])

# Features
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

# Age features
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

# Age interactions
df_all['under5_x_rain30d'] = df_all['is_under_5'] * df_all['rain_sum_30d']
df_all['under5_x_tavg30d'] = df_all['is_under_5'] * df_all['tavg_30d']
df_all['age_x_rain30d'] = df_all['age'] * df_all['rain_sum_30d']

train_clean = df_all[df_all['is_train'] == 1].sort_values(by='ID').reset_index(drop=True)
test_clean = df_all[df_all['is_train'] == 0].sort_values(by='ID').reset_index(drop=True)

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'is_train', 'location', 'zone', 'gender', 'hot_days_30d']
features = [c for c in train_clean.columns if c not in drop_cols]

X = train_clean[features].copy()
y = train_clean['is_climate_sensitive'].astype(int)
X_test = test_clean[features].copy()

# Add Smoothed OOF Target Encoding for age_group and month
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
global_mean = y.mean()

for col in ['age_group', 'month', 'year']:
    X[f'{col}_te'] = np.nan
    X_test[f'{col}_te'] = np.nan
    
    for train_idx, val_idx in skf.split(X, y):
        tr_y = y.iloc[train_idx]
        tr_col = X.iloc[train_idx][col]
        
        te_map = tr_y.groupby(tr_col).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
        X.iloc[val_idx, X.columns.get_loc(f'{col}_te')] = X.iloc[val_idx][col].map(te_map).fillna(global_mean)
        
    full_map = y.groupby(X[col]).agg(lambda x: (x.sum() + 10 * global_mean) / (len(x) + 10))
    X_test[f'{col}_te'] = X_test[col].map(full_map).fillna(global_mean)

# Re-list features
features = list(X.columns)

def eval_score(y_true, y_pred_proba):
    y_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score

# Stage 1 Training: LightGBM, XGBoost, CatBoost, ExtraTrees, HistGB
print("--- STAGE 1 MODELING ---")
oof_lgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))

oof_xgb = np.zeros(len(X))
test_xgb = np.zeros(len(X_test))

oof_cat = np.zeros(len(X))
test_cat = np.zeros(len(X_test))

oof_et = np.zeros(len(X))
test_et = np.zeros(len(X_test))

X_imp = X.fillna(X.median())
X_test_imp = X_test.fillna(X.median())

for train_idx, val_idx in skf.split(X, y):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # LGBM
    model_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5.0
    
    # XGB
    model_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5.0
    
    # CatBoost
    model_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, early_stopping_rounds=50, verbose=0)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / 5.0
    
    # ExtraTrees
    model_et = ExtraTreesClassifier(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)
    model_et.fit(X_imp.iloc[train_idx], y_tr)
    oof_et[val_idx] = model_et.predict_proba(X_imp.iloc[val_idx])[:, 1]
    test_et += model_et.predict_proba(X_test_imp)[:, 1] / 5.0

print(f"LGBM  Score: {eval_score(y, oof_lgb)[2]:.5f}")
print(f"XGB   Score: {eval_score(y, oof_xgb)[2]:.5f}")
print(f"CatB  Score: {eval_score(y, oof_cat)[2]:.5f}")
print(f"ExTree Score: {eval_score(y, oof_et)[2]:.5f}")

oof_initial = 0.35 * oof_cat + 0.35 * oof_lgb + 0.20 * oof_xgb + 0.10 * oof_et
test_initial = 0.35 * test_cat + 0.35 * test_lgb + 0.20 * test_xgb + 0.10 * test_et
print(f"Stage 1 Blend Score: {eval_score(y, oof_initial)[2]:.5f}")

# STAGE 2: PSEUDO-LABELING ON CONFIDENT TEST PREDICTIONS
print("\n--- STAGE 2 PSEUDO-LABELING ---")
conf_high = test_initial > 0.88
conf_low = test_initial < 0.12
conf_mask = conf_high | conf_low

print(f"High confidence test samples found: {conf_mask.sum()} / {len(X_test)} (High: {conf_high.sum()}, Low: {conf_low.sum()})")

X_pseudo = pd.concat([X, X_test[conf_mask]], ignore_index=True)
y_pseudo = pd.concat([y, pd.Series((test_initial[conf_mask] >= 0.5).astype(int))], ignore_index=True)

oof_lgb_p = np.zeros(len(X))
test_lgb_p = np.zeros(len(X_test))

oof_cat_p = np.zeros(len(X))
test_cat_p = np.zeros(len(X_test))

oof_xgb_p = np.zeros(len(X))
test_xgb_p = np.zeros(len(X_test))

skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf2.split(X, y):
    # Train on full pseudo dataset excluding current val_idx
    train_mask_p = list(set(range(len(X_pseudo))) - set(val_idx))
    X_tr_p, y_tr_p = X_pseudo.iloc[train_mask_p], y_pseudo.iloc[train_mask_p]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # LGBM
    m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    m_lgb.fit(X_tr_p, y_tr_p, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb_p[val_idx] = m_lgb.predict_proba(X_va)[:, 1]
    test_lgb_p += m_lgb.predict_proba(X_test)[:, 1] / 5.0
    
    # CatBoost
    m_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, early_stopping_rounds=50, verbose=0)
    m_cat.fit(X_tr_p, y_tr_p, eval_set=(X_va, y_va), verbose=False)
    oof_cat_p[val_idx] = m_cat.predict_proba(X_va)[:, 1]
    test_cat_p += m_cat.predict_proba(X_test)[:, 1] / 5.0
    
    # XGBoost
    m_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
    m_xgb.fit(X_tr_p, y_tr_p, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb_p[val_idx] = m_xgb.predict_proba(X_va)[:, 1]
    test_xgb_p += m_xgb.predict_proba(X_test)[:, 1] / 5.0

oof_pseudo_blend = 0.40 * oof_cat_p + 0.40 * oof_lgb_p + 0.20 * oof_xgb_p
test_pseudo_blend = 0.40 * test_cat_p + 0.40 * test_lgb_p + 0.20 * test_xgb_p

f1_p, auc_p, score_p = eval_score(y, oof_pseudo_blend)
print(f"Pseudo-labeled OOF Score: {score_p:.5f} (F1: {f1_p:.5f}, AUC: {auc_p:.5f})")

# STAGE 3: LOGIT SHIFT CALIBRATION
eps = 1e-6
p_clipped = np.clip(oof_pseudo_blend, eps, 1 - eps)
l = logit(p_clipped)

best_c = 0.0
best_score = score_p
best_f1 = f1_p
best_auc = auc_p

for c in np.linspace(-3.0, 3.0, 601):
    p_shifted = expit(l + c)
    f1, auc, score = eval_score(y, p_shifted)
    if score > best_score:
        best_score = score
        best_f1 = f1
        best_auc = auc
        best_c = c

print(f"\n--- FINAL CALIBRATED & PSEUDO-LABELED PERFORMANCE ---")
print(f"Best shift constant c: {best_c:.4f}")
print(f"Final OOF F1 @ 0.5:  {best_f1:.5f}")
print(f"Final OOF ROC-AUC:   {best_auc:.5f}")
print(f"Final Multi-Metric Score: {best_score:.5f}")

# Generate Final Calibrated Submission
test_clipped = np.clip(test_pseudo_blend, eps, 1 - eps)
test_calibrated = expit(logit(test_clipped) + best_c)

sub = pd.DataFrame({
    'ID': test_clean['ID'],
    'TargetF1': (test_calibrated >= 0.5).astype(int),
    'TargetRAUC': test_calibrated
})

sub.to_csv('submission_advanced.csv', index=False)
print("Saved advanced submission to submission_advanced.csv")
