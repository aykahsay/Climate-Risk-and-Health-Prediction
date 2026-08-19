import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import logit, expit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')

train_df = train.merge(climate.drop(columns=['deathdate'], errors='ignore'), on='ID', how='left')

# Group by location or district
train_df['district'] = train_df['location'].apply(lambda x: x.split(',')[-2].strip() if len(x.split(','))>=2 else x)
groups = train_df['location'].values

print("--- EVALUATING GROUP K-FOLD (Grouped by Location) ---")

# Simple features for test
train_df['deathdate_dt'] = pd.to_datetime(train_df['deathdate'])
train_df['year'] = train_df['deathdate_dt'].dt.year
train_df['month'] = train_df['deathdate_dt'].dt.month
train_df['dayofyear'] = train_df['deathdate_dt'].dt.dayofyear
train_df['log_age'] = np.log1p(train_df['age'])
train_df['is_under_5'] = (train_df['age'] < 5).astype(int)
train_df['gender_female'] = (train_df['gender'] == 'Female').astype(int)
train_df['zone_rural'] = (train_df['zone'] == 'Rural').astype(int)

train_df['temp_range_day'] = train_df['max_temperature'] - train_df['min_temperature']
train_df['tavg_anomaly_30d'] = train_df['avg_temperature'] - train_df['tavg_30d']
train_df['rain_ratio_7_30'] = train_df['rain_sum_7d'] / (train_df['rain_sum_30d'] + 1e-5)
train_df['precip_anomaly_30d'] = train_df['precipitation'] - (train_df['rain_sum_30d'] / 30.0)

drop_cols = ['ID', 'deathdate', 'deathdate_dt', 'is_climate_sensitive', 'location', 'district', 'zone', 'gender', 'hot_days_30d']
features = [c for c in train_df.columns if c not in drop_cols]

X = train_df[features]
y = train_df['is_climate_sensitive'].astype(int)

def eval_score(y_true, y_pred_proba):
    y_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score

sgkf = StratifiedGroupKFold(n_splits=5)
oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

for train_idx, val_idx in sgkf.split(X, y, groups=groups):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    # LGBM
    model_lgb = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, num_leaves=15, random_state=42, verbose=-1)
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
    
    # CatBoost
    model_cat = CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, random_seed=42, early_stopping_rounds=30, verbose=0)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)[:, 1]

f1_l, auc_l, score_l = eval_score(y, oof_lgb)
f1_c, auc_c, score_c = eval_score(y, oof_cat)

print(f"[GroupKFold LGBM]     OOF F1: {f1_l:.5f} | ROC-AUC: {auc_l:.5f} | Score: {score_l:.5f}")
print(f"[GroupKFold CatBoost] OOF F1: {f1_c:.5f} | ROC-AUC: {auc_c:.5f} | Score: {score_c:.5f}")

# Look at feature importances under GroupKFold
model_cat.fit(X, y, verbose=0)
imp = pd.Series(model_cat.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- TOP FEATURE IMPORTANCES UNDER SPATIAL GENERALIZATION ---")
print(imp.head(15))
