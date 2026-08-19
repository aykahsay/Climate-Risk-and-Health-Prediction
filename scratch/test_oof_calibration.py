import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import logit, expit
from src.features import engineer_features
from src.models import train_cv_lgb, train_cv_xgb, train_cv_catboost

train_df = pd.read_csv('data/Train.csv')
test_df = pd.read_csv('data/Test.csv')
climate_df = pd.read_csv('data/climate_features.csv')

X_train_full, kmeans_model = engineer_features(train_df, climate_df, fit_kmeans=True)
X_test_full, _ = engineer_features(test_df, climate_df, kmeans_model=kmeans_model, fit_kmeans=False)

y = train_df['is_climate_sensitive'].values
X_train = X_train_full.drop(columns=['ID', 'is_climate_sensitive'], errors='ignore')
X_test = X_test_full.drop(columns=['ID'], errors='ignore')

oof_lgb, _, _ = train_cv_lgb(X_train, pd.Series(y), X_test)
oof_xgb, _, _ = train_cv_xgb(X_train, pd.Series(y), X_test)
oof_cat, _, _ = train_cv_catboost(X_train, pd.Series(y), X_test)

oof_raw = 0.35 * oof_lgb + 0.35 * oof_xgb + 0.30 * oof_cat

f1_init = f1_score(y, (oof_raw >= 0.5).astype(int))
auc_init = roc_auc_score(y, oof_raw)
score_init = 0.60 * f1_init + 0.40 * auc_init

print(f"\n--- BEFORE CALIBRATION ---")
print(f"OOF F1 @ 0.5: {f1_init:.5f} | ROC-AUC: {auc_init:.5f} | Score: {score_init:.5f}")

# Logit shift tuning
eps = 1e-6
p_clipped = np.clip(oof_raw, eps, 1 - eps)
l = logit(p_clipped)

best_c = 0.0
best_score = score_init
best_f1 = f1_init
best_auc = auc_init

for c in np.linspace(-3.0, 3.0, 601):
    p_shifted = expit(l + c)
    f1 = f1_score(y, (p_shifted >= 0.5).astype(int))
    auc = roc_auc_score(y, p_shifted)
    score = 0.60 * f1 + 0.40 * auc
    if score > best_score:
        best_score = score
        best_f1 = f1
        best_auc = auc
        best_c = c

print(f"\n--- AFTER LOGIT SHIFT CALIBRATION ---")
print(f"Optimal shift c: {best_c:.4f}")
print(f"Calibrated OOF F1 @ 0.5: {best_f1:.5f} | ROC-AUC: {best_auc:.5f} | Score: {best_score:.5f}")
