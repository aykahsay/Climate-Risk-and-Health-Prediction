import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
from src.features import build_train_test_features
from src.models import LGB_PARAMS

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')
X, X_test, y, test_ids = build_train_test_features(train, test, climate)


def eval_score(y_true, proba):
    f1 = f1_score(y_true, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_true, proba)
    return f1, auc, 0.6 * f1 + 0.4 * auc


CONFIDENCE_THRESHOLDS = [0.03, 0.05, 0.10, 0.15]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_oof = np.zeros(len(X))
pseudo_oof = {thr: np.zeros(len(X)) for thr in CONFIDENCE_THRESHOLDS}
pseudo_counts = {thr: [] for thr in CONFIDENCE_THRESHOLDS}

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    # Baseline model: trained only on this fold's 4/5 training split
    base_model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
    base_model.fit(X_tr, y_tr)
    baseline_oof[va_idx] = base_model.predict_proba(X_va)[:, 1]

    # Predict on the REAL test set using only this fold's training data (never touches val fold)
    test_proba = base_model.predict_proba(X_test)[:, 1]

    for thr in CONFIDENCE_THRESHOLDS:
        confident_mask = (test_proba <= thr) | (test_proba >= 1 - thr)
        pseudo_labels = (test_proba[confident_mask] >= 0.5).astype(int)
        pseudo_counts[thr].append(confident_mask.sum())

        X_pseudo = X_test.iloc[confident_mask].copy()
        X_aug = pd.concat([X_tr, X_pseudo], ignore_index=True)
        y_aug = np.concatenate([y_tr, pseudo_labels])

        aug_model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=42)
        aug_model.fit(X_aug, y_aug)
        pseudo_oof[thr][va_idx] = aug_model.predict_proba(X_va)[:, 1]

f1, auc, score = eval_score(y, baseline_oof)
print(f"BASELINE (no pseudo-labels): F1={f1:.5f} AUC={auc:.5f} score={score:.5f}")
print()
for thr in CONFIDENCE_THRESHOLDS:
    f1, auc, score = eval_score(y, pseudo_oof[thr])
    avg_n = np.mean(pseudo_counts[thr])
    print(f"PSEUDO thr={thr:.2f} (avg {avg_n:.0f}/{len(X_test)} test rows used): "
          f"F1={f1:.5f} AUC={auc:.5f} score={score:.5f}")
