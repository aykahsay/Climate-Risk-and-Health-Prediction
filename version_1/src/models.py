import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import logit, expit
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def evaluate_multi_metric(y_true, y_pred_proba):
    """Official multi-metric score: 0.60 * F1 + 0.40 * ROC_AUC. TargetF1 uses the mandatory
    default threshold of 0.5."""
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    score = 0.60 * f1 + 0.40 * auc
    return f1, auc, score


LGB_PARAMS = dict(
    objective='binary', learning_rate=0.010128940359921752, num_leaves=45, max_depth=3,
    feature_fraction=0.5055150782121488, bagging_fraction=0.63639015076024, bagging_freq=1,
    min_data_in_leaf=48, lambda_l1=0.001374996752144145, lambda_l2=2.577212543825008,
    scale_pos_weight=1.6217630156300469, n_estimators=341, verbose=-1,
)

XGB_PARAMS = dict(
    objective='binary:logistic', learning_rate=0.013051232685336154, max_depth=3,
    min_child_weight=9, subsample=0.7618942903727399, colsample_bytree=0.9566487112332043,
    reg_alpha=1.5546826273112857, reg_lambda=0.21053037325877444,
    scale_pos_weight=1.648167008682644, n_estimators=281, verbosity=0,
)

CAT_PARAMS = dict(
    loss_function='Logloss', learning_rate=0.01406941814710441, depth=3,
    l2_leaf_reg=5.189158692277641, scale_pos_weight=1.3382077395509584,
    bagging_temperature=0.5235541747075964, iterations=449, verbose=0,
)


def _train_cv_pseudo(model_ctor, name, X, y, X_test, n_splits=5, random_state=42, pseudo_threshold=0.05):
    """5-fold CV with validated pseudo-label augmentation: within each fold, a model trained on
    the 4 training folds only proposes pseudo-labels for the REAL test set (rows with predicted
    probability <= pseudo_threshold or >= 1-pseudo_threshold). Those confident rows are added to
    that fold's training data and a second model is trained on the augmented set, then scored on
    the held-out validation fold. The validation fold never influences the pseudo-labels used to
    augment its own training data, so this stays leakage-free. Threshold 0.05 was chosen via a
    nested honest comparison (see scratch/test_pseudo_labeling.py) that showed a small, consistent
    (non-noise) gain across multiple CV seeds versus no pseudo-labeling.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    y_arr = y if isinstance(y, np.ndarray) else y.values

    for train_idx, val_idx in skf.split(X, y_arr):
        X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
        X_va = X.iloc[val_idx]

        base_model = model_ctor(random_state)
        base_model.fit(X_tr, y_tr)
        test_proba_fold = base_model.predict_proba(X_test)[:, 1]

        confident_mask = (test_proba_fold <= pseudo_threshold) | (test_proba_fold >= 1 - pseudo_threshold)
        pseudo_labels = (test_proba_fold[confident_mask] >= 0.5).astype(int)
        X_aug = pd.concat([X_tr, X_test.iloc[confident_mask]], ignore_index=True)
        y_aug = np.concatenate([y_tr, pseudo_labels])

        aug_model = model_ctor(random_state)
        aug_model.fit(X_aug, y_aug)
        oof_preds[val_idx] = aug_model.predict_proba(X_va)[:, 1]
        test_preds += aug_model.predict_proba(X_test)[:, 1] / n_splits

    f1, auc, score = evaluate_multi_metric(y, oof_preds)
    print(f"[{name}] OOF F1: {f1:.5f} | ROC-AUC: {auc:.5f} | Final Score: {score:.5f}")
    return oof_preds, test_preds, score


def train_cv_lgb(X, y, X_test, n_splits=5, random_state=42):
    return _train_cv_pseudo(
        lambda rs: lgb.LGBMClassifier(**LGB_PARAMS, random_state=rs),
        "LightGBM", X, y, X_test, n_splits, random_state,
    )


def train_cv_xgb(X, y, X_test, n_splits=5, random_state=42):
    return _train_cv_pseudo(
        lambda rs: xgb.XGBClassifier(**XGB_PARAMS, random_state=rs),
        "XGBoost ", X, y, X_test, n_splits, random_state,
    )


def train_cv_catboost(X, y, X_test, n_splits=5, random_state=42):
    return _train_cv_pseudo(
        lambda rs: CatBoostClassifier(**CAT_PARAMS, random_seed=rs),
        "CatBoost", X, y, X_test, n_splits, random_state,
    )


def nested_logit_calibration(oof_proba, y, test_proba, n_splits=5, random_state=123):
    """Fits a single monotonic logit-scale calibration (p -> sigmoid(s*logit(p)+c)) with proper
    cross-validation: for each fold, (s, c) is chosen using only the OTHER folds' OOF predictions,
    then applied to the held-out fold. This is rank-order-preserving *within* each fold's own
    mapping and avoids fitting the calibration on the same points it is scored on (unlike a naive
    single grid-search over the full OOF set, which would leak the evaluation labels into the
    calibration choice). Returns the honestly cross-validated calibrated OOF array plus a single
    (s, c) fit on the FULL OOF (for deployment on the real, unseen test set).
    """
    eps = 1e-6
    y = np.asarray(y)
    l_oof = logit(np.clip(oof_proba, eps, 1 - eps))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    calibrated_oof = np.zeros_like(oof_proba)

    def best_sc(l_fit, y_fit):
        best_score, best_s, best_c = -1.0, 1.0, 0.0
        for s in np.linspace(0.5, 2.0, 16):
            for c in np.linspace(-1.5, 1.5, 31):
                p = expit(s * l_fit + c)
                f1, auc, score = evaluate_multi_metric(y_fit, p)
                if score > best_score:
                    best_score, best_s, best_c = score, s, c
        return best_s, best_c

    for fit_idx, hold_idx in skf.split(l_oof, y):
        s, c = best_sc(l_oof[fit_idx], y[fit_idx])
        calibrated_oof[hold_idx] = expit(s * l_oof[hold_idx] + c)

    f1, auc, score = evaluate_multi_metric(y, calibrated_oof)
    print(f"[Calibration] Nested CV calibrated OOF F1: {f1:.5f} | ROC-AUC: {auc:.5f} | Score: {score:.5f}")

    # Final (s, c) fit on the FULL OOF set, applied once to the real test predictions.
    s_final, c_final = best_sc(l_oof, y)
    l_test = logit(np.clip(test_proba, eps, 1 - eps))
    calibrated_test = expit(s_final * l_test + c_final)
    print(f"[Calibration] Final scale={s_final:.3f} shift={c_final:.3f} (fit on full OOF, applied to test)")

    return calibrated_oof, calibrated_test, score
