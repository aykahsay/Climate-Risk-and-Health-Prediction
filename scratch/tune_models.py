import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import json
from src.features import build_train_test_features

train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')
climate = pd.read_csv('data/climate_features.csv')
X, X_test_unused, y, _ = build_train_test_features(train, test, climate)

N_SPLITS = 5
SEEDS = [42, 7]  # average over 2 seeds for less noisy objective

def score_fn(y_true, proba):
    f1 = f1_score(y_true, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_true, proba)
    return 0.6 * f1 + 0.4 * auc, f1, auc

def cv_eval_lgb(params):
    scores = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        for tr, va in skf.split(X, y):
            m = lgb.LGBMClassifier(**params, random_state=seed, verbose=-1)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        s, _, _ = score_fn(y, oof)
        scores.append(s)
    return np.mean(scores)

def cv_eval_xgb(params):
    scores = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        for tr, va in skf.split(X, y):
            m = xgb.XGBClassifier(**params, random_state=seed, verbosity=0)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        s, _, _ = score_fn(y, oof)
        scores.append(s)
    return np.mean(scores)

def cv_eval_cat(params):
    scores = []
    for seed in SEEDS:
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        for tr, va in skf.split(X, y):
            m = CatBoostClassifier(**params, random_seed=seed, verbose=0)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        s, _, _ = score_fn(y, oof)
        scores.append(s)
    return np.mean(scores)

def obj_lgb(trial):
    params = {
        'objective': 'binary',
        'n_estimators': trial.suggest_int('n_estimators', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 7, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 60),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 5.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 3.0),
    }
    return cv_eval_lgb(params)

def obj_xgb(trial):
    params = {
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 3.0),
    }
    return cv_eval_xgb(params)

def obj_cat(trial):
    params = {
        'loss_function': 'Logloss',
        'iterations': trial.suggest_int('iterations', 150, 600),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        'depth': trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.7, 3.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    }
    return cv_eval_cat(params)

if __name__ == '__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    results = {}

    print(f'Tuning LightGBM ({n_trials} trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(obj_lgb, n_trials=n_trials, show_progress_bar=False)
    print('LGB best score:', study.best_value)
    print('LGB best params:', study.best_params)
    results['lgb'] = {'score': study.best_value, 'params': study.best_params}

    print(f'\nTuning XGBoost ({n_trials} trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(obj_xgb, n_trials=n_trials, show_progress_bar=False)
    print('XGB best score:', study.best_value)
    print('XGB best params:', study.best_params)
    results['xgb'] = {'score': study.best_value, 'params': study.best_params}

    print(f'\nTuning CatBoost ({max(15, n_trials//2)} trials)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(obj_cat, n_trials=max(15, n_trials // 2), show_progress_bar=False)
    print('CAT best score:', study.best_value)
    print('CAT best params:', study.best_params)
    results['cat'] = {'score': study.best_value, 'params': study.best_params}

    with open('scratch/best_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved best params to scratch/best_params.json')
