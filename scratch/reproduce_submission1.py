import os
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np

from src.features import engineer_features
from src.models import (
    train_cv_lgb,
    train_cv_xgb,
    train_cv_catboost,
    evaluate_multi_metric,
    nested_logit_calibration
)

def main():
    print("="*60)
    print("REPRODUCING & ENHANCING SUBMISSION 1 (Mj2xpLL3)")
    print("="*60)
    
    train_df = pd.read_csv('data/Train.csv')
    test_df = pd.read_csv('data/Test.csv')
    climate_df = pd.read_csv('data/climate_features.csv')
    
    X_train_full, kmeans_model = engineer_features(train_df, climate_df, fit_kmeans=True)
    X_test_full, _ = engineer_features(test_df, climate_df, kmeans_model=kmeans_model, fit_kmeans=False)
    
    y = train_df['is_climate_sensitive'].astype(int).values
    test_ids = test_df['ID']
    
    X_train = X_train_full.drop(columns=['ID', 'is_climate_sensitive'], errors='ignore').copy()
    X_test = X_test_full.drop(columns=['ID'], errors='ignore').copy()
    
    print(f"Dataset X_train shape: {X_train.shape}")
    
    oof_cat, test_cat, cat_score = train_cv_catboost(X_train, pd.Series(y), X_test)
    oof_lgb, test_lgb, lgb_score = train_cv_lgb(X_train, pd.Series(y), X_test)
    oof_xgb, test_xgb, xgb_score = train_cv_xgb(X_train, pd.Series(y), X_test)
    
    # Submission 1 Model Blend: 0.40 CatBoost + 0.35 LightGBM + 0.25 XGBoost
    oof_sub1 = 0.40 * oof_cat + 0.35 * oof_lgb + 0.25 * oof_xgb
    test_sub1 = 0.40 * test_cat + 0.35 * test_lgb + 0.25 * test_xgb
    
    f1_sub1, auc_sub1, score_sub1 = evaluate_multi_metric(y, oof_sub1)
    print(f"\n--- SUBMISSION 1 REPRODUCED ENSEMBLE ---")
    print(f"OOF F1 @ 0.5: {f1_sub1:.5f} | ROC-AUC: {auc_sub1:.5f} | Multi-Metric Score: {score_sub1:.5f}")
    
    submission = pd.DataFrame({
        'ID': test_ids,
        'TargetF1': (test_sub1 >= 0.5).astype(int),
        'TargetRAUC': test_sub1
    })
    
    submission.to_csv('submission_sub1_reproduced.csv', index=False)
    submission.to_csv('submission.csv', index=False)
    print("\nSaved submission.csv and submission_sub1_reproduced.csv successfully!")
    print(f"TargetF1 distribution:\n{submission['TargetF1'].value_counts()}")
    print(f"Strict Threshold 0.5 check: {(submission['TargetF1'] == (submission['TargetRAUC'] >= 0.5).astype(int)).all()}")

if __name__ == '__main__':
    main()
