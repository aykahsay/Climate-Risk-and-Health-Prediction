import pandas as pd

from src.features import build_train_test_features
from src.models import (
    train_cv_lgb,
    train_cv_xgb,
    train_cv_catboost,
    evaluate_multi_metric,
    nested_logit_calibration,
)


def main():
    print("=" * 60)
    print("CLIMATE RISK AND HEALTH PREDICTION - TRAINING PIPELINE")
    print("=" * 60)

    train_df = pd.read_csv('data/Train.csv')
    test_df = pd.read_csv('data/Test.csv')
    climate_df = pd.read_csv('data/climate_features.csv')
    sample_sub = pd.read_csv('data/SampleSubmission.csv')

    print(f"Loaded Train: {train_df.shape}, Test: {test_df.shape}, Climate: {climate_df.shape}")

    X_train, X_test, y, test_ids = build_train_test_features(train_df, test_df, climate_df)
    print(f"Engineered Train Features: {X_train.shape[1]} features.")

    print("\n--- Training Models (5-Fold Stratified CV, tuned hyperparameters) ---")
    oof_lgb, test_lgb, score_lgb = train_cv_lgb(X_train, y, X_test)
    oof_xgb, test_xgb, score_xgb = train_cv_xgb(X_train, y, X_test)
    oof_cat, test_cat, score_cat = train_cv_catboost(X_train, y, X_test)

    print("\n--- Finding Optimal Ensemble Weights ---")
    best_score, best_weights = -1.0, None
    weight_candidates = [
        (0.40, 0.30, 0.30), (0.34, 0.33, 0.33), (0.30, 0.30, 0.40),
        (0.25, 0.25, 0.50), (0.50, 0.25, 0.25), (0.25, 0.50, 0.25),
        (0.20, 0.20, 0.60), (0.40, 0.20, 0.40),
    ]
    for w in weight_candidates:
        oof_blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
        f1, auc, score = evaluate_multi_metric(y, oof_blend)
        if score > best_score:
            best_score, best_weights = score, w
            best_f1, best_auc = f1, auc

    print(f"[ENSEMBLE] Weights: LGB={best_weights[0]}, XGB={best_weights[1]}, Cat={best_weights[2]}")
    print(f"[ENSEMBLE] OOF F1: {best_f1:.5f} | ROC-AUC: {best_auc:.5f} | Score: {best_score:.5f}")

    oof_blend = best_weights[0] * oof_lgb + best_weights[1] * oof_xgb + best_weights[2] * oof_cat
    test_blend = best_weights[0] * test_lgb + best_weights[1] * test_xgb + best_weights[2] * test_cat

    print("\n--- Nested (leakage-free) Logit Calibration ---")
    oof_calibrated, test_calibrated, calibrated_score = nested_logit_calibration(oof_blend, y, test_blend)

    # Use the calibrated output only if it honestly beats the raw ensemble under nested CV;
    # otherwise keep the simpler, already-validated raw ensemble output.
    if calibrated_score > best_score:
        print(f"\nUsing CALIBRATED predictions (nested score {calibrated_score:.5f} > raw {best_score:.5f})")
        final_test_proba = test_calibrated
        final_score = calibrated_score
    else:
        print(f"\nCalibration did not help under nested CV ({calibrated_score:.5f} <= {best_score:.5f}); using raw ensemble")
        final_test_proba = test_blend
        final_score = best_score

    final_label = (final_test_proba >= 0.5).astype(int)

    submission = pd.DataFrame({
        'ID': test_ids,
        'TargetF1': final_label,
        'TargetRAUC': final_test_proba,
    })
    submission.to_csv('submission.csv', index=False)
    print(f"\nSaved submission to submission.csv (estimated OOF score: {final_score:.5f})")

    print("\n--- SUBMISSION VERIFICATION ---")
    print(f"Shape match: {submission.shape == sample_sub.shape} (Expected: {sample_sub.shape}, Actual: {submission.shape})")
    print(f"Columns match: {list(submission.columns) == list(sample_sub.columns)}")
    print(f"Missing values: {submission.isnull().sum().sum()}")
    print(f"TargetF1 distribution:\n{submission['TargetF1'].value_counts()}")
    print(f"TargetRAUC min: {submission['TargetRAUC'].min():.4f}, max: {submission['TargetRAUC'].max():.4f}, mean: {submission['TargetRAUC'].mean():.4f}")
    is_threshold_correct = (submission['TargetF1'] == (submission['TargetRAUC'] >= 0.5).astype(int)).all()
    print(f"Strict Threshold 0.5 constraint satisfied: {is_threshold_correct}")


if __name__ == '__main__':
    main()
