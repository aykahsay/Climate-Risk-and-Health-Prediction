import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import logit, expit

def optimize_logit_shift(y_true, y_pred_proba):
    # Clip probas to avoid inf in logit
    eps = 1e-6
    p_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    l = logit(p_clipped)
    
    best_c = 0.0
    best_f1 = f1_score(y_true, (y_pred_proba >= 0.5).astype(int))
    best_auc = roc_auc_score(y_true, y_pred_proba)
    best_score = 0.60 * best_f1 + 0.40 * best_auc
    
    for c in np.linspace(-3.0, 3.0, 601):
        p_shifted = expit(l + c)
        f1 = f1_score(y_true, (p_shifted >= 0.5).astype(int))
        auc = roc_auc_score(y_true, p_shifted)
        score = 0.60 * f1 + 0.40 * auc
        if score > best_score:
            best_score = score
            best_f1 = f1
            best_auc = auc
            best_c = c
            
    print(f"Original F1 @ 0.5: {f1_score(y_true, (y_pred_proba >= 0.5).astype(int)):.5f}")
    print(f"Shifted c={best_c:.3f} | F1 @ 0.5: {best_f1:.5f} | AUC: {best_auc:.5f} | Best Score: {best_score:.5f}")
    return best_c, expit(l + best_c)

# Test on simulated predictions
np.random.seed(42)
y_sim = np.random.binomial(1, 0.65, size=1000)
p_sim = np.clip(y_sim * 0.4 + np.random.normal(0.4, 0.2, size=1000), 0.01, 0.99)
optimize_logit_shift(y_sim, p_sim)
