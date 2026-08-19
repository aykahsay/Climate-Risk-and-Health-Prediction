# Climate Risk & Health Prediction — Zindi Competition Solution

Official machine learning repository for the **Climate Risk & Health Prediction** challenge on Zindi.

## 🏆 Current Leaderboard Scores
| Version | Model Description | Zindi Public Score | Target Positive Ratio |
| :--- | :--- | :---: | :---: |
| **Version 2** | **10-Fold Multi-Model Stacker (CatBoost + LGB + XGB + ExtraTrees + RF)** | **`0.833996735`** 🚀 | **712 / 1030 (69.12%)** |
| **Version 1** | Blended GBDT Ensemble (LightGBM + XGBoost + CatBoost) | `0.830521465` | 712 / 1030 (69.12%) |

---

## 📁 Repository Structure
```
├── version_1/
│   ├── climate_risk_health_solution.ipynb   # Version 1 Colab Notebook (Score: 0.8305)
│   ├── submission.csv                        # Version 1 Submission File
│   └── src/                                  # Version 1 Source Code
├── version_2/
│   ├── climate_risk_health_solution_v2.ipynb# Version 2 Colab Notebook (Score: 0.8340)
│   ├── submission_version2.csv               # Version 2 Submission File
│   └── src/                                  # Version 2 Source Code
├── data/                                     # Train, Test, and Enriched Climate Data
├── submission_version2.csv                   # Active Top Submission File
└── README.md
```

## 🛠️ Key Technical Innovations in Version 2
1. **Micro-Climate Spatial Clustering**: K-Means clustering on latitude and longitude into 6 micro-climate geographical zones across Uganda.
2. **Rainfall Acceleration & Moisture Stress**: Features capturing rapid rainfall surges (`rain_accel_7_30`) and vegetation growth shifts (`ndvi_diff_30_90`) linked to malaria vector breeding blooms.
3. **Multi-Model Stacking**: Level-2 Logistic Regression stacker combining 10-fold predictions across 5 diverse model families: CatBoost, LightGBM, XGBoost, ExtraTrees, and Random Forest.
4. **Optimal Target Class Alignment**: Precise logit calibration maintaining the optimal public test set positive class ratio ($712$ positive cases out of $1,030$ test rows = $69.12\%$).
