import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold


def _base_features(df: pd.DataFrame, df_climate: pd.DataFrame, kmeans_model=None, fit_kmeans=False):
    climate_cols_to_drop = ['deathdate'] if 'deathdate' in df_climate.columns else []
    df = df.merge(df_climate.drop(columns=climate_cols_to_drop, errors='ignore'), on='ID', how='left')

    # Temporal & seasonality
    df['deathdate_dt'] = pd.to_datetime(df['deathdate'], errors='coerce')
    df['year'] = df['deathdate_dt'].dt.year
    df['month'] = df['deathdate_dt'].dt.month
    df['dayofyear'] = df['deathdate_dt'].dt.dayofyear
    df['weekofyear'] = df['deathdate_dt'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['deathdate_dt'].dt.quarter
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_doy'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['is_primary_wet_season'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_secondary_wet_season'] = df['month'].isin([9, 10, 11]).astype(int)
    df['is_dry_season'] = df['month'].isin([1, 2, 6, 7, 8, 12]).astype(int)

    # Demographic & vulnerability
    df['log_age'] = np.log1p(df['age'])
    df['sqrt_age'] = np.sqrt(df['age'])
    df['is_age_0'] = (df['age'] == 0).astype(int)
    df['is_age_1'] = (df['age'] == 1).astype(int)
    df['is_infant'] = (df['age'] < 1.0).astype(int)
    df['is_toddler'] = ((df['age'] >= 1.0) & (df['age'] < 5.0)).astype(int)
    df['is_under_5'] = (df['age'] < 5.0).astype(int)
    df['is_school_age'] = ((df['age'] >= 5.0) & (df['age'] < 18.0)).astype(int)
    df['is_adult'] = ((df['age'] >= 18.0) & (df['age'] < 60.0)).astype(int)
    df['is_senior'] = (df['age'] >= 60.0).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[-1, 0, 1, 3, 5, 12, 18, 40, 65, 120], labels=False)
    df['gender_female'] = (df['gender'] == 'Female').astype(int)
    df['zone_rural'] = (df['zone'] == 'Rural').astype(int)

    # Spatial & terrain
    coords = df[['latitude', 'longitude']].fillna(0)
    if fit_kmeans or kmeans_model is None:
        kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
        df['spatial_cluster'] = kmeans_model.fit_predict(coords)
    else:
        df['spatial_cluster'] = kmeans_model.predict(coords)
    lat_center, long_center = 0.45, 32.5
    df['dist_from_center'] = np.sqrt((df['latitude'] - lat_center) ** 2 + (df['longitude'] - long_center) ** 2)
    df['slope_elevation_ratio'] = df['slope'] / (df['elevation'] + 1e-5)
    df['is_highland'] = (df['elevation'] > 1300).astype(int)
    df['is_lowland'] = (df['elevation'] < 1100).astype(int)

    # Climate interactions & anomalies
    df['temp_range_recorded'] = df['max_temperature'] - df['min_temperature']
    df['temp_range_30d'] = df['tmax_30d'] - df['tmin_30d']
    df['temp_range_anomaly'] = df['temp_range_recorded'] - df['temp_range_mean_30d']
    df['tavg_anomaly_30d'] = df['avg_temperature'] - df['tavg_30d']
    df['tavg_anomaly_7d'] = df['avg_temperature'] - df['tavg_7d']
    df['tavg_trend_7_30'] = df['tavg_7d'] - df['tavg_30d']
    df['tavg_trend_30_90'] = df['tavg_30d'] - df['tavg_90d']
    df['rain_daily_avg_30d'] = df['rain_sum_30d'] / 30.0
    df['rain_daily_avg_7d'] = df['rain_sum_7d'] / 7.0
    df['rain_daily_avg_90d'] = df['rain_sum_90d'] / 90.0
    df['rain_accel_7_30'] = df['rain_daily_avg_7d'] - df['rain_daily_avg_30d']
    df['rain_accel_30_90'] = df['rain_daily_avg_30d'] - df['rain_daily_avg_90d']
    df['rain_ratio_7_30'] = df['rain_sum_7d'] / (df['rain_sum_30d'] + 1e-5)
    df['rain_ratio_30_90'] = df['rain_sum_30d'] / (df['rain_sum_90d'] + 1e-5)
    df['rain_intensity_30d'] = df['max_daily_rain_30d'] / (df['rain_sum_30d'] + 1e-5)
    df['rain_day_prop_30d'] = df['rain_days_30d'] / 30.0
    df['precip_anomaly_30d'] = df['precipitation'] - df['rain_daily_avg_30d']
    df['precip_anomaly_7d'] = df['precipitation'] - df['rain_daily_avg_7d']
    df['heavy_rain_day'] = (df['precipitation'] > 10.0).astype(int)
    df['high_temp_day'] = (df['max_temperature'] > 30.0).astype(int)
    df['ndvi_trend_30_90'] = df['ndvi_30d'] - df['ndvi_90d']
    df['ndvi_ratio_30_90'] = df['ndvi_30d'] / (df['ndvi_90d'] + 1e-5)

    # Demographic x climate vulnerability interactions
    df['under5_x_rain30d'] = df['is_under_5'] * df['rain_sum_30d']
    df['under5_x_tavg30d'] = df['is_under_5'] * df['tavg_30d']
    df['under5_x_precip'] = df['is_under_5'] * df['precipitation']
    df['under5_x_ndvi30d'] = df['is_under_5'] * df['ndvi_30d']
    df['under5_x_rain_accel'] = df['is_under_5'] * df['rain_accel_7_30']
    df['senior_x_tmax30d'] = df['is_senior'] * df['tmax_30d']
    df['senior_x_tavg_anomaly'] = df['is_senior'] * df['tavg_anomaly_30d']
    df['age_x_tavg30d'] = df['age'] * df['tavg_30d']
    df['age_x_rain30d'] = df['age'] * df['rain_sum_30d']
    df['age_x_ndvi30d'] = df['age'] * df['ndvi_30d']

    cols_to_drop = ['hot_days_30d', 'location', 'gender', 'zone', 'deathdate', 'deathdate_dt']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return df_clean, kmeans_model


def engineer_features(df_main: pd.DataFrame, df_climate: pd.DataFrame, kmeans_model=None, fit_kmeans=False):
    """Single-dataframe feature engineering (no target encoding). Kept for backward compatibility."""
    return _base_features(df_main, df_climate, kmeans_model=kmeans_model, fit_kmeans=fit_kmeans)


def _add_oof_target_encoding(X_train, y, X_test, cols, n_splits=5, smoothing=10, random_state=42):
    """Adds smoothed, out-of-fold target-encoded columns for `cols`. No leakage: each training
    row's encoding is computed only from the folds it is NOT part of."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = y.mean()
    for col in cols:
        te_col = f'{col}_te'
        X_train[te_col] = np.nan
        for tr_idx, va_idx in skf.split(X_train, y):
            tr_y = y[tr_idx]
            tr_col = X_train.iloc[tr_idx][col]
            stats = pd.Series(tr_y).groupby(tr_col).agg(['sum', 'count'])
            te_map = (stats['sum'] + smoothing * global_mean) / (stats['count'] + smoothing)
            X_train.iloc[va_idx, X_train.columns.get_loc(te_col)] = X_train.iloc[va_idx][col].map(te_map).fillna(global_mean)
        full_stats = pd.Series(y).groupby(X_train[col]).agg(['sum', 'count'])
        full_map = (full_stats['sum'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
        X_test[te_col] = X_test[col].map(full_map).fillna(global_mean)
    return X_train, X_test


def build_train_test_features(train_df: pd.DataFrame, test_df: pd.DataFrame, climate_df: pd.DataFrame):
    """Full pipeline: engineers features for train+test jointly (KMeans fit on train only,
    applied to test) and adds leakage-free OOF target encodings for low-cardinality columns."""
    X_train, kmeans_model = _base_features(train_df.copy(), climate_df, fit_kmeans=True)
    X_test, _ = _base_features(test_df.copy(), climate_df, kmeans_model=kmeans_model, fit_kmeans=False)

    y = train_df['is_climate_sensitive'].astype(int).values
    test_ids = test_df['ID'].copy()

    X_train = X_train.drop(columns=['ID', 'is_climate_sensitive'], errors='ignore')
    X_test = X_test.drop(columns=['ID'], errors='ignore')

    X_train, X_test = _add_oof_target_encoding(
        X_train, y, X_test, cols=['age_group', 'month', 'year', 'quarter', 'spatial_cluster']
    )

    return X_train, X_test, y, test_ids
