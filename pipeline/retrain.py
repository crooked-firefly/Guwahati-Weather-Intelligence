# =============================================================
# retrain.py
# Guwahati Daily Weather — Retrain ML Model on Latest Data
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

# ============================================================
# CONFIGURATION — relative paths
# ============================================================
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH  = os.path.join(BASE_DIR, '..', 'data', 'processed_data.csv')
MODEL_ARTIFACTS_PATH = os.path.join(BASE_DIR, '..', 'model_artifacts')

FINAL_FEATURES = [
    'tempmax', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover',
    'visibility', 'is_heatwave', 'is_coldwave', 'is_humidday', 'is_windyday',
    'pressure_drop_flag', 'wind_direction_E', 'wind_direction_NE',
    'wind_direction_NONE', 'wind_direction_NW', 'wind_direction_S',
    'wind_direction_SE', 'wind_direction_SW', 'seasons_Post-Monsoon',
    'seasons_Summer'
]

COLS_TO_SCALE = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'dew', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover',
    'visibility', 'moonphase', 'temprature_range', 'dew_point_diff',
    'winddir_sin', 'winddir_cos'
]

ORDINAL_COLS = ['visibility_level', 'temp_stress_category',
                'WindSpeedCategory', 'SeaLevelPressureCategory']

ORDINAL_CATEGORIES = [
    ['Low', 'Medium', 'High'],
    ['Cold Stress', 'Below Optimal', 'Optimal', 'Beyond Optimal', 'Heat Stress'],
    ['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong', 'Severe / Damaging'],
    ['Very Low Pressure (Deep Cyclonic System)', 'Low Pressure',
     'Slightly Below Normal', 'Normal Pressure', 'Slightly Above Normal',
     'High Pressure', 'Very High Pressure (Strong Anticyclone)']
]

NOMINAL_COLS = ['wind_direction', 'seasons']

LEAKAGE_COLS = [
    'precip', 'precipprob', 'precipcover',
    'precip_hours', 'precip_hours_bucket', 'rain_intensity',
    'predicted_rain', 'rain_probability'
]

NOT_USEFUL_COLS = [
    'name', 'datetime', 'sunrise', 'sunset', 'description',
    'icon', 'stations', 'primary_station', 'secondary_stations',
    'weekday_name', 'conditions', 'preciptype',
    'is_remote_reading', 'is_weekend', 'years_bucket',
    'actual_binary', 'predicted_binary', 'rain_prob_decimal'
]

TARGET_COL = 'is_rainy_day'


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
def load_data():
    print("📂 Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# STEP 2 — PREPARE FEATURES
# ============================================================
def prepare_features(df):
    print("\n🔧 Preparing features...")

    df = df[df[TARGET_COL].notna()].copy()
    y  = (df[TARGET_COL] == 'Rainy Day').astype(int)

    drop_cols = [col for col in LEAKAGE_COLS + NOT_USEFUL_COLS if col in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COL], errors='ignore')

    # Recreate winddir_sin/cos
    if 'winddir_sin' not in X.columns or 'winddir_cos' not in X.columns:
        if 'winddir' in X.columns:
            X['winddir_sin'] = np.sin(np.radians(X['winddir'].fillna(0)))
            X['winddir_cos'] = np.cos(np.radians(X['winddir'].fillna(0)))

    print(f"✅ Features prepared: {X.shape[1]} columns")
    print(f"   Rainy Days: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Dry Days:   {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    return X, y


# ============================================================
# STEP 3 — ENCODE AND SCALE
# ============================================================
def encode_and_scale(X):
    print("\n⚙️  Encoding and scaling...")

    available_ordinal = [col for col in ORDINAL_COLS if col in X.columns]
    ordinal_encoder   = OrdinalEncoder(
        categories=[ORDINAL_CATEGORIES[ORDINAL_COLS.index(col)] for col in available_ordinal],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    X[available_ordinal] = ordinal_encoder.fit_transform(X[available_ordinal])

    available_nominal = [col for col in NOMINAL_COLS if col in X.columns]
    onehot_encoder    = OneHotEncoder(
        sparse_output=False,
        handle_unknown='ignore'
    )
    encoded       = onehot_encoder.fit_transform(X[available_nominal])
    encoded_names = onehot_encoder.get_feature_names_out(available_nominal)
    encoded_df    = pd.DataFrame(encoded, columns=encoded_names, index=X.index)
    X             = pd.concat([X.drop(columns=available_nominal), encoded_df], axis=1)

    available_scale    = [col for col in COLS_TO_SCALE if col in X.columns]
    scaler             = StandardScaler()
    X[available_scale] = scaler.fit_transform(X[available_scale])

    print(f"✅ Encoding and scaling complete")
    return X, scaler, ordinal_encoder, onehot_encoder


# ============================================================
# STEP 4 — SELECT FINAL FEATURES
# ============================================================
def select_features(X):
    print("\n🎯 Selecting final features...")
    for col in FINAL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X_final = X[FINAL_FEATURES]
    print(f"✅ Final features: {len(FINAL_FEATURES)}")
    return X_final


# ============================================================
# STEP 5 — TRAIN MODEL
# ============================================================
def train_model(X, y):
    print("\n🤖 Training Logistic Regression...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    print(f"\n📊 New Model Performance:")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   F1 Score:  {f1*100:.2f}%")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    return model, accuracy, roc_auc


# ============================================================
# STEP 6 — SAVE ARTIFACTS
# ============================================================
def save_artifacts(model, scaler, ordinal_encoder, onehot_encoder):
    print(f"\n💾 Saving artifacts to: {MODEL_ARTIFACTS_PATH}")
    os.makedirs(MODEL_ARTIFACTS_PATH, exist_ok=True)

    joblib.dump(model,           os.path.join(MODEL_ARTIFACTS_PATH, 'rainfall_model.pkl'))
    joblib.dump(scaler,          os.path.join(MODEL_ARTIFACTS_PATH, 'scaler.pkl'))
    joblib.dump(ordinal_encoder, os.path.join(MODEL_ARTIFACTS_PATH, 'ordinal_encoder.pkl'))
    joblib.dump(onehot_encoder,  os.path.join(MODEL_ARTIFACTS_PATH, 'onehot_encoder.pkl'))

    print("✅ All artifacts saved!")
    print("   ✅ rainfall_model.pkl")
    print("   ✅ scaler.pkl")
    print("   ✅ ordinal_encoder.pkl")
    print("   ✅ onehot_encoder.pkl")


# ============================================================
# MASTER RETRAIN FUNCTION
# ============================================================
def run_retrain():
    print("=" * 55)
    print("  🔄 GUWAHATI WEATHER — MODEL RETRAINING")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    start = datetime.now()

    df                                              = load_data()
    X, y                                            = prepare_features(df)
    X_encoded, scaler, ordinal_encoder, onehot_encoder = encode_and_scale(X)
    X_final                                         = select_features(X_encoded)
    model, accuracy, roc_auc                        = train_model(X_final, y)
    save_artifacts(model, scaler, ordinal_encoder, onehot_encoder)

    elapsed = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 55)
    print("  ✅ RETRAINING COMPLETE!")
    print(f"  📊 Accuracy : {accuracy*100:.2f}%")
    print(f"  📈 ROC-AUC  : {roc_auc:.4f}")
    print(f"  ⏱️  Time     : {elapsed:.1f} seconds")
    print("=" * 55)

    return model


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_retrain()