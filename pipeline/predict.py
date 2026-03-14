# =============================================================
# predict.py
# Guwahati Daily Weather — ML Prediction Pipeline
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# CONFIGURATION — relative paths for GitHub Actions
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

ORDINAL_COLS = [
    'visibility_level', 'temp_stress_category',
    'WindSpeedCategory', 'SeaLevelPressureCategory'
]

NOMINAL_COLS = ['wind_direction', 'seasons']

COLS_TO_SCALE = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike',
    'dew', 'humidity', 'windspeed', 'sealevelpressure', 'cloudcover',
    'visibility', 'moonphase', 'temprature_range', 'dew_point_diff',
    'winddir_sin', 'winddir_cos'
]


# ============================================================
# STEP 1 — LOAD ARTIFACTS
# ============================================================
def load_artifacts():
    print("🔄 Loading model artifacts...")
    model           = joblib.load(os.path.join(MODEL_ARTIFACTS_PATH, 'rainfall_model.pkl'))
    scaler          = joblib.load(os.path.join(MODEL_ARTIFACTS_PATH, 'scaler.pkl'))
    ordinal_encoder = joblib.load(os.path.join(MODEL_ARTIFACTS_PATH, 'ordinal_encoder.pkl'))
    onehot_encoder  = joblib.load(os.path.join(MODEL_ARTIFACTS_PATH, 'onehot_encoder.pkl'))
    print("✅ All artifacts loaded!")
    return model, scaler, ordinal_encoder, onehot_encoder


# ============================================================
# STEP 2 — LOAD PROCESSED DATA
# ============================================================
def load_processed_data(filepath):
    print("🔄 Loading processed data...")
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Processed data loaded: {df.shape}")
    return df


# ============================================================
# STEP 3 — PREPROCESS FOR PREDICTION
# ============================================================
def preprocess_for_prediction(df, scaler, ordinal_encoder, onehot_encoder):
    print("🔄 Preprocessing data for prediction...")
    X = df.copy()

    leakage_cols = [
        'precip', 'precipprob', 'precipcover',
        'precip_hours', 'precip_hours_bucket', 'rain_intensity'
    ]
    X.drop(columns=leakage_cols, inplace=True, errors='ignore')

    not_useful_cols = [
        'name', 'datetime', 'sunrise', 'sunset', 'description',
        'icon', 'stations', 'primary_station', 'secondary_stations',
        'weekday_name', 'conditions', 'preciptype',
        'is_remote_reading', 'is_weekend', 'years_bucket'
    ]
    X.drop(columns=not_useful_cols, inplace=True, errors='ignore')
    X.drop(columns=['is_rainy_day'], inplace=True, errors='ignore')

    # Recreate winddir_sin and winddir_cos
    if 'winddir_sin' not in X.columns or 'winddir_cos' not in X.columns:
        X['winddir'] = pd.to_numeric(X['winddir'], errors='coerce').fillna(0)
        X['winddir_sin'] = np.sin(np.radians(X['winddir']))
        X['winddir_cos'] = np.cos(np.radians(X['winddir']))
        print("✅ winddir_sin and winddir_cos recreated")

    X.drop(columns=['winddir'], inplace=True, errors='ignore')

    available_ordinal = [col for col in ORDINAL_COLS if col in X.columns]
    X[available_ordinal] = ordinal_encoder.transform(X[available_ordinal])

    available_nominal     = [col for col in NOMINAL_COLS if col in X.columns]
    encoded               = onehot_encoder.transform(X[available_nominal])
    encoded_feature_names = onehot_encoder.get_feature_names_out(available_nominal)
    encoded_df            = pd.DataFrame(encoded, columns=encoded_feature_names, index=X.index)
    X = pd.concat([X.drop(columns=available_nominal), encoded_df], axis=1)

    available_scale    = [col for col in COLS_TO_SCALE if col in X.columns]
    X[available_scale] = scaler.transform(X[available_scale])

    available_features = [col for col in FINAL_FEATURES if col in X.columns]
    X = X[available_features]

    print(f" Preprocessing complete — shape: {X.shape}")
    return X


# ============================================================
# STEP 4 — GENERATE PREDICTIONS
# ============================================================
def generate_predictions(df, X, model):
    print("🔄 Generating predictions...")
    y_pred       = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    df['predicted_rain']   = np.where(y_pred == 1, 'Rainy Day', 'Dry Day')
    df['rain_probability'] = (y_pred_proba * 100).round(2)
    print(f" Predictions generated!")
    print(df['predicted_rain'].value_counts())
    print(f"\n Average Rain Probability: {df['rain_probability'].mean():.2f}%")
    return df


# ============================================================
# STEP 5 — SAVE PREDICTIONS
# ============================================================
def save_predictions(df, save_path):
    df.to_csv(save_path, index=False)
    print(f"\n✅ Predictions saved to: {save_path}")


# ============================================================
# STEP 6 — EVALUATE MODEL
# ============================================================
def evaluate_model(df):
    if 'is_rainy_day' not in df.columns:
        print("⚠️ No actual labels — skipping evaluation")
        return
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    y_actual = (df['is_rainy_day'] == 'Rainy Day').astype(int)
    y_pred   = (df['predicted_rain'] == 'Rainy Day').astype(int)
    y_proba  = df['rain_probability'] / 100
    print("\n📊 Model Evaluation:")
    print(classification_report(y_actual, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_actual, y_proba):.4f}")
    print(confusion_matrix(y_actual, y_pred))


# ============================================================
# MASTER PREDICTION PIPELINE
# ============================================================
def run_prediction_pipeline(
    processed_data_path=PROCESSED_DATA_PATH,
    save_path=PROCESSED_DATA_PATH
):
    print("🚀 Starting Prediction Pipeline...\n")
    model, scaler, ordinal_encoder, onehot_encoder = load_artifacts()
    df          = load_processed_data(processed_data_path)
    df_original = df.copy()
    X           = preprocess_for_prediction(df, scaler, ordinal_encoder, onehot_encoder)
    df_original = generate_predictions(df_original, X, model)
    evaluate_model(df_original)
    save_predictions(df_original, save_path)
    print(f"\n🎯 Prediction Pipeline Complete!")
    return df_original


# ============================================================
# RUN — only when run directly on local machine
# ============================================================
if __name__ == "__main__":
    df = run_prediction_pipeline(
        processed_data_path=PROCESSED_DATA_PATH,
        save_path=PROCESSED_DATA_PATH
    )
    print("\nSample predictions:")
    print(df[['datetime', 'is_rainy_day', 'predicted_rain', 'rain_probability']].tail(10))