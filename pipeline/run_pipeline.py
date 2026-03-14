# =============================================================
# run_pipeline.py
# Guwahati Daily Weather — Master Pipeline
# Feature Engineering + Retrain + Predictions
# =============================================================

import os
import sys
import time
from datetime import datetime
# Ensure Python finds other pipeline files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# CONFIGURATION — relative paths (works on any machine)
# ============================================================
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH        = os.path.join(BASE_DIR, '..', 'data', 'guwahati_weather_1972_2025.csv')
PROCESSED_DATA_PATH  = os.path.join(BASE_DIR, '..', 'data', 'processed_data.csv')
MODEL_ARTIFACTS_PATH = os.path.join(BASE_DIR, '..', 'model_artifacts')


# ============================================================
# HELPERS
# ============================================================
def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def separator():
    print("\n" + "="*60 + "\n")


# ============================================================
# STEP 1 — FEATURE ENGINEERING
# ============================================================
def run_feature_engineering():
    log("Starting Feature Engineering...")

    from feature_engineering import run_feature_engineering as fe

    start = time.time()
    df    = fe(
        filepath  = RAW_DATA_PATH,
        save_path = PROCESSED_DATA_PATH
    )
    elapsed = time.time() - start

    log(f"Feature Engineering Complete! ({elapsed:.1f}s)")
    log(f"Processed data shape: {df.shape}")
    return df


# ============================================================
# STEP 2 — RETRAIN MODEL
# ============================================================
def run_retrain():
    log("Starting Model Retraining...")

    from retrain import run_retrain as retrain


   

    start   = time.time()
    model   = retrain()
    elapsed = time.time() - start

    log(f"Model Retraining Complete! ({elapsed:.1f}s)")
    return model


# ============================================================
# STEP 3 — PREDICTIONS
# ============================================================
def run_predictions():
    log("Starting Prediction Pipeline...")

    from predict import run_prediction_pipeline

    start   = time.time()
    df      = run_prediction_pipeline(
        processed_data_path = PROCESSED_DATA_PATH,
        save_path           = PROCESSED_DATA_PATH
    )
    elapsed = time.time() - start

    log(f"Predictions Complete! ({elapsed:.1f}s)")
    log(f"Predicted Rain Days: {(df['predicted_rain'] == 'Rainy Day').sum()}")
    log(f"Predicted Dry Days:  {(df['predicted_rain'] == 'Dry Day').sum()}")
    log(f"Avg Rain Probability: {df['rain_probability'].mean():.2f}%")
    return df


# ============================================================
# STEP 4 — VALIDATE OUTPUT
# ============================================================
def validate_output(df):
    log("Validating output...")
    checks_passed = True

    if len(df) == 0:
        log("❌ FAILED — Dataframe is empty!")
        checks_passed = False
    else:
        log(f"✅ Row count: {len(df)}")

    if 'predicted_rain' not in df.columns:
        log("❌ FAILED — predicted_rain column missing!")
        checks_passed = False
    else:
        log("✅ predicted_rain column exists")

    if 'rain_probability' not in df.columns:
        log("❌ FAILED — rain_probability column missing!")
        checks_passed = False
    else:
        log("✅ rain_probability column exists")

    null_preds = df['predicted_rain'].isnull().sum()
    if null_preds > 0:
        log(f"⚠️ WARNING — {null_preds} null values in predicted_rain")
    else:
        log("✅ No null predictions")

    if os.path.exists(PROCESSED_DATA_PATH):
        size_mb = os.path.getsize(PROCESSED_DATA_PATH) / (1024 * 1024)
        log(f"✅ Output file exists ({size_mb:.2f} MB)")
    else:
        log("❌ FAILED — Output file not found!")
        checks_passed = False

    if checks_passed:
        log("✅ All validation checks passed!")
    else:
        log("❌ Some validation checks failed!")

    return checks_passed


# ============================================================
# MASTER PIPELINE
# ============================================================
def run_pipeline():
    separator()
    log("🌦️ GUWAHATI WEATHER PIPELINE STARTED")
    log(f"Run Date: {datetime.now().strftime('%A, %d %B %Y')}")
    separator()

    total_start = time.time()

    try:
        # Step 1 — Feature Engineering
        separator()
        run_feature_engineering()

        # Step 2 — Retrain Model
        separator()
        run_retrain()

        # Step 3 — Predictions
        separator()
        df_final = run_predictions()

        # Step 4 — Validate
        separator()
        validate_output(df_final)

        total_elapsed = time.time() - total_start
        separator()
        log("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        log(f"⏱️  Total Time: {total_elapsed:.1f} seconds")
        log(f"📁 Output: {PROCESSED_DATA_PATH}")
        separator()

        return df_final

    except Exception as e:
        separator()
        log(f"❌ PIPELINE FAILED — {str(e)}")
        import traceback
        traceback.print_exc()
        separator()
        raise e


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    run_pipeline()