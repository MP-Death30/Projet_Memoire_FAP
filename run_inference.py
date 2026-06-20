# run_inference.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # Sécurité additionnelle

import pandas as pd
import joblib
from pathlib import Path
import argparse

from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule, predict_demand_xgboost

def execute_inference(input_csv, predictor_type="LSTM"):
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_FILE = DATA_DIR / "enriched_schedule_ready.csv"

    print(f"Chargement du planning brut : {input_csv}")
    raw_schedule = pd.read_csv(input_csv, sep=r'[,;]', engine='python')
    df_history = pd.read_parquet(DATA_DIR / "dataset_lstm.parquet")

    if predictor_type == "LSTM":
        print("Initialisation du moteur LSTM...")
        model = tf.keras.models.load_model(MODELS_DIR / "lstm_multi_input.keras")
        scaler = joblib.load(MODELS_DIR / "scaler_target.pkl")
        enriched_df = predict_demand_for_schedule(raw_schedule, model, scaler, df_history, "2025-01-01")
    else:
        print("Initialisation du moteur XGBoost...")
        model_path = str(MODELS_DIR / "xgboost_demand_model.json")
        mapping_path = str(MODELS_DIR / "xgb_airport_mapping.pkl")
        enriched_df = predict_demand_xgboost(raw_schedule, model_path, mapping_path, df_history)

    # Sécurisation des formats pour Streamlit
    enriched_df['Dept Time'] = pd.to_datetime(enriched_df['Dept Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    enriched_df['Arr Time'] = pd.to_datetime(enriched_df['Arr Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    enriched_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Inférence terminée. Fichier prêt pour allocation : {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Chemin vers le planning brut CSV")
    parser.add_argument("--model", choices=["LSTM", "XGBOOST"], default="LSTM")
    args = parser.parse_args()
    
    execute_inference(args.input, args.model)