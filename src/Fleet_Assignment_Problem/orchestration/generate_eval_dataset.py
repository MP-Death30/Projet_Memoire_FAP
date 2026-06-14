import pandas as pd
from pathlib import Path
import os
import joblib
import tensorflow as tf
import random
import numpy as np
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule

def generate_evaluation_dataset():
    BASE_DIR = Path(__file__).resolve().parents[3]
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    
    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    num_aircraft = sum(inventory_map.get(row['fleet_id'], 5) for _, row in fleet_types_df.iterrows())
    
    df_history = pd.read_parquet(DATA_LSTM_FILE)
    
    # Fixation de l'entropie pour garantir un réseau identique à chaque évaluation
    random.seed(42)
    np.random.seed(42)
    raw_schedule = generate_dynamic_schedule(num_aircraft)
    
    predictor_type = os.environ.get("PREDICTOR_TYPE", "LSTM")
    
    if predictor_type == "LSTM":
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule
        lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
        lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
        enriched_schedule = predict_demand_for_schedule(raw_schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
    else:
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_xgboost
        XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.json"
        MAPPING_PATH = BASE_DIR / "models" / "xgb_airport_mapping.pkl"
        enriched_schedule = predict_demand_xgboost(raw_schedule, XGB_MODEL_PATH, MAPPING_PATH, df_history)
    
    enriched_schedule.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    generate_evaluation_dataset()