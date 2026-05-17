import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import pandas as pd
import joblib
from pathlib import Path
from src.Fleet_Assignment_Problem.operations.inference_ppo import generate_ppo_demand_state

def main():
    BASE_DIR = Path(__file__).resolve().parents[3]
    MODEL_PATH = BASE_DIR / "models" / "lstm_multi_input.keras"
    SCALER_PATH = BASE_DIR / "models" / "scaler_target.pkl"
    DATA_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    OUT_FILE = BASE_DIR / "data" / "processed" / "precomputed_demand_2025.parquet"

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df_history = pd.read_parquet(DATA_FILE)

    routes = df_history['route'].unique()
    all_demands = []

    print(f"Calcul de la matrice de demande pour {len(routes)} routes...")
    for r in routes:
        try:
            df_dem, _, _ = generate_ppo_demand_state(r, model, scaler, df_history, "2025-01-01")
            all_demands.append(df_dem)
        except Exception:
            continue

    final_df = pd.concat(all_demands)
    final_df.to_parquet(OUT_FILE, index=False)
    print(f"Demande figée et sauvegardée : {OUT_FILE}")

if __name__ == "__main__":
    main()