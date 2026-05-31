import pandas as pd
import joblib
from pathlib import Path
import os

# Blocage strict de la VRAM au niveau de l'OS avant l'import TF
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule
from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule

def generate():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    POOL_FILE = BASE_DIR / "data" / "processed" / "ppo_schedule_pool.pkl"

    print("--- PHASE 1/2 : MOTEUR TENSORFLOW (Génération de la Matrice) ---")
    lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
    lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
    df_history = pd.read_parquet(DATA_LSTM_FILE)

    EVAL_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    eval_df = pd.read_parquet(EVAL_FILE)
    all_airports = pd.concat([eval_df['From'], eval_df['To']]).unique()
    airport_to_idx = {apt: i for i, apt in enumerate(all_airports)}

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    num_aircraft = sum(inventory_map.get(row['fleet_id'], 5) for _, row in fleet_types_df.iterrows())

    NUM_SCENARIOS = 30
    schedule_pool = []
    
    print(f"Pré-calcul de {NUM_SCENARIOS} scénarios...")
    for i in range(NUM_SCENARIOS):
        raw_schedule = generate_dynamic_schedule(num_aircraft)
        sched = predict_demand_for_schedule(raw_schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
        
        sched['Dept Time'] = pd.to_datetime(sched['Dept Time'])
        sched['Arr Time'] = pd.to_datetime(sched['Arr Time'])
        t0 = sched['Dept Time'].min()
        sched['Dept_Time_Minutes'] = (sched['Dept Time'] - t0).dt.total_seconds() / 60.0
        sched['Arr_Time_Minutes'] = (sched['Arr Time'] - t0).dt.total_seconds() / 60.0
        
        sched['Origin_Idx'] = sched['From'].map(airport_to_idx).fillna(0)
        sched['Dest_Idx'] = sched['To'].map(airport_to_idx).fillna(0)
        
        if 'flight_demand' not in sched.columns:
            sched['flight_demand'] = 150.0
        sched['Predicted_Demand'] = sched['flight_demand'].clip(upper=180)
        
        schedule_pool.append(sched)
        if (i + 1) % 10 == 0:
            print(f"  -> {i + 1} / {NUM_SCENARIOS} générés.")

    joblib.dump(schedule_pool, POOL_FILE)
    print(f"Pool sauvegardé avec succès : {POOL_FILE}")

if __name__ == "__main__":
    generate()