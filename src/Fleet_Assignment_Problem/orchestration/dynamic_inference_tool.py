import argparse
import pandas as pd
from pathlib import Path
import os
import joblib
import tensorflow as tf
import subprocess
import shutil
import json

def run_dynamic_pipeline(predictor_type, agent_type):
    BASE_DIR = Path(__file__).resolve().parents[3]
    BASE_SCHEDULE = BASE_DIR / "data" / "processed" / "base_schedule_unpredicted.csv"
    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    EVAL_TARGET = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    FINAL_OUTPUT = BASE_DIR / "data" / "processed" / f"final_allocation_{predictor_type}_{agent_type}.csv"
    
    if not BASE_SCHEDULE.exists():
        raise FileNotFoundError("Exécuter export_base_schedule.py avant d'utiliser cet outil.")
        
    raw_schedule = pd.read_csv(BASE_SCHEDULE)
    df_history = pd.read_parquet(DATA_LSTM_FILE)
    
    if predictor_type == "LSTM":
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule
        lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
        lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
        enriched_schedule = predict_demand_for_schedule(raw_schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
    elif predictor_type == "XGBOOST":
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_xgboost
        XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.json"
        MAPPING_PATH = BASE_DIR / "models" / "xgb_airport_mapping.pkl"
        enriched_schedule = predict_demand_xgboost(raw_schedule, XGB_MODEL_PATH, MAPPING_PATH, df_history)

    enriched_schedule.to_parquet(EVAL_TARGET, index=False)
    
    env = os.environ.copy()
    env["PREDICTOR_TYPE"] = predictor_type
    
    eval_script = f"src/Fleet_Assignment_Problem/orchestration/evaluate_{agent_type.lower()}.py"
    if not (BASE_DIR / eval_script).exists():
        raise FileNotFoundError(f"Le script d'évaluation {eval_script} est introuvable.")
        
    subprocess.run(["python", eval_script], env=env, check=True)
    
    METRICS_FILE = BASE_DIR / "data" / "processed" / "temp_metrics.json"
    if METRICS_FILE.exists():
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
            print(f"[{predictor_type} + {agent_type}] Métriques : {metrics}")
    
    # Cartographie des fichiers de sortie selon l'agent
    output_map = {
        "GREEDY": BASE_DIR / "data" / "processed" / "greedy_allocations.csv",
        "PPO": BASE_DIR / "data" / "processed" / "ppo_allocations.csv",
        "MAPPO": BASE_DIR / "data" / "processed" / "mappo_allocations.csv" 
    }
    
    source_output = output_map.get(agent_type)
    if source_output and source_output.exists():
        shutil.copy(source_output, FINAL_OUTPUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outil d'inférence dynamique pour le Fleet Assignment Problem")
    parser.add_argument("--predictor", type=str, required=True, choices=["LSTM", "XGBOOST"])
    parser.add_argument("--agent", type=str, required=True, choices=["GREEDY", "PPO", "MAPPO"])
    
    args = parser.parse_args()
    run_dynamic_pipeline(args.predictor, args.agent)