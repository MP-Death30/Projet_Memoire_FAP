import subprocess
import time
import pandas as pd
import json
import os
from pathlib import Path

def run_benchmark():
    BASE_DIR = Path(__file__).resolve().parents[3]
    METRICS_FILE = BASE_DIR / "data" / "processed" / "temp_metrics.json"
    RESULTS_CSV = BASE_DIR / "data" / "processed" / "benchmark_results.csv"
    
    pipelines = [
        {"agent": "PPO", "predictor": "LSTM"},
        {"agent": "PPO", "predictor": "XGBOOST"},
        {"agent": "MAPPO", "predictor": "LSTM"},
        {"agent": "MAPPO", "predictor": "XGBOOST"}
    ]
    
    results = []
    
    for pipe in pipelines:
        agent = pipe["agent"]
        predictor = pipe["predictor"]
        print(f"\n--- DÉMARRAGE PIPELINE : {agent} + {predictor} ---")
        
        env = os.environ.copy()
        env["PREDICTOR_TYPE"] = predictor
        
        if METRICS_FILE.exists():
            METRICS_FILE.unlink()
            
        metrics_record = {
            "Agent": agent,
            "Predictor": predictor,
            "Pool_Gen_Time_sec": 0,
            "Train_Time_sec": 0,
            "Eval_Time_sec": 0,
            "Margin": 0,
            "Spill_Rate": 0
        }
        
        try:
            t0 = time.perf_counter()
            if agent == "PPO":
                subprocess.run(["python", "src/Fleet_Assignment_Problem/orchestration/generate_pool.py"], env=env, check=True)
            metrics_record["Pool_Gen_Time_sec"] = round(time.perf_counter() - t0, 2)
            
            t0 = time.perf_counter()
            train_script = f"src/Fleet_Assignment_Problem/models/train_{agent.lower()}.py"
            subprocess.run(["python", train_script], env=env, check=True)
            metrics_record["Train_Time_sec"] = round(time.perf_counter() - t0, 2)
            
            t0 = time.perf_counter()
            eval_script = f"src/Fleet_Assignment_Problem/orchestration/evaluate_{agent.lower()}.py"
            subprocess.run(["python", eval_script], env=env, check=True)
            metrics_record["Eval_Time_sec"] = round(time.perf_counter() - t0, 2)
            
            if METRICS_FILE.exists():
                with open(METRICS_FILE, 'r') as f:
                    eval_metrics = json.load(f)
                metrics_record["Margin"] = eval_metrics.get("Margin_Generated", 0)
                metrics_record["Spill_Rate"] = eval_metrics.get("Spill_Rate", 0)
                
        except subprocess.CalledProcessError as e:
            print(f"Échec critique sur le pipeline {agent}-{predictor} : {e}")
            metrics_record["Margin"] = "ERROR"
            
        results.append(metrics_record)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_CSV, index=False)
        print(f"Pipeline {agent}-{predictor} terminé.")

    print(f"\nBenchmark global achevé. Registre : {RESULTS_CSV}")

if __name__ == "__main__":
    run_benchmark()