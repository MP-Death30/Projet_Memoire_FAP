import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
import joblib

# Importations des modules d'architecture
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule
from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule, predict_demand_xgboost

def setup_directories(base_dir: Path):
    dirs = [
        base_dir / "data" / "benchmark",
        base_dir / "data" / "benchmark" / "lstm",
        base_dir / "data" / "benchmark" / "xgboost",
        base_dir / "data" / "processed"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def calculate_metrics(df: pd.DataFrame) -> tuple:
    # Remplissage de sécurité pour le tarif si manquant dans la matrice
    if 'Tarif' not in df.columns:
        df['Tarif'] = 100.0

    # Isolement des vols non assignés
    df['Agent_Capacity'] = np.where(df['Agent_ID'] == -1, 0, df['Agent_Capacity'])
    
    pax_assignes = np.minimum(df['Predicted_Demand'], df['Agent_Capacity'])
    pax_spill = df['Predicted_Demand'] - pax_assignes

    ca_recupere = np.sum(pax_assignes * df['Tarif'])
    ca_possible = np.sum(df['Predicted_Demand'] * df['Tarif'])
    
    marge_recuperee = df['Margin_Generated'].sum()
    # Estimation de la marge possible (CA maximum - coût fixe estimé de l'affectation optimale)
    # Remplacer 'Max_Possible_Margin' par le calcul interne exact de votre environnement s'il est disponible
    marge_possible = marge_recuperee + np.sum(pax_spill * df['Tarif'] * 1.0) 

    total_demand = df['Predicted_Demand'].sum()
    taux_spill = pax_spill.sum() / total_demand if total_demand > 0 else 0

    return ca_recupere, ca_possible, marge_recuperee, marge_possible, taux_spill

def execute_benchmark(n_plannings: int):
    BASE_DIR = Path(__file__).resolve().parent
    setup_directories(BASE_DIR)
    
    # Chargement des artefacts de prédiction en mémoire vive
    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    EVAL_TARGET = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    df_history = pd.read_parquet(DATA_LSTM_FILE)

    lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
    lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
    
    XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.json"
    MAPPING_PATH = BASE_DIR / "models" / "xgb_airport_mapping.pkl"

    models_demand = ["LSTM", "XGBOOST"]
    agents_assign = ["GREEDY", "PPO", "MAPPO"]
    
    output_map = {
        "GREEDY": BASE_DIR / "data" / "processed" / "greedy_allocations.csv",
        "PPO": BASE_DIR / "data" / "processed" / "ppo_allocations.csv",
        "MAPPO": BASE_DIR / "data" / "processed" / "mappo_allocations.csv"
    }
    
    results = {f"{md}_{ag}": {'ca_ratio': [], 'margin_ratio': [], 'spill': []} 
               for md in models_demand for ag in agents_assign}

    for i in range(n_plannings):
        plan_id = f"planning_{i}"
        plan_path = BASE_DIR / "data" / "benchmark" / f"{plan_id}.csv"
        
        # 1. Génération
        schedule = generate_dynamic_schedule(num_aircraft=28)  #generate_dynamic_schedule(num_aircraft, days_to_simulate=7, base_date=datetime(2025, 1, 1)):
        # Formatage temporel de sécurité
        if 'Dept Time' in schedule.columns:
            schedule['Dept Time'] = pd.to_datetime(schedule['Dept Time'], dayfirst=True, format='mixed')
            if 'Arr Time' in schedule.columns:
                schedule['Arr Time'] = pd.to_datetime(schedule['Arr Time'], dayfirst=True, format='mixed')
            schedule = schedule.sort_values(by='Dept Time').reset_index(drop=True)
            
        schedule.to_csv(plan_path, index=False)

        for md_name in models_demand:
            # 2. Prédiction de la demande selon le modèle
            if md_name == "LSTM":
                enriched_schedule = predict_demand_for_schedule(schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
                save_path = BASE_DIR / "data" / "benchmark" / "lstm" / f"{plan_id}_pred.parquet"
            else:
                enriched_schedule = predict_demand_xgboost(schedule, XGB_MODEL_PATH, MAPPING_PATH, df_history)
                save_path = BASE_DIR / "data" / "benchmark" / "xgboost" / f"{plan_id}_pred.parquet"

            if 'Dept Time' in enriched_schedule.columns:
                enriched_schedule['Dept Time'] = pd.to_datetime(enriched_schedule['Dept Time'])
                enriched_schedule = enriched_schedule.sort_values(by='Dept Time').reset_index(drop=True)

            enriched_schedule.to_parquet(save_path, index=False)
            enriched_schedule.to_parquet(EVAL_TARGET, index=False) # Cible d'évaluation pour le sous-processus

            # 3. Exécution séquentielle des agents d'affectation
            for ag_name in agents_assign:
                env = os.environ.copy()
                env["PREDICTOR_TYPE"] = md_name
                
                eval_script = BASE_DIR / "src" / "Fleet_Assignment_Problem" / "orchestration" / f"evaluate_{ag_name.lower()}.py"
                subprocess.run(["python", str(eval_script)], env=env, check=True)

                result_file = output_map.get(ag_name)
                if result_file and result_file.exists():
                    df_res = pd.read_csv(result_file)
                    
                    ca_rec, ca_poss, marg_rec, marg_poss, spill = calculate_metrics(df_res)
                    
                    group_key = f"{md_name}_{ag_name}"
                    results[group_key]['ca_ratio'].append(ca_rec / ca_poss if ca_poss > 0 else 0)
                    results[group_key]['margin_ratio'].append(marg_rec / marg_poss if marg_poss != 0 else 0)
                    results[group_key]['spill'].append(spill)

    # 4. Agrégation des métriques
    final_metrics = []
    for group_key, data in results.items():
        md, ag = group_key.split('_')
        final_metrics.append({
            'Modele_Demande': md,
            'Agent_Affectation': ag,
            'Taux_Recuperation_CA_Moyen': np.mean(data['ca_ratio']),
            'Taux_Marge_Nette_Moyen': np.mean(data['margin_ratio']),
            'Taux_Spill_Moyen': np.mean(data['spill'])
        })
        
    df_results = pd.DataFrame(final_metrics)
    df_results.head(2)
    output_file = BASE_DIR / "resultat_benchmark_multi_planning.csv"
    df_results.to_csv(output_file, index=False, float_format='%.4f')

if __name__ == "__main__":
    N_PLANNINGS = 100 
    execute_benchmark(N_PLANNINGS)