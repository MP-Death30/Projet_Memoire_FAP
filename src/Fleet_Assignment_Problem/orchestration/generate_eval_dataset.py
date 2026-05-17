import pandas as pd
from pathlib import Path
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule
from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule

def generate_evaluation_dataset():
    BASE_DIR = Path(__file__).resolve().parents[3]
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    
    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    num_aircraft = sum(inventory_map.get(row['fleet_id'], 5) for _, row in fleet_types_df.iterrows())
    
    print("Génération du planning stochastique de base...")
    raw_schedule = generate_dynamic_schedule(num_aircraft)
    
    print("Enrichissement LSTM (Pax Inference)...")
    enriched_schedule = predict_demand_for_schedule(raw_schedule, start_date_str="2025-01-01")
    
    enriched_schedule.to_parquet(OUTPUT_PATH, index=False)
    print(f"Dataset d'évaluation figé et généré : {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_evaluation_dataset()