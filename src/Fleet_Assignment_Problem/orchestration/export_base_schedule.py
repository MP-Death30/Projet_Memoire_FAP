import pandas as pd
from pathlib import Path
import random
import numpy as np
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule

def export_base():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "base_schedule_unpredicted.csv"
    
    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    num_aircraft = sum(inventory_map.get(row['fleet_id'], 5) for _, row in fleet_types_df.iterrows())
    
    random.seed(42)
    np.random.seed(42)
    raw_schedule = generate_dynamic_schedule(num_aircraft)
    
    raw_schedule.to_csv(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    export_base()