import torch
import pandas as pd
from pathlib import Path

from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.orchestration.mappo_trainer import MAPPOTrainer, collect_trajectories
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule
from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_from_precomputed

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

def run_training():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    DEMAND_FILE = BASE_DIR / "data" / "processed" / "precomputed_demand_2025.parquet"
    MODEL_DIR = BASE_DIR / "models" / "mappo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not DEMAND_FILE.exists():
        raise FileNotFoundError("Matrice manquante. Exécuter precompute_demand.py d'abord.")

    df_precomputed = pd.read_parquet(DEMAND_FILE)
    fleet_types_df = pd.read_parquet(FLEET_FILE)

    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    physical_fleet = []
    for _, row in fleet_types_df.iterrows():
        count = inventory_map.get(row['fleet_id'], 5)
        for _ in range(count):
            physical_fleet.append({
                'position': 0.0, 'capacity': float(row['capacity']), 
                'speed': float(row['speed_kmh']), 'cost': float(row['cost'])
            })

    trainer = MAPPOTrainer(flight_dim=6, agent_dim=5, embed_dim=128)

    epochs = 1000
    for epoch in range(epochs):
        raw_schedule = generate_dynamic_schedule(len(physical_fleet))
        schedule_df = predict_demand_from_precomputed(raw_schedule, df_precomputed)
        
        airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
        num_airports = len(airports)
        max_flights = len(schedule_df)
        
        env = FAPParallelEnv(num_airports=num_airports, max_flights=max_flights)
        
        rollouts = collect_trajectories(env, trainer, physical_fleet, schedule_df, steps=max_flights)
        trainer.train_step(rollouts)
        
        if epoch % 10 == 0:
            avg_reward = torch.stack([r['rewards'] for r in rollouts]).mean().item()
            print(f"Epoch {epoch} | Récompense moyenne : {avg_reward:.2f} | Vols: {max_flights}")

    torch.save(trainer.policy.state_dict(), MODEL_DIR / "mappo_policy.pth")
    print(f"Modèle sauvegardé : {MODEL_DIR / 'mappo_policy.pth'}")

if __name__ == "__main__":
    run_training()