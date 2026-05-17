import torch
import pandas as pd
from pathlib import Path
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.orchestration.mappo_trainer import MAPPOTrainer, collect_trajectories

def run_training():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    MODEL_DIR = BASE_DIR / "models" / "mappo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    
    physical_fleet = []
    for _, row in fleet_types_df.iterrows():
        count = inventory_map.get(row['fleet_id'], 5)
        for _ in range(count):
            physical_fleet.append({
                'position': 0.0, 
                'capacity': float(row['capacity']), 
                'speed': float(row['speed_kmh']), 
                'cost': 5000.0 # Coût fixe par défaut
            })

    # Dimension spatiale : 50 aéroports (Ajuster selon len(airports) réel)
    env = FAPParallelEnv(num_airports=50, max_flights=200)
    trainer = MAPPOTrainer(flight_dim=6, agent_dim=5, embed_dim=128)

    epochs = 1000
    for epoch in range(epochs):
        rollouts = collect_trajectories(env, trainer, physical_fleet, steps=128)
        trainer.train_step(rollouts)
        
        if epoch % 10 == 0:
            avg_reward = torch.stack([r['rewards'] for r in rollouts]).mean().item()
            print(f"Epoch {epoch} | Récompense moyenne : {avg_reward:.2f}")

    torch.save(trainer.policy.state_dict(), MODEL_DIR / "mappo_policy.pth")
    print(f"Modèle sauvegardé : {MODEL_DIR / 'mappo_policy.pth'}")

if __name__ == "__main__":
    run_training()