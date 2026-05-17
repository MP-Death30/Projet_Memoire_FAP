import pandas as pd
import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

def train_agent():
    BASE_DIR = Path(__file__).resolve().parents[3]
    SCHEDULE_FILE = BASE_DIR / "data" / "processed" / "ppo_network_state_2025.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    MODEL_DIR = BASE_DIR / "models" / "ppo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    schedule_df = pd.read_parquet(SCHEDULE_FILE)
    schedule_df['Dept Time'] = pd.to_datetime(schedule_df['Dept Time'])
    schedule_df['Arr Time'] = pd.to_datetime(schedule_df['Arr Time'])
    
    # Chronologie absolue
    t0 = schedule_df['Dept Time'].min()
    schedule_df['Dept_Time_Minutes'] = (schedule_df['Dept Time'] - t0).dt.total_seconds() / 60.0
    schedule_df['Arr_Time_Minutes'] = (schedule_df['Arr Time'] - t0).dt.total_seconds() / 60.0
    
    airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}
    
    schedule_df['Origin_Idx'] = schedule_df['From'].map(airport_to_idx)
    schedule_df['Dest_Idx'] = schedule_df['To'].map(airport_to_idx)
    schedule_df['Predicted_Demand'] = schedule_df['flight_demand'].clip(upper=180)

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    
    physical_fleet = []
    tail_id = 0
    for _, row in fleet_types_df.iterrows():
        count = inventory_map.get(row['fleet_id'], 5)
        for _ in range(count):
            ac = row.to_dict()
            ac['tail_number'] = f"AC_{tail_id}"
            physical_fleet.append(ac)
            tail_id += 1
            
    fleet_df = pd.DataFrame(physical_fleet)
    num_airports = len(airports)

    env = FAPEnv(schedule_df, fleet_df, num_airports)
    env = ActionMasker(env, mask_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Augmentation de la capacité du réseau Acteur-Critique
    policy_kwargs = dict(net_arch=[256, 256])

    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs # Ajout ici
    )

    # Allongement de la phase d'apprentissage
    model.learn(total_timesteps=1500000) 
    model.save(MODEL_DIR / "maskable_ppo_model")

if __name__ == "__main__":
    train_agent()