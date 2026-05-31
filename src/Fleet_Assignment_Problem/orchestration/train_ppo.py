import pandas as pd
import numpy as np
import random
from pathlib import Path
import joblib
import gymnasium as gym

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback

from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.action_masks()

class PoolScheduleCallback(BaseCallback):
    def __init__(self, env, schedule_pool, verbose=0):
        super().__init__(verbose)
        self.custom_env = env.unwrapped
        self.schedule_pool = schedule_pool

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.custom_env.schedule_df = random.choice(self.schedule_pool)
        return True

def train_agent():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    POOL_FILE = BASE_DIR / "data" / "processed" / "ppo_schedule_pool.pkl"
    MODEL_DIR = BASE_DIR / "models" / "ppo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- PHASE 2/2 : MOTEUR PYTORCH (GPU - Apprentissage PPO) ---")
    if not POOL_FILE.exists():
        raise FileNotFoundError(f"Exécuter d'abord generate_pool.py. Fichier manquant : {POOL_FILE}")

    schedule_pool = joblib.load(POOL_FILE)
    print(f"Pool de {len(schedule_pool)} scénarios chargé en RAM.")

    EVAL_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    eval_df = pd.read_parquet(EVAL_FILE)
    all_airports = pd.concat([eval_df['From'], eval_df['To']]).unique()
    num_airports = len(all_airports)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Acquisition matérielle réussie. Entraînement sur : {device}")

    env = FAPEnv(schedule_pool[0], fleet_df, num_airports)
    env = ActionMasker(env, mask_fn)

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
        policy_kwargs=policy_kwargs
    )

    dynamic_callback = PoolScheduleCallback(env, schedule_pool, verbose=0)

    model.learn(total_timesteps=1500000, callback=dynamic_callback) 
    
    model.save(MODEL_DIR / "maskable_ppo_model")
    print(f"Modèle sauvegardé : {MODEL_DIR}")

if __name__ == "__main__":
    train_agent()