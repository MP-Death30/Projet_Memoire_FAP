import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv

def mask_fn(env: FAPEnv) -> np.ndarray:
    return env.action_masks()

def train_agent():
    BASE_DIR = Path(__file__).resolve().parents[3]
    SCHEDULE_FILE = BASE_DIR / "data" / "processed" / "ppo_network_state_2025.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    MODEL_DIR = BASE_DIR / "models" / "ppo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingénierie des caractéristiques du jeu de données pour l'environnement
    schedule_df = pd.read_parquet(SCHEDULE_FILE)
    schedule_df['Dept Time'] = pd.to_datetime(schedule_df['Dept Time'])
    schedule_df['Arr Time'] = pd.to_datetime(schedule_df['Arr Time'])
    schedule_df['Dept_Time_Minutes'] = schedule_df['Dept Time'].dt.hour * 60 + schedule_df['Dept Time'].dt.minute
    schedule_df['Arr_Time_Minutes'] = schedule_df['Arr Time'].dt.hour * 60 + schedule_df['Arr Time'].dt.minute
    
    airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}
    
    schedule_df['Origin_Idx'] = schedule_df['From'].map(airport_to_idx)
    schedule_df['Dest_Idx'] = schedule_df['To'].map(airport_to_idx)
    schedule_df['Predicted_Demand'] = schedule_df['flight_demand']
    schedule_df['Predicted_Demand'] = schedule_df['Predicted_Demand'].clip(upper=180) #pour correspondre à la réalité d'un vol monocouloir

    # 1. Chargement des types de flotte
    fleet_types_df = pd.read_parquet(FLEET_FILE)
    
    # 2. Définition stricte de l'inventaire physique (Ajuster les clés avec les vrais fleet_id du dataset)
    # Exemple de distribution type Transavia
    inventory_map = {
        '737': 12, 
        'A320': 8,
        'Embraer190': 8
    }
    
    # 3. Génération des appareils individuels
    physical_fleet = []
    tail_id = 0
    
    for _, row in fleet_types_df.iterrows():
        fam = row['fleet_id']
        count = inventory_map.get(fam, 5) # 5 unités par défaut si modèle inconnu
        
        for _ in range(count):
            ac = row.to_dict()
            ac['tail_number'] = f"AC_{tail_id}"
            physical_fleet.append(ac)
            tail_id += 1
            
    fleet_df = pd.DataFrame(physical_fleet)
    num_airports = len(airports)

    # 2. Instanciation et encapsulation
    env = FAPEnv(schedule_df, fleet_df, num_airports)
    env = ActionMasker(env, mask_fn)

    # 3. Forçage matériel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware assigné pour l'entraînement : {device.upper()}")

    # 4. Architecture MaskablePPO
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )

    # 5. Exécution de la boucle d'apprentissage
    model.learn(total_timesteps=150000)

    # 6. Persistance
    model.save(MODEL_DIR / "maskable_ppo_model")

if __name__ == "__main__":
    train_agent()