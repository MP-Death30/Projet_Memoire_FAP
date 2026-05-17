import torch
import pandas as pd
from pathlib import Path
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.models.pointer_net import MAPPOPolicy

def run_evaluation():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    EVAL_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    MODEL_PATH = BASE_DIR / "models" / "mappo_fap" / "mappo_policy.pth"

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    physical_fleet = []
    for _, row in fleet_types_df.iterrows():
        for _ in range(inventory_map.get(row['fleet_id'], 5)):
            physical_fleet.append({
                'position': 0.0, 
                'capacity': float(row['capacity']), 
                'speed': float(row['speed_kmh']), 
                'cost': float(row['cost'])
            })

    schedule_df = pd.read_parquet(EVAL_FILE)
    
    airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
    num_airports = len(airports)
    max_flights = len(schedule_df)
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}

    for ac in physical_fleet:
        ac['position'] = float(airport_to_idx.get("LFPO", 0.0))

    env = FAPParallelEnv(num_airports=num_airports, max_flights=max_flights)
    
    policy = MAPPOPolicy(flight_dim=6, agent_dim=5, embed_dim=128)
    policy.load_state_dict(torch.load(MODEL_PATH))
    policy.eval()

    flights_data = [{
        'origin': float(airport_to_idx[r['From']]), 
        'dest': float(airport_to_idx[r['To']]), 
        'dep_time': r['Dept Time'].timestamp() / 60.0, 
        'arr_time': r['Arr Time'].timestamp() / 60.0, 
        'pax': float(r.get('flight_demand', 150)), 
        'fare': float(r['Tarif'])
    } for _, r in schedule_df.iterrows()]

    (obs_a, obs_f, pad_mask), masks = env.reset(physical_fleet, flights_data)
    
    done = False
    total_rewards = 0

    with torch.no_grad():
        while not done:
            flight_emb = policy.flight_encoder(obs_f.unsqueeze(0), pad_mask=pad_mask.unsqueeze(0))
            logits = []
            for i in range(obs_a.size(0)):
                logits.append(policy.actor(obs_a[i].unsqueeze(0), flight_emb, masks[i].unsqueeze(0)))
            logits = torch.stack(logits, dim=1)
            
            actions = torch.argmax(logits, dim=-1).squeeze(0)
            
            obs_a, obs_f, pad_mask, masks, rewards, done = env.step(actions)
            total_rewards += rewards.sum().item()

    print(f"Évaluation terminale. Marge opérationnelle brute : {total_rewards:.2f}")

if __name__ == "__main__":
    run_evaluation()