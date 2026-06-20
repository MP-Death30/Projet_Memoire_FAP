import torch
import pandas as pd
from pathlib import Path
import json
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.models.pointer_net import MAPPOPolicy

def run_evaluation():
    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    EVAL_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    MODEL_PATH = BASE_DIR / "models" / "mappo_fap" / "mappo_policy.pth"

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    prefix_map = {'737': 'B', 'A320': 'A', 'Embraer190': 'E'}
    
    physical_fleet = []
    for _, row in fleet_types_df.iterrows():
        f_id = row['fleet_id']
        prefix = prefix_map.get(f_id, 'U')
        for _ in range(inventory_map.get(f_id, 5)):
            physical_fleet.append({
                'position': 0.0, 
                'capacity': float(row['capacity']), 
                'speed': float(row['speed_kmh']), 
                'cost': float(row['cost']),
                'prefix': prefix
            })

    schedule_df = pd.read_parquet(EVAL_FILE)
    
    if 'flight_demand' not in schedule_df.columns:
        schedule_df['flight_demand'] = 150.0
    schedule_df['Predicted_Demand'] = schedule_df['flight_demand'].clip(upper=180)

    airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
    num_airports = len(airports)
    max_flights = len(schedule_df)
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}

    for ac in physical_fleet:
        ac['position'] = float(airport_to_idx.get("LFPO", 0.0))

    env = FAPParallelEnv(num_airports=num_airports, max_flights=max_flights)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = MAPPOPolicy(flight_dim=6, agent_dim=5, embed_dim=128).to(device)
    policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy.eval()

    # CORRECTION : Suppression de la variable min_time. 
    # Les tenseurs d'entrée sont désormais transmis en timestamps absolus, 
    # symétriquement à la fonction collect_trajectories de mappo_trainer.py
    flights_data = [{
        'origin': float(airport_to_idx[r['From']]), 
        'dest': float(airport_to_idx[r['To']]), 
        'dep_time': r['Dept Time'].timestamp() / 60.0, 
        'arr_time': r['Arr Time'].timestamp() / 60.0, 
        'pax': float(r['Predicted_Demand']), 
        'fare': float(r['Tarif'])
    } for _, r in schedule_df.iterrows()]

    (obs_a, obs_f, pad_mask), masks = env.reset(physical_fleet, flights_data)
    
    obs_a = obs_a.to(device)
    obs_f = obs_f.to(device)
    pad_mask = pad_mask.to(device)
    masks = masks.to(device)
    
    done = False
    step_count = 0

    with torch.no_grad():
        while not done:
            flight_emb = policy.flight_encoder(obs_f.unsqueeze(0), pad_mask=pad_mask.unsqueeze(0))
            logits = []
            for i in range(obs_a.size(0)):
                logits.append(policy.actor(obs_a[i].unsqueeze(0), flight_emb, masks[i].unsqueeze(0)))
            logits = torch.stack(logits, dim=1)
            
            actions = torch.argmax(logits, dim=-1).squeeze(0)
            
            obs_a, obs_f, pad_mask, masks, rewards, done = env.step(actions)
            
            if not done:
                obs_a = obs_a.to(device)
                obs_f = obs_f.to(device)
                pad_mask = pad_mask.to(device)
                masks = masks.to(device)

            step_count += 1
            if step_count > max_flights * 3:
                break

    schedule_df['Agent_ID'] = -1
    schedule_df['Aircraft_Code'] = "SPILL"
    schedule_df['Agent_Capacity'] = 0.0
    schedule_df['Agent_Cost'] = 0.0
    schedule_df['Margin_Generated'] = 0.0
    schedule_df['Spill_Cost'] = 0.0

    for record in env.assignment_history:
        f_idx = record['flight_index']
        a_idx = record['agent_index']
        
        # Sécurité : On ignore l'agent virtuel du spill (-1) pour l'extraction de la flotte physique
        if a_idx != -1:
            schedule_df.at[f_idx, 'Agent_ID'] = a_idx
            schedule_df.at[f_idx, 'Aircraft_Code'] = f"{physical_fleet[a_idx]['prefix']}{a_idx}"
            schedule_df.at[f_idx, 'Agent_Capacity'] = physical_fleet[a_idx]['capacity']
            schedule_df.at[f_idx, 'Agent_Cost'] = physical_fleet[a_idx]['cost']
            schedule_df.at[f_idx, 'Margin_Generated'] = record['margin']

    unmet_pax = (schedule_df['Predicted_Demand'] - schedule_df['Agent_Capacity']).clip(lower=0)
    schedule_df['Spill_Cost'] = unmet_pax * schedule_df['Tarif']

    unassigned_mask = schedule_df['Agent_ID'] == -1
    schedule_df.loc[unassigned_mask, 'Margin_Generated'] = -schedule_df.loc[unassigned_mask, 'Spill_Cost']

    marge_totale = schedule_df['Margin_Generated'].sum()
    taux_spill = (schedule_df['Agent_ID'] == -1).mean() * 100

    metrics = {
        "Margin_Generated": float(marge_totale),
        "Spill_Rate": float(taux_spill)
    }
    
    METRICS_FILE = BASE_DIR / "data" / "processed" / "temp_metrics.json"
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

    CSV_PATH = BASE_DIR / "data" / "processed" / "mappo_allocations.csv"
    schedule_df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    run_evaluation()