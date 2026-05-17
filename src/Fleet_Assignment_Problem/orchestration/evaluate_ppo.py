import pandas as pd
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv
from src.Fleet_Assignment_Problem.orchestration.train_ppo import mask_fn

BASE_DIR = Path(__file__).resolve().parents[3]
SCHEDULE_FILE = BASE_DIR / "data" / "processed" / "ppo_network_state_2025.parquet"
FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
MODEL_PATH = BASE_DIR / "models" / "ppo_fap" / "maskable_ppo_model"

def evaluate_ppo():
    schedule_df = pd.read_parquet(SCHEDULE_FILE)
    schedule_df['Dept Time'] = pd.to_datetime(schedule_df['Dept Time'])
    schedule_df['Arr Time'] = pd.to_datetime(schedule_df['Arr Time'])
    
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

    tail_to_type = {ac['tail_number']: ac['fleet_id'] for ac in physical_fleet}
    
    env = FAPEnv(schedule_df, fleet_df, len(airports))
    env = ActionMasker(env, mask_fn)
    
    model = MaskablePPO.load(MODEL_PATH)
    
    obs, _ = env.reset()
    terminated = False
    truncated = False
    
    results = []
    step_idx = 0
    
    while not (terminated or truncated):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        flight = schedule_df.iloc[step_idx]
        is_spilled = (action == len(physical_fleet))

        if is_spilled:
            tail_num = 'UNASSIGNED'
            ac_type = 'NONE'
        else:
            # Sécurité pour éviter les index hors limites
            if action < len(physical_fleet):
                tail_num = physical_fleet[action]['tail_number']
                ac_type = tail_to_type.get(tail_num, 'UNKNOWN')
            else:
                tail_num = 'INVALID_INDEX'
                ac_type = 'ERROR'
        
        results.append({
            'Flight#': flight['Flight#'],
            'Route': f"{flight['From']}->{flight['To']}",
            'Tail_Number': tail_num,
            'Aircraft_Type': tail_to_type.get(tail_num, 'NONE'), # Ajout de la famille
            'Revenue': info.get('revenue', 0),
            'Spill_Cost': info.get('spill_cost', 0),
            'Delay_Minutes': info.get('delay_minutes', 0),
            'Reward': reward
        })
        
        step_idx += 1

    df_results = pd.DataFrame(results)
    output_path = BASE_DIR / "data" / "processed" / "evaluation_results_ppo.csv"
    df_results.to_csv(output_path, index=False, sep=';')
    
    print(f"Extraction terminée : {len(df_results)} vols traités.")
    print(f"Taux d'annulation (Unassigned) : {(df_results['Tail_Number'] == 'UNASSIGNED').mean() * 100:.2f}%")
    print(f"Fichier : {output_path}")

if __name__ == "__main__":
    evaluate_ppo()