import pandas as pd
from pathlib import Path
import json
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv
from src.Fleet_Assignment_Problem.models.train_ppo import mask_fn

BASE_DIR = Path(__file__).resolve().parents[3]
SCHEDULE_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
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
    
    if 'flight_demand' not in schedule_df.columns:
        schedule_df['flight_demand'] = 150.0
    schedule_df['Predicted_Demand'] = schedule_df['flight_demand'].clip(upper=180)

    fleet_types_df = pd.read_parquet(FLEET_FILE)
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    prefix_map = {'737': 'B', 'A320': 'A', 'Embraer190': 'E'}
    
    physical_fleet = []
    tail_id = 0

    for _, row in fleet_types_df.iterrows():
        f_id = row['fleet_id']
        prefix = prefix_map.get(f_id, 'U')
        count = inventory_map.get(f_id, 5)
        for _ in range(count):
            ac = row.to_dict()
            if 'cost' in ac and 'cost_per_flight' not in ac:
                ac['cost_per_flight'] = float(ac['cost'])
            ac['tail_number'] = f"AC_{tail_id}"
            ac['prefix'] = prefix
            physical_fleet.append(ac)
            tail_id += 1
            
    fleet_df = pd.DataFrame(physical_fleet)

    env = FAPEnv(schedule_df, fleet_df, len(airports))
    env = ActionMasker(env, mask_fn)
    
    model = MaskablePPO.load(MODEL_PATH)
    
    obs, _ = env.reset()
    terminated = False
    truncated = False
    
    schedule_df['Agent_ID'] = -1
    schedule_df['Aircraft_Code'] = "SPILL"
    schedule_df['Agent_Capacity'] = 0.0
    schedule_df['Agent_Cost'] = 0.0
    schedule_df['Margin_Generated'] = 0.0
    schedule_df['Spill_Cost'] = 0.0
    
    step_idx = 0
    
    while not (terminated or truncated):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        is_spilled = (action == len(physical_fleet))

        if not is_spilled and action < len(physical_fleet):
            ac = physical_fleet[action]
            schedule_df.at[step_idx, 'Agent_ID'] = action
            schedule_df.at[step_idx, 'Aircraft_Code'] = f"{ac['prefix']}{action}"
            schedule_df.at[step_idx, 'Agent_Capacity'] = float(ac['capacity'])
            
            cout_vol = float(ac.get('cost', ac.get('cost_per_flight', 5000)))
            schedule_df.at[step_idx, 'Agent_Cost'] = cout_vol
            
            margin = info.get('revenue', 0) - cout_vol
            schedule_df.at[step_idx, 'Margin_Generated'] = margin

        step_idx += 1

    unmet_pax = (schedule_df['Predicted_Demand'] - schedule_df['Agent_Capacity']).clip(lower=0)
    schedule_df['Spill_Cost'] = unmet_pax * schedule_df['Tarif']

    unassigned_mask = schedule_df['Agent_ID'] == -1
    schedule_df.loc[unassigned_mask, 'Margin_Generated'] = -schedule_df.loc[unassigned_mask, 'Spill_Cost']

    cols_to_drop = ['Origin_Idx', 'Dest_Idx', 'Dept_Time_Minutes', 'Arr_Time_Minutes']
    export_df = schedule_df.drop(columns=[c for c in cols_to_drop if c in schedule_df.columns])

    marge_totale = export_df['Margin_Generated'].sum()
    taux_spill = (export_df['Agent_ID'] == -1).mean() * 100

    metrics = {
        "Margin_Generated": float(marge_totale),
        "Spill_Rate": float(taux_spill)
    }
    
    METRICS_FILE = BASE_DIR / "data" / "processed" / "temp_metrics.json"
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

    CSV_PATH = BASE_DIR / "data" / "processed" / "ppo_allocations.csv"
    export_df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    evaluate_ppo()