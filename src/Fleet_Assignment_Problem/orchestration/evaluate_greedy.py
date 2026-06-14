import pandas as pd
from pathlib import Path
import json

def evaluate_greedy():
    BASE_DIR = Path(__file__).resolve().parents[3]
    SCHEDULE_FILE = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    
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
        for _ in range(inventory_map.get(f_id, 5)):
            physical_fleet.append({
                'id': tail_id,
                'position': float(airport_to_idx.get("LFPO", 0.0)),
                'available_time': 0.0,
                'capacity': float(row['capacity']),
                'cost': float(row['cost']),
                'prefix': prefix
            })
            tail_id += 1

    schedule_df['Agent_ID'] = -1
    schedule_df['Aircraft_Code'] = "SPILL"
    schedule_df['Agent_Capacity'] = 0.0
    schedule_df['Agent_Cost'] = 0.0
    schedule_df['Margin_Generated'] = 0.0
    schedule_df['Spill_Cost'] = 0.0
    
    schedule_df = schedule_df.sort_values('Dept_Time_Minutes')
    
    for idx, flight in schedule_df.iterrows():
        best_ac_id = -1
        best_margin = -float('inf')
        
        for ac in physical_fleet:
            if ac['position'] == flight['Origin_Idx'] and ac['available_time'] <= flight['Dept_Time_Minutes']:
                pax = min(ac['capacity'], flight['Predicted_Demand'])
                revenue = pax * flight['Tarif']
                margin = revenue - ac['cost']
                
                if margin > best_margin:
                    best_margin = margin
                    best_ac_id = ac['id']
                    
        if best_ac_id != -1:
            ac = physical_fleet[best_ac_id]
            schedule_df.at[idx, 'Agent_ID'] = best_ac_id
            schedule_df.at[idx, 'Aircraft_Code'] = f"{ac['prefix']}{best_ac_id}"
            schedule_df.at[idx, 'Agent_Capacity'] = ac['capacity']
            schedule_df.at[idx, 'Agent_Cost'] = ac['cost']
            schedule_df.at[idx, 'Margin_Generated'] = best_margin
            
            ac['position'] = flight['Dest_Idx']
            flight_duration = flight['Arr_Time_Minutes'] - flight['Dept_Time_Minutes']
            ac['available_time'] = flight['Dept_Time_Minutes'] + flight_duration + 50.0

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
        
    cols_to_drop = ['Origin_Idx', 'Dest_Idx', 'Dept_Time_Minutes', 'Arr_Time_Minutes']
    export_df = schedule_df.drop(columns=[c for c in cols_to_drop if c in schedule_df.columns])
    
    CSV_PATH = BASE_DIR / "data" / "processed" / "greedy_allocations.csv"
    export_df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    evaluate_greedy()