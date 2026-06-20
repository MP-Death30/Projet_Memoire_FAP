# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import torch
import subprocess
import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.Fleet_Assignment_Problem.environments.fap_env import FAPEnv
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.models.pointer_net import MAPPOPolicy

st.set_page_config(page_title="FAP Allocation Engine", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

@st.cache_resource
def load_allocation_models(num_airports, max_flights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo_model = MaskablePPO.load(MODELS_DIR / "ppo_fap" / "maskable_ppo_model")
    
    mappo_policy = MAPPOPolicy(flight_dim=6, agent_dim=5, embed_dim=128).to(device)
    mappo_policy.load_state_dict(torch.load(MODELS_DIR / "mappo_fap" / "mappo_policy.pth", map_location=device))
    mappo_policy.eval()
    
    return ppo_model, mappo_policy, device

@st.cache_data
def load_fleet_infrastructure():
    fleet_types_df = pd.read_parquet(DATA_DIR / "fleet_data.parquet")
    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    prefix_map = {'737': 'B', 'A320': 'A', 'Embraer190': 'E'}
    
    physical_fleet = []
    tail_id = 0
    for _, row in fleet_types_df.iterrows():
        f_id = row['fleet_id']
        count = inventory_map.get(f_id, 5)
        for _ in range(count):
            ac = row.to_dict()
            ac['cost_per_flight'] = float(ac['cost'])
            ac['tail_number'] = f"AC_{tail_id}"
            ac['prefix'] = prefix_map.get(f_id, 'U')
            ac['id'] = tail_id
            
            # Alignement exhaustif : récupération de la colonne réelle ou fallback sur la constante du générateur
            ac['speed'] = float(ac.get('speed_kmh', 850.0))
            ac['available_time'] = 0.0
            
            physical_fleet.append(ac)
            tail_id += 1
    return fleet_types_df, physical_fleet

try:
    fleet_types_df, physical_fleet = load_fleet_infrastructure()
except FileNotFoundError:
    st.error("Infrastructure de flotte introuvable. Exécutez les pipelines de données au préalable.")
    st.stop()

st.sidebar.header("Configuration de la Simulation")
uploaded_file = st.sidebar.file_uploader("Importer le planning brut (CSV)", type="csv")

st.sidebar.markdown("---")
predictor_type = st.sidebar.selectbox("Modèle de Prévision de Demande", ["LSTM", "XGBOOST"])
agent_type = st.sidebar.selectbox("Algorithme d'Affectation Flotte", ["GREEDY", "PPO", "MAPPO"])
base_spill_cost = st.sidebar.slider("Pénalité de Spill (Coefficient)", 1.0, 3.0, 1.5, 0.1)

execute = st.sidebar.button("Exécuter l'Optimisation Globale")

if uploaded_file is not None and execute:
    # 1. Sauvegarde temporaire du flux brut pour le processus d'inférence
    temp_raw_path = DATA_DIR / "temp_ui_upload.csv"
    with open(temp_raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. Phase d'Inférence Isolée (Sous-processus TensorFlow/CPU)
    st.info(f"Étape 1 : Prédiction de la demande via {predictor_type} (Isolation Mémoire Activable)...")
    
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = str(BASE_DIR)
    
    try:
        process = subprocess.run([
            "python", "run_inference.py", 
            "--input", str(temp_raw_path), 
            "--model", predictor_type
        ], env=env_vars, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        st.error("Erreur critique lors de l'inférence.")
        st.code(e.stderr)
        st.stop()
        
    # 3. Récupération des données enrichies
    enriched_file_path = DATA_DIR / "enriched_schedule_ready.csv"
    if not enriched_file_path.exists():
        st.error("Le fichier d'inférence n'a pas été généré.")
        st.stop()
        
    enriched_df = pd.read_csv(enriched_file_path, sep=r'[,;]', engine='python')
    
    if 'Predicted_Demand' not in enriched_df.columns and 'flight_demand' in enriched_df.columns:
        enriched_df['Predicted_Demand'] = enriched_df['flight_demand'].clip(upper=180)

    # Nettoyage des données
    enriched_df['Dept Time'] = pd.to_datetime(enriched_df['Dept Time'])
    enriched_df['Arr Time'] = pd.to_datetime(enriched_df['Arr Time'])
    t0 = enriched_df['Dept Time'].min()
    enriched_df['Dept_Time_Minutes'] = (enriched_df['Dept Time'] - t0).dt.total_seconds() / 60.0
    enriched_df['Arr_Time_Minutes'] = (enriched_df['Arr Time'] - t0).dt.total_seconds() / 60.0
    
    airports = pd.concat([enriched_df['From'], enriched_df['To']]).unique()
    num_airports = len(airports)
    max_flights = len(enriched_df)
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}
    
    enriched_df['Origin_Idx'] = enriched_df['From'].map(airport_to_idx)
    enriched_df['Dest_Idx'] = enriched_df['To'].map(airport_to_idx)
    
    # 4. Phase d'Allocation (Processus Principal PyTorch/GPU)
    ppo_model, mappo_policy, device = load_allocation_models(num_airports, max_flights)
    
    enriched_df['Agent_ID'] = -1
    enriched_df['Aircraft_Code'] = "SPILL"
    enriched_df['Agent_Capacity'] = 0.0
    enriched_df['Agent_Cost'] = 0.0
    enriched_df['Margin_Generated'] = 0.0

    st.info(f"Étape 2 : Traitement spatial de {max_flights} vols via l'agent {agent_type}...")

    if agent_type == "GREEDY":
        local_fleet = [{**ac, 'position': float(airport_to_idx.get("LFPO", 0.0)), 'available_time': 0.0} for ac in physical_fleet]
        for idx, flight in enriched_df.sort_values('Dept_Time_Minutes').iterrows():
            best_ac_id, best_margin = -1, -float('inf')
            for ac in local_fleet:
                if ac['position'] == flight['Origin_Idx'] and ac['available_time'] <= flight['Dept_Time_Minutes']:
                    margin = (min(ac['capacity'], flight['Predicted_Demand']) * flight['Tarif']) - ac['cost']
                    if margin > best_margin:
                        best_margin, best_ac_id = margin, ac['id']
            if best_ac_id != -1:
                ac = local_fleet[best_ac_id]
                enriched_df.at[idx, 'Agent_ID'] = best_ac_id
                enriched_df.at[idx, 'Aircraft_Code'] = f"{ac['prefix']}{best_ac_id}"
                enriched_df.at[idx, 'Agent_Capacity'] = ac['capacity']
                enriched_df.at[idx, 'Agent_Cost'] = ac['cost']
                enriched_df.at[idx, 'Margin_Generated'] = best_margin
                ac['position'] = flight['Dest_Idx']
                ac['available_time'] = flight['Dept_Time_Minutes'] + (flight['Arr_Time_Minutes'] - flight['Dept_Time_Minutes']) + 50.0

    elif agent_type == "PPO":
        env = FAPEnv(enriched_df, pd.DataFrame(physical_fleet), num_airports, base_spill_cost=base_spill_cost)
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        obs, _ = env.reset()
        terminated = False
        step_idx = 0
        while not terminated:
            action, _ = ppo_model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, _, terminated, _, info = env.step(action)
            if action < len(physical_fleet):
                ac = physical_fleet[action]
                enriched_df.at[step_idx, 'Agent_ID'] = action
                enriched_df.at[step_idx, 'Aircraft_Code'] = f"{ac['prefix']}{action}"
                enriched_df.at[step_idx, 'Agent_Capacity'] = ac['capacity']
                enriched_df.at[step_idx, 'Agent_Cost'] = ac['cost_per_flight']
                enriched_df.at[step_idx, 'Margin_Generated'] = info.get('revenue', 0) - ac['cost_per_flight']
            step_idx += 1

    elif agent_type == "MAPPO":
        local_fleet = [{**ac, 'position': float(airport_to_idx.get("LFPO", 0.0))} for ac in physical_fleet]
        env = FAPParallelEnv(num_airports=num_airports, max_flights=max_flights, spill_penalty_coef=base_spill_cost)
        min_time_sec = enriched_df['Dept Time'].min().timestamp() / 60.0
        flights_data = [{
            'origin': float(airport_to_idx[r['From']]), 'dest': float(airport_to_idx[r['To']]), 
            'dep_time': (r['Dept Time'].timestamp() / 60.0) - min_time_sec, 
            'arr_time': (r['Arr Time'].timestamp() / 60.0) - min_time_sec, 
            'pax': float(r['Predicted_Demand']), 'fare': float(r['Tarif'])
        } for _, r in enriched_df.iterrows()]
        
        (obs_a, obs_f, pad_mask), masks = env.reset(local_fleet, flights_data)
        done = False
        while not done:
            obs_a, obs_f, pad_mask, masks = obs_a.to(device), obs_f.to(device), pad_mask.to(device), masks.to(device)
            with torch.no_grad():
                flight_emb = mappo_policy.flight_encoder(obs_f.unsqueeze(0), pad_mask=pad_mask.unsqueeze(0))
                logits = torch.stack([mappo_policy.actor(obs_a[i].unsqueeze(0), flight_emb, masks[i].unsqueeze(0)) for i in range(obs_a.size(0))], dim=1)
                actions = torch.argmax(logits, dim=-1).squeeze(0)
            obs_a, obs_f, pad_mask, masks, _, done = env.step(actions)
            
        for rec in env.assignment_history:
            f_idx, a_idx = rec['flight_index'], rec['agent_index']
            if a_idx != -1:
                ac = physical_fleet[a_idx]
                enriched_df.at[f_idx, 'Agent_ID'] = a_idx
                enriched_df.at[f_idx, 'Aircraft_Code'] = f"{ac['prefix']}{a_idx}"
                enriched_df.at[f_idx, 'Agent_Capacity'], enriched_df.at[f_idx, 'Agent_Cost'], enriched_df.at[f_idx, 'Margin_Generated'] = ac['capacity'], ac['cost'], rec['margin']

    # 5. Consolidation et Restitution
    unmet_pax = (enriched_df['Predicted_Demand'] - enriched_df['Agent_Capacity']).clip(lower=0)
    enriched_df['Spill_Cost'] = unmet_pax * enriched_df['Tarif'] * base_spill_cost
    unassigned_mask = enriched_df['Agent_ID'] == -1
    enriched_df.loc[unassigned_mask, 'Margin_Generated'] = -enriched_df.loc[unassigned_mask, 'Spill_Cost']

    st.success("Calculs d'optimisation terminés.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Marge Nette", f"{enriched_df['Margin_Generated'].sum():,.2f} €")
    c2.metric("Coût d'Opportunité (Spill)", f"{enriched_df['Spill_Cost'].sum():,.2f} €")
    c3.metric("Rupture Réseau", f"{unassigned_mask.sum()} / {max_flights}")
    
    df_gantt = enriched_df[~unassigned_mask].copy()
    if not df_gantt.empty:
        fig = px.timeline(df_gantt, x_start="Dept Time", x_end="Arr Time", y="Aircraft_Code", color="To", title="Ordonnancement Flotte")
        fig.update_yaxes(categoryorder="category ascending")
        st.plotly_chart(fig, use_container_width=True)
        
    enriched_df['Delta'] = enriched_df['Predicted_Demand'] - enriched_df['Agent_Capacity']
    enriched_df['Statut'] = enriched_df.apply(lambda r: "CRITIQUE (Spill Intégral)" if r['Agent_ID'] == -1 else ("SOUS-CAPACITÉ" if r['Delta'] > 20 else ("SURCAPACITÉ" if r['Delta'] < -40 else "ADÉQUAT")), axis=1)
    
    st.dataframe(enriched_df[enriched_df['Statut'] != "ADÉQUAT"][['Dept Time', 'Flight#', 'From', 'To', 'Aircraft_Code', 'Predicted_Demand', 'Agent_Capacity', 'Delta', 'Statut']].sort_values('Dept Time'), use_container_width=True)

elif not execute:
    st.warning("En attente de configuration.")