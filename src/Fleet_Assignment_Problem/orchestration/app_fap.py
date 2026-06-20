import streamlit as st
import pandas as pd
from pathlib import Path
import os
import subprocess
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Interface FAP - Inférence et Affectation")

BASE_DIR = Path(__file__).resolve().parents[3]

st.sidebar.header("Paramètres")
uploaded_file = st.sidebar.file_uploader("1. Importer planning brut (.csv)", type=["csv"])
predictor_type = st.sidebar.selectbox("2. Modèle de prévision", ["LSTM", "XGBOOST"])
agent_type = st.sidebar.selectbox("3. Algorithme d'affectation", ["GREEDY", "PPO", "MAPPO"])

if st.sidebar.button("Exécuter"):
    if not uploaded_file:
        st.error("Fichier d'entrée requis.")
        st.stop()

    # 1. Ingestion résiliente
    raw_schedule = pd.read_csv(uploaded_file)
    if 'Dept Time' not in raw_schedule.columns:
        uploaded_file.seek(0)
        raw_schedule = pd.read_csv(uploaded_file, sep=';')
        
    if 'Dept Time' not in raw_schedule.columns:
        st.error("Échec d'intégrité de la matrice. Le fichier soumis ne respecte pas le schéma du planning.")
        st.stop()

    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    EVAL_TARGET = BASE_DIR / "data" / "processed" / "eval_schedule_fap.parquet"

    df_history = pd.read_parquet(DATA_LSTM_FILE)

    # 2. Inférence de la demande
    st.text("1/3 : Calcul des probabilités de demande...")
    if predictor_type == "LSTM":
        import tensorflow as tf
        import joblib
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule
        lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
        lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
        enriched_schedule = predict_demand_for_schedule(raw_schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
    else:
        from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_xgboost
        XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_demand_model.json"
        MAPPING_PATH = BASE_DIR / "models" / "xgb_airport_mapping.pkl"
        enriched_schedule = predict_demand_xgboost(raw_schedule, XGB_MODEL_PATH, MAPPING_PATH, df_history)

    enriched_schedule.to_parquet(EVAL_TARGET, index=False)

    # 3. Exécution de l'algorithme d'affectation
    st.text(f"2/3 : Exécution de l'agent {agent_type}...")
    env = os.environ.copy()
    env["PREDICTOR_TYPE"] = predictor_type
    
    eval_script = BASE_DIR / "src" / "Fleet_Assignment_Problem" / "orchestration" / f"evaluate_{agent_type.lower()}.py"
    subprocess.run(["python", str(eval_script)], env=env, check=True)

    # 4. Restitution Visuelle Native (Remplacement de visualize_horizon)
    st.text("3/3 : Génération du tableau de bord...")
    output_map = {
        "GREEDY": BASE_DIR / "data" / "processed" / "greedy_allocations.csv",
        "PPO": BASE_DIR / "data" / "processed" / "ppo_allocations.csv",
        "MAPPO": BASE_DIR / "data" / "processed" / "mappo_allocations.csv"
    }
    
    result_file = output_map.get(agent_type)
    if result_file and result_file.exists():
        enriched_df = pd.read_csv(result_file)
        
        # Consolidation des métriques
        if 'Spill_Cost' not in enriched_df.columns:
            unmet_pax = (enriched_df['Predicted_Demand'] - enriched_df['Agent_Capacity']).clip(lower=0)
            base_spill_cost = 1.0  # Multiplicateur de pénalité par défaut
            enriched_df['Spill_Cost'] = unmet_pax * enriched_df.get('Tarif', 100) * base_spill_cost
            
        unassigned_mask = enriched_df['Agent_ID'] == -1
        max_flights = len(enriched_df)
        
        st.success("Calculs d'optimisation terminés.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Marge Nette", f"{enriched_df['Margin_Generated'].sum():,.2f} €")
        c2.metric("Coût d'Opportunité (Spill)", f"{enriched_df['Spill_Cost'].sum():,.2f} €")
        c3.metric("Rupture Réseau", f"{unassigned_mask.sum()} / {max_flights}")
        
        # Graphique de Gantt
        df_gantt = enriched_df[~unassigned_mask].copy()
        if not df_gantt.empty:
            df_gantt['Dept Time'] = pd.to_datetime(df_gantt['Dept Time'])
            df_gantt['Arr Time'] = pd.to_datetime(df_gantt['Arr Time'])
            fig = px.timeline(df_gantt, x_start="Dept Time", x_end="Arr Time", y="Aircraft_Code", color="To", title=f"Ordonnancement Flotte ({agent_type})")
            fig.update_yaxes(categoryorder="category ascending")
            st.plotly_chart(fig, use_container_width=True)
            
        # Détection des anomalies
        enriched_df['Delta'] = enriched_df['Predicted_Demand'] - enriched_df['Agent_Capacity']
        enriched_df['Statut'] = enriched_df.apply(lambda r: "CRITIQUE (Spill Intégral)" if r['Agent_ID'] == -1 else ("SOUS-CAPACITÉ" if r['Delta'] > 20 else ("SURCAPACITÉ" if r['Delta'] < -40 else "ADÉQUAT")), axis=1)
        
        cols_to_display = ['Dept Time', 'From', 'To', 'Aircraft_Code', 'Predicted_Demand', 'Agent_Capacity', 'Delta', 'Statut']
        if 'Flight#' in enriched_df.columns:
            cols_to_display.insert(1, 'Flight#')
            
        st.dataframe(enriched_df[enriched_df['Statut'] != "ADÉQUAT"][cols_to_display].sort_values('Dept Time'), use_container_width=True)