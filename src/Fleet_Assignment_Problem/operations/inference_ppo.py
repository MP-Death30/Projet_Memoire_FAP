import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
import joblib

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "lstm_multi_input.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler_target.pkl"
DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
VACANCES_FILE = BASE_DIR / "data" / "raw" / "Evenement_pays" / "vacances_consolidees.parquet"
EVENTS_FILE = BASE_DIR / "data" / "raw" / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
SCHEDULE_FILE = BASE_DIR / "data" / "raw" / "Flight_Schedule" / "schedule_fap.csv"

SEQ_LEN = 21
HORIZON = 7

COUNTRY_MAPPING = {
    'DE': 'Allemagne', 'EL': 'Grèce', 'GR': 'Grèce', 'ES': 'Espagne',
    'FR': 'France', 'IE': 'Irlande', 'IT': 'Italie', 'MT': 'Malte',
    'PT': 'Portugal'
}
REVERSE_MAPPING = {v: k for k, v in COUNTRY_MAPPING.items()}

TIME_OF_DAY_WEIGHTS = {
    'morning': 0.45,
    'midday': 0.20,
    'evening': 0.35
}

def get_time_bank(hour):
    if 6 <= hour < 11: return 'morning'
    elif 11 <= hour < 16: return 'midday'
    else: return 'evening'

def build_future_context(target_dates, pays_dep_code, pays_arr_code):
    df_future = pd.DataFrame({'date': pd.to_datetime(target_dates)})
    df_future['jour_semaine'] = df_future['date'].dt.dayofweek
    df_future['mois'] = df_future['date'].dt.month
    
    df_future['jour_sin'] = np.sin(2 * np.pi * df_future['jour_semaine'] / 7.0)
    df_future['jour_cos'] = np.cos(2 * np.pi * df_future['jour_semaine'] / 7.0)
    df_future['mois_sin'] = np.sin(2 * np.pi * df_future['mois'] / 12.0)
    df_future['mois_cos'] = np.cos(2 * np.pi * df_future['mois'] / 12.0)
    
    pays_dep_nom = COUNTRY_MAPPING.get(pays_dep_code, pays_dep_code)
    pays_arr_nom = COUNTRY_MAPPING.get(pays_arr_code, pays_arr_code)
    
    df_holidays = pd.read_parquet(VACANCES_FILE)
    
    df_events = pd.read_csv(EVENTS_FILE, sep=";")
    df_events['date_debut'] = pd.to_datetime(df_events['date_debut'], format='%d/%m/%Y')
    df_events['date_fin'] = pd.to_datetime(df_events['date_fin'], format='%d/%m/%Y')
    df_events['pays'] = df_events['pays'].map(REVERSE_MAPPING).fillna(df_events['pays'])
    
    records_ev = []
    for _, row in df_events.iterrows():
        dates_ev = pd.date_range(start=row['date_debut'], end=row['date_fin'])
        for d in dates_ev:
            records_ev.append({'date': d, 'pays': row['pays'], 'is_evenement': 1})
    lookup_ev = pd.DataFrame(records_ev).drop_duplicates() if records_ev else pd.DataFrame(columns=['date', 'pays', 'is_evenement'])

    for pfx, p_code, p_nom in [('depart', pays_dep_code, pays_dep_nom), ('arrivee', pays_arr_code, pays_arr_nom)]:
        df_future['pays'] = p_nom
        df_future = df_future.merge(df_holidays, on=['date', 'pays'], how='left').rename(columns={'sum_percent': f'vacances_{pfx}'})
        df_future = df_future.drop(columns=['pays'])
        
        df_future['pays'] = p_code
        df_future = df_future.merge(lookup_ev, on=['date', 'pays'], how='left').rename(columns={'is_evenement': f'evenement_{pfx}'})
        df_future = df_future.drop(columns=['pays'])

    df_future['evenement_depart'] = df_future['evenement_depart'].fillna(0).astype(int)
    df_future['evenement_arrivee'] = df_future['evenement_arrivee'].fillna(0).astype(int)
    df_future['vacances_depart'] = df_future['vacances_depart'].fillna(0.0)
    df_future['vacances_arrivee'] = df_future['vacances_arrivee'].fillna(0.0)
    
    cols_exo = [
        'evenement_depart', 'vacances_depart', 
        'evenement_arrivee', 'vacances_arrivee', 
        'jour_sin', 'jour_cos', 'mois_sin', 'mois_cos'
    ]
    return df_future, df_future[cols_exo].values.flatten().reshape(1, -1)

def generate_ppo_demand_state(route, start_date_str="2025-01-01"):
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    df_history = pd.read_parquet(DATA_LSTM_FILE)
    df_route = df_history[df_history['route'] == route].sort_values('date').reset_index(drop=True)
    
    start_date = pd.to_datetime(start_date_str)
    history_cutoff = start_date - pd.Timedelta(days=1)
    
    history_seq = df_route[df_route['date'] <= history_cutoff].tail(SEQ_LEN)
    if len(history_seq) < SEQ_LEN:
        raise ValueError(f"Profondeur historique insuffisante. {len(history_seq)}/{SEQ_LEN} jours disponibles.")
        
    X_seq = scaler.transform(history_seq[['value_jour']]).reshape(1, SEQ_LEN, 1)
    
    parts = route.split('_')
    pays_dep_code, code_dep, pays_arr_code, code_arr = parts[0], parts[1], parts[2], parts[3]
    
    target_dates = pd.date_range(start=start_date, periods=HORIZON, freq='D')
    df_future, X_exo = build_future_context(target_dates, pays_dep_code, pays_arr_code)
    
    pred_scaled = model.predict([X_seq, X_exo], verbose=0)
    pred_values = scaler.inverse_transform(pred_scaled).flatten()
    
    df_future['route'] = route
    df_future['predicted_demand'] = np.round(pred_values).astype(int).clip(min=0)
    
    return df_future[['date', 'route', 'predicted_demand']], code_dep, code_arr

def map_demand_to_schedule(df_demand, code_dep, code_arr):
    df_sched = pd.read_csv(SCHEDULE_FILE)
    df_sched['Dept Time'] = pd.to_datetime(df_sched['Dept Time'])
    df_sched['date'] = df_sched['Dept Time'].dt.normalize()
    
    mask_route = (df_sched['From'] == code_dep) & (df_sched['To'] == code_arr)
    df_route_sched = df_sched[mask_route].copy()
    
    df_merged = df_route_sched.merge(df_demand, on='date', how='inner')
    
    df_merged['bank'] = df_merged['Dept Time'].dt.hour.apply(get_time_bank)
    df_merged['weight'] = df_merged['bank'].map(TIME_OF_DAY_WEIGHTS)
    
    daily_weight_sum = df_merged.groupby('date')['weight'].transform('sum')
    df_merged['normalized_weight'] = df_merged['weight'] / daily_weight_sum
    
    df_merged['flight_demand'] = np.round(df_merged['predicted_demand'] * df_merged['normalized_weight']).astype(int)
    
    for date, group in df_merged.groupby('date'):
        diff = df_demand.loc[df_demand['date'] == date, 'predicted_demand'].values[0] - group['flight_demand'].sum()
        if diff != 0:
            idx_max_weight = group['normalized_weight'].idxmax()
            df_merged.loc[idx_max_weight, 'flight_demand'] += diff

    return df_merged[['Flight#', 'From', 'To', 'Dept Time', 'Arr Time', 'flight_demand', 'Tarif']].sort_values('Dept Time')

if __name__ == "__main__":
    import logging
    
    logging.info("Extraction du référentiel complet des routes...")
    df_history = pd.read_parquet(DATA_LSTM_FILE)
    routes_actives = df_history['route'].unique()
    
    date_cible = "2025-01-01"
    ppo_states_global = []
    
    logging.info(f"Début de l'itération sur {len(routes_actives)} routes...")
    
    for route in routes_actives:
        try:
            df_demande, origin_icao, dest_icao = generate_ppo_demand_state(route, date_cible)
            df_ppo_state = map_demand_to_schedule(df_demande, origin_icao, dest_icao)
            
            if not df_ppo_state.empty:
                ppo_states_global.append(df_ppo_state)
                
        except Exception as e:
            # Élimination silencieuse des routes aux historiques fragmentaires (< 21 jours)
            # ou absentes du planning de vol généré
            continue
            
    if ppo_states_global:
        df_final_network_state = pd.concat(ppo_states_global, ignore_index=True)
        df_final_network_state = df_final_network_state.sort_values(by=['Dept Time'])
        
        # --- AJOUT DE LA SAUVEGARDE ---
        OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ppo_network_state_2025.parquet"
        df_final_network_state.to_parquet(OUTPUT_PATH, index=False)
        
        print(f"Résultat enregistré avec succès : {OUTPUT_PATH}")
        # ------------------------------
    else:
        logging.error("Échec : Aucune route n'a généré de matrice d'état valide.")

        