from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
MAIN_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2023_2024.parquet"
VACANCES_FILE = DATA_RAW / "Evenement_pays" / "vacances_consolidees.parquet"
EVENTS_FILE = DATA_RAW / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"

COUNTRY_MAPPING = {
    'DE': 'Allemagne', 'EL': 'Grèce', 'GR': 'Grèce', 'ES': 'Espagne',
    'FR': 'France', 'IE': 'Irlande', 'IT': 'Italie', 'MT': 'Malte',
    'PT': 'Portugal'
}
REVERSE_MAPPING = {v: k for k, v in COUNTRY_MAPPING.items()}

def create_daily_lookup(df, flag_name):
    records = []
    for _, row in df.iterrows():
        dates = pd.date_range(start=row['date_debut'], end=row['date_fin'])
        for d in dates:
            records.append({'date': d, 'pays': row['pays'], flag_name: 1})
    return pd.DataFrame(records).drop_duplicates()

def generate_dataset():
    df_main = pd.read_parquet(MAIN_FILE)
    df_events = pd.read_csv(EVENTS_FILE, sep=";")
    
    # Remplacement de l'import : le fichier est déjà granulaire et formaté
    df_holidays = pd.read_parquet(VACANCES_FILE)

    df_main['date'] = pd.to_datetime(df_main['date'])
    df_events['date_debut'] = pd.to_datetime(df_events['date_debut'], format='%d/%m/%Y')
    df_events['date_fin'] = pd.to_datetime(df_events['date_fin'], format='%d/%m/%Y')

    df_events['pays'] = df_events['pays'].map(REVERSE_MAPPING).fillna(df_events['pays'])
    
    # Conservation de la fonction lookup uniquement pour les évènements isolés
    lookup_ev = create_daily_lookup(df_events, 'is_evenement')

    for pfx in ['depart', 'arrivee']:
        # Jointure évènements (Binaire 0/1)
        df_main = df_main.merge(
            lookup_ev.rename(columns={'pays': f'pays_{pfx}', 'is_evenement': f'evenement_{pfx}'}),
            on=['date', f'pays_{pfx}'], how='left'
        )
        # Jointure pondération vacances (Continu 0.0 - 1.0)
        df_main = df_main.merge(
            df_holidays.rename(columns={'pays': f'pays_{pfx}', 'sum_percent': f'vacances_{pfx}'}),
            on=['date', f'pays_{pfx}'], how='left'
        )

    # Résolution des NaN par typage strict
    df_main['evenement_depart'] = df_main['evenement_depart'].fillna(0).astype(int)
    df_main['evenement_arrivee'] = df_main['evenement_arrivee'].fillna(0).astype(int)
    df_main['vacances_depart'] = df_main['vacances_depart'].fillna(0.0)
    df_main['vacances_arrivee'] = df_main['vacances_arrivee'].fillna(0.0)

    # Injection du bruit stochastique
    noise = np.random.normal(loc=1.0, scale=0.08, size=len(df_main))
    df_main['value_jour'] = (df_main['value_jour'] * noise).round().astype(int)
    df_main['value_jour'] = df_main['value_jour'].clip(lower=0)

    to_drop = ['period', 'value_mensuelle', 'norm_coeff', 
               'pays_depart', 'aeroport_depart', 'pays_arrivee', 'aeroport_arrivee']
    df_main = df_main.drop(columns=to_drop, errors='ignore')

    # Variables cycliques
    df_main['jour_semaine'] = df_main['date'].dt.dayofweek
    df_main['mois'] = df_main['date'].dt.month

    # Préservation de la séquence temporelle (Critique LSTM)
    df_main = df_main.sort_values(['route', 'date']).reset_index(drop=True)

    df_main.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    generate_dataset()