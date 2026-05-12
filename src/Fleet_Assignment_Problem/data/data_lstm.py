from pathlib import Path
import pandas as pd
import numpy as np

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_RAW = BASE_DIR / "data" / "raw"
MAIN_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2023_2024.parquet"
VACANCES_FILE = DATA_RAW / "Evenement_pays" / "Vacances_scolaire_2024_2027.csv"
EVENTS_FILE = DATA_RAW / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"

# Standardisation des pays
COUNTRY_MAPPING = {
    'DE': 'Allemagne', 'DK': 'Danemark', 'EL': 'Grèce', 'ES': 'Espagne',
    'FR': 'France', 'IE': 'Irlande', 'IT': 'Italie', 'MT': 'Malte',
    'PT': 'Portugal', 'SE': 'Suède',
}
REVERSE_MAPPING = {v: k for k, v in COUNTRY_MAPPING.items()}

def create_daily_lookup(df, flag_name):
    """Transforme des plages de dates en index journalier par pays."""
    records = []
    for _, row in df.iterrows():
        dates = pd.date_range(start=row['date_debut'], end=row['date_fin'])
        for d in dates:
            records.append({'date': d, 'pays': row['pays'], flag_name: 1})
    return pd.DataFrame(records).drop_duplicates()

# Pipeline de transformation
def generate_dataset():
    # Chargement
    df_main = pd.read_parquet(MAIN_FILE)
    df_events = pd.read_csv(EVENTS_FILE, sep=";")
    df_holidays = pd.read_csv(VACANCES_FILE, sep=";")

    # Dates
    df_main['date'] = pd.to_datetime(df_main['date'])
    for df in [df_events, df_holidays]:
        df['date_debut'] = pd.to_datetime(df['date_debut'], format='%d/%m/%Y')
        df['date_fin'] = pd.to_datetime(df['date_fin'], format='%d/%m/%Y')

    # Mapping pays (Events utilise les noms longs)
    df_events['pays'] = df_events['pays'].map(REVERSE_MAPPING).fillna(df_events['pays'])

    # Tables de recherche
    lookup_ev = create_daily_lookup(df_events, 'is_evenement')
    lookup_ho = create_daily_lookup(df_holidays, 'is_vacances')

    # Enrichissement par jointure
    for pfx in ['depart', 'arrivee']:
        df_main = df_main.merge(
            lookup_ev.rename(columns={'pays': f'pays_{pfx}', 'is_evenement': f'evenement_{pfx}'}),
            on=['date', f'pays_{pfx}'], how='left'
        )
        df_main = df_main.merge(
            lookup_ho.rename(columns={'pays': f'pays_{pfx}', 'is_vacances': f'vacances_{pfx}'}),
            on=['date', f'pays_{pfx}'], how='left'
        )

    # Nettoyage et formatage final
    flags = ['evenement_depart', 'vacances_depart', 'evenement_arrivee', 'vacances_arrivee']
    df_main[flags] = df_main[flags].fillna(0).astype(int)

    # --- INJECTION DU BRUIT STOCHASTIQUE ---
    # Application d'une déviation standard de 8% sur la demande théorique
    noise = np.random.normal(loc=1.0, scale=0.08, size=len(df_main))
    df_main['value_jour'] = (df_main['value_jour'] * noise).round().astype(int)
    # Troncature des valeurs aberrantes (impossible d'avoir des passagers négatifs)
    df_main['value_jour'] = df_main['value_jour'].clip(lower=0)

    # Suppression des colonnes provoquant des fuites de données ou redondantes
    to_drop = ['period', 'value_mensuelle', 'norm_coeff', 
               'pays_depart', 'aeroport_depart', 'pays_arrivee', 'aeroport_arrivee']
    df_main = df_main.drop(columns=to_drop, errors='ignore')

    # Features cycliques pour la saisonnalité
    df_main['jour_semaine'] = df_main['date'].dt.dayofweek
    df_main['mois'] = df_main['date'].dt.month

    # Tri séquentiel par route (critique pour LSTM)
    df_main = df_main.sort_values(['route', 'date']).reset_index(drop=True)

    # Export (Exigence native Keras pour le modèle final ultérieur)
    df_main.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    generate_dataset()