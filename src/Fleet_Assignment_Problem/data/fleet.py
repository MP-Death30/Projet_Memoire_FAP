import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_fleet_data():
    BASE_DIR = Path(__file__).resolve().parents[3]
    INPUT_FILE = BASE_DIR / "data" / "raw" / "Plane_caracteristic" / "CADO_airplane_database_v1.0.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"

    if not INPUT_FILE.exists():
        logging.error(f"Fichier introuvable : {INPUT_FILE}")
        return

    logging.info(f"Chargement de la flotte depuis {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE, sep=';', dtype=str)
    
    # Suppression des lignes de métadonnées
    df = df.iloc[2:].copy()
    
    # Mapping des colonnes
    cols_to_keep = {
        'iata_code': 'code',
        'name': 'name',
        'n_pax': 'capacity',
        'cruise_speed': 'speed_kmh',
        'nominal_range': 'range_km',
        'mtow': 'mtow',
        'max_fuel': 'max_fuel',
        'airplane_type': 'category'
    }
    
    # Renommage et sélection
    available_cols = [c for c in cols_to_keep.keys() if c in df.columns]
    df = df[available_cols].rename(columns=cols_to_keep)

    # Conversion numérique
    numeric_cols = ['capacity', 'speed_kmh', 'range_km', 'mtow', 'max_fuel']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- FILTRES UTILISATEUR STRICTS ---
    logging.info(f"Total avant filtrage : {len(df)}")
    
    # 1. Nettoyage technique (Suppression des NaN critiques uniquement)
    # Un avion sans capacité ou sans code est inutilisable
    df_clean = df.dropna(subset=['code', 'capacity'])

    # 2. Logique de filtrage spécifiée
    df_clean = df_clean[
        (df_clean['capacity'] >= 30) &          # Capacité >= 30
        (df_clean['code'] != 'unknown') &       # Différent de unknown
        (df_clean['code'].str.len() == 3)       # Code IATA valide (3 char)
    ]

    # --- CRÉATION ID UNIQUE ---
    # Puisqu'on garde les variantes, 'code' n'est plus unique.
    # On crée 'fleet_id' : code + partie du nom (ex: 744_Boeing747-400)
    df_clean['fleet_id'] = df_clean['code'] + '_' + df_clean['name'].astype(str).str.replace(' ', '').str[:10]

    # Sauvegarde
    df_clean.to_parquet(OUTPUT_FILE, index=False)
    
    logging.info(f"SUCCÈS : Flotte consolidée ({len(df_clean)} avions).")
    logging.info(f"Sauvegardé dans {OUTPUT_FILE}")

if __name__ == "__main__":
    process_fleet_data()