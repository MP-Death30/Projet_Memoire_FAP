# Standardise la base de données matérielle. Type et isole les capacités, vitesses de croisière et rayons d'action nominaux des aéronefs.

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_fleet_data():
    BASE_DIR = Path(__file__).resolve().parents[3]
    INPUT_FILE = BASE_DIR / "data" / "raw" / "Plane_caracteristic" / "caracteristique_famille_moyen_courrier.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"

    if not INPUT_FILE.exists():
        logging.error(f"Fichier introuvable : {INPUT_FILE}")
        return

    logging.info(f"Chargement de la flotte depuis {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8')
    
    cols_to_keep = {
        'Famille': 'fleet_id',
        'Moy. n_pax': 'capacity',
        'Moy. Cruise_speed (km/h)': 'speed_kmh',
        'Moy. nominal_range (km)': 'range_km',
        'Cout_operation': 'cost'
    }
    
    available_cols = [c for c in cols_to_keep.keys() if c in df.columns]
    df = df[available_cols].rename(columns=cols_to_keep)

    numeric_cols = ['capacity', 'speed_kmh', 'range_km', 'cost']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=['fleet_id', 'capacity', 'speed_kmh', 'range_km', 'cost']).copy()

    df_clean.to_parquet(OUTPUT_FILE, index=False)
    
    logging.info(f"SUCCÈS : Flotte consolidée ({len(df_clean)} familles d'avions).")
    logging.info(f"Sauvegardé dans {OUTPUT_FILE}")

if __name__ == "__main__":
    process_fleet_data()