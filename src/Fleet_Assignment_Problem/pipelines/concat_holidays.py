# Unifie les fichiers de congés européens. Formate la proportion de population inactive sous forme de variable explicative continue.

import pandas as pd
import zipfile
from pathlib import Path

def process_holidays_zip():
    BASE_DIR = Path(__file__).resolve().parents[3]
    ZIP_PATH = BASE_DIR / "data" / "raw" / "Evenement_pays" / "fcal_API_Data_Vacances_20260512.zip"
    OUTPUT_PATH = BASE_DIR / "data" / "raw" / "Evenement_pays" / "vacances_consolidees.parquet"

    COUNTRY_MAPPING = {
        'DE': 'Allemagne', 'EL': 'Grèce', 'GR': 'Grèce',
        'ES': 'Espagne', 'FR': 'France', 'IE': 'Irlande', 'IT': 'Italie',
        'MT': 'Malte', 'PT': 'Portugal'
    }

    dfs = []
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        for filename in z.namelist():
            if filename.startswith("fcal_API_data_") and filename.endswith(".csv"):
                with z.open(filename) as f:
                    df_temp = pd.read_csv(f, sep=';', encoding='utf-8', on_bad_lines='skip')
                    dfs.append(df_temp)

    if not dfs:
        return

    df_concat = pd.concat(dfs, ignore_index=True)

    df_concat['date'] = pd.to_datetime(df_concat['date'], format='%d.%m.%Y')
    df_concat['pays'] = df_concat['iso'].map(COUNTRY_MAPPING)
    
    # Élimination des pays non mappés (Danemark, Suède)
    df_concat = df_concat.dropna(subset=['pays'])
    
    df_concat['sum_percent'] = pd.to_numeric(
        df_concat['sum_percent'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0) / 100.0

    df_final = df_concat[['date', 'pays', 'sum_percent']].copy()
    df_final = df_final.groupby(['date', 'pays'], as_index=False)['sum_percent'].max()

    df_final.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    process_holidays_zip()