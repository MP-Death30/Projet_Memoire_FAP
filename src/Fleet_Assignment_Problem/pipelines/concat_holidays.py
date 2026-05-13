import pandas as pd
import zipfile
from pathlib import Path

def process_holidays_zip():
    BASE_DIR = Path(__file__).resolve().parents[3]
    ZIP_PATH = BASE_DIR / "data" / "raw" / "Evenement_pays" / "fcal_API_Data_Vacances_20260512.zip"
    OUTPUT_PATH = BASE_DIR / "data" / "raw" / "Evenement_pays" / "vacances_consolidees.parquet"

    # Alignement du référentiel ISO avec le référentiel projet
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
                    # Extraction stricte selon la structure fournie (séparateur tabulation détecté)
                    df_temp = pd.read_csv(f, sep='\t')
                    dfs.append(df_temp)

    df_concat = pd.concat(dfs, ignore_index=True)

    # Typage temporel et spatial
    df_concat['date'] = pd.to_datetime(df_concat['date'], format='%d.%m.%Y')
    df_concat['pays'] = df_concat['iso'].map(COUNTRY_MAPPING).fillna(df_concat['iso'])
    
    # Transformation de la variable explicative (Échelle 0.0 - 1.0 pour le réseau de neurones)
    df_concat['sum_percent'] = pd.to_numeric(
        df_concat['sum_percent'].astype(str).str.replace(',', '.'), 
        errors='coerce'
    ).fillna(0) / 100.0

    # Isolement des features requises
    df_final = df_concat[['date', 'pays', 'sum_percent']].copy()
    
    # Déduplication en cas de chevauchement sur la clé composite
    df_final = df_final.groupby(['date', 'pays'], as_index=False)['sum_percent'].max()

    df_final.to_parquet(OUTPUT_PATH, index=False)

if __name__ == "__main__":
    process_holidays_zip()