# Traitement des dossiers gzip + nettoyage des fichiers

import pandas as pd

def load_and_melt_data(file_path):
    # 1. Chargement & Nettoyage colonnes
    df = pd.read_csv(file_path, sep='\t', compression='gzip')
    df.columns = [c.strip() for c in df.columns]
    id_col = df.columns[0]

    # 2. Filtrage structurel (M, PAS, PAS_CRD)
    df = df[df[id_col].str.startswith('M,PAS,PAS_CRD')].copy()

    # 3. Pivotage (Wide to Long)
    df_long = df.melt(id_vars=[id_col], var_name='period', value_name='value')

    # 4. Filtrage Temporel (YYYY-MM uniquement)
    df_long = df_long[df_long['period'].str.match(r'^\d{4}-\d{2}$')].copy()

    # 5. Nettoyage Valeurs
    df_long['value'] = pd.to_numeric(df_long['value'].astype(str).str.replace(':', ''), errors='coerce')
    df_long = df_long.dropna(subset=['value'])

    # 6. Extraction Route
    df_long['route'] = df_long[id_col].apply(lambda x: x.split(',')[-1])
    
    return df_long[['period', 'route', 'value']]