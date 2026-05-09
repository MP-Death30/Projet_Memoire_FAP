import pandas as pd
from pathlib import Path

def split_monthly_to_daily_2024():
    BASE_DIR = Path(__file__).resolve().parents[3]
    TRAFFIC_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_network_traffic.parquet"
    COEFF_FILE = BASE_DIR / "data" / "raw" / "Coefficients_repartition" / "coefficient_repartition_jour.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2024.parquet"

    if not TRAFFIC_FILE.exists():
        print(f"Échec : Fichier trafic introuvable -> {TRAFFIC_FILE}")
        return
        
    if not COEFF_FILE.exists():
        print(f"Échec : Fichier coefficients introuvable -> {COEFF_FILE}")
        return

    df_traffic = pd.read_parquet(TRAFFIC_FILE)
    df_traffic = df_traffic[df_traffic['period'].str.startswith('2024')].copy()
    
    if 'date' in df_traffic.columns:
        df_traffic = df_traffic.drop(columns=['date'])

    try:
        df_coeff = pd.read_csv(COEFF_FILE, sep=None, engine='python', decimal='.', encoding='utf-8-sig')
    except Exception as e:
        print(f"Échec de lecture du CSV : {e}")
        return

    df_coeff.columns = df_coeff.columns.str.strip()
    
    if 'jour_du_mois' not in df_coeff.columns:
        print(f"Colonnes détectées dans le CSV : {list(df_coeff.columns)}")
        print("Échec : La colonne 'jour_du_mois' reste introuvable.")
        return

    df_coeff['type_jour'] = df_coeff['type_jour'].astype(str).str.strip()

    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    cal = pd.DataFrame({'date': dates})
    
    cal['period'] = cal['date'].dt.strftime('%Y-%m')
    cal['jour_du_mois'] = cal['date'].dt.day
    cal['days_in_month'] = cal['date'].dt.daysinmonth
    cal['type_jour'] = cal['date'].dt.dayofweek.apply(lambda x: 'w' if x >= 5 else 's')

    cal = cal.merge(df_coeff, on=['jour_du_mois', 'type_jour'], how='left')

    def select_coeff(row):
        dim = row['days_in_month']
        if dim in [28, 29]:
            return row['coefficient_28_29']
        elif dim == 30:
            return row['coefficient_30']
        else:
            return row['coefficient_31']

    cal['raw_coeff'] = cal.apply(select_coeff, axis=1)

    cal['norm_coeff'] = cal['raw_coeff'] / cal.groupby('period')['raw_coeff'].transform('sum')
    
    cal_clean = cal[['period', 'date', 'norm_coeff']].copy()

    df_daily = df_traffic.merge(cal_clean, on='period', how='inner')

    df_daily['value_jour'] = (df_daily['value'] * df_daily['norm_coeff']).round().astype(int)

    df_daily = df_daily.rename(columns={'value': 'value_mensuelle'})
    
    cols_order = [
        'period', 'date', 'route', 'pays_depart', 'aeroport_depart', 
        'pays_arrivee', 'aeroport_arrivee', 'value_mensuelle', 'norm_coeff', 'value_jour'
    ]
    existing_cols = [c for c in cols_order if c in df_daily.columns]
    df_daily = df_daily[existing_cols]

    df_daily.to_parquet(OUTPUT_FILE, index=False)
    
    routes_count = df_daily['route'].nunique()
    print(f"Désagrégation terminée : {len(df_daily)} lignes générées pour {routes_count} routes sur 2024.")
    print(f"Fichier sauvegardé : {OUTPUT_FILE}")

if __name__ == "__main__":
    split_monthly_to_daily_2024()