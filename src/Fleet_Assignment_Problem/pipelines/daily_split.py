import pandas as pd
import numpy as np
from pathlib import Path

def expand_date_ranges(df, country_col):
    df_exp = df.copy()
    df_exp['date'] = df_exp.apply(lambda row: pd.date_range(row['date_debut'], row['date_fin']), axis=1)
    df_exp = df_exp.explode('date')
    df_exp['flag'] = True
    return df_exp[['date', country_col, 'flag']].drop_duplicates()

def split_monthly_to_daily_2023_2024():
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_RAW = BASE_DIR / "data" / "raw"
    TRAFFIC_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_network_traffic.parquet"
    COEFF_FILE = DATA_RAW / "Coefficients_repartition" / "coefficient_repartition_jour.csv"
    VACANCES_FILE = DATA_RAW / "Evenement_pays" / "vacances_consolidees.parquet"
    EVENTS_FILE = DATA_RAW / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2023_2024.parquet"

    MULT_VACANCES = 1.25
    MULT_EVENT = 1.40

    df_traffic = pd.read_parquet(TRAFFIC_FILE)
    df_traffic = df_traffic[df_traffic['period'].str[:4].astype(int).between(2023, 2024)].copy()
    if 'date' in df_traffic.columns:
        df_traffic = df_traffic.drop(columns=['date'])
    df_traffic = df_traffic.rename(columns={'value': 'value_mensuelle'})

    COUNTRY_MAPPING = {
        'DE': 'Allemagne',
        'EL': 'Grèce',
        'ES': 'Espagne',
        'FR': 'France',
        'IE': 'Irlande',
        'IT': 'Italie',
        'MT': 'Malte',
        'PT': 'Portugal',
    }

    df_vac_daily = pd.read_parquet(VACANCES_FILE)

    df_evt = pd.read_csv(EVENTS_FILE, sep=';', encoding='utf-8')
    df_evt['pays'] = df_evt['pays'].replace(COUNTRY_MAPPING) 
    df_evt['date_debut'] = pd.to_datetime(df_evt['date_debut'], format='%d/%m/%Y')
    df_evt['date_fin'] = pd.to_datetime(df_evt['date_fin'], format='%d/%m/%Y')
    df_evt_daily = expand_date_ranges(df_evt, 'pays').rename(columns={'flag': 'is_event'})

    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    cal = pd.DataFrame({'date': dates})
    cal['period'] = cal['date'].dt.strftime('%Y-%m') 
    cal['jour_du_mois'] = cal['date'].dt.day
    cal['type_jour'] = np.where(cal['date'].dt.dayofweek >= 5, 'w', 's') 
    cal['days_in_month'] = cal['date'].dt.days_in_month

    df_coeffs = pd.read_csv(COEFF_FILE, sep=';', encoding='utf-8')
    cal = cal.merge(df_coeffs, on=['jour_du_mois', 'type_jour'], how='left')

    conditions = [
        cal['days_in_month'].isin([28, 29]),
        cal['days_in_month'] == 30,
        cal['days_in_month'] == 31
    ]
    choices = [
        cal['coefficient_28_29'],
        cal['coefficient_30'],
        cal['coefficient_31']
    ]
    cal['base_weight'] = np.select(conditions, choices, default=0.0)

    cols_to_drop = [
        'jour_du_mois', 'type_jour', 'days_in_month', 
        'coefficient_28_29', 'coefficient_30', 'coefficient_31'
    ]
    cal = cal.drop(columns=cols_to_drop)

    df_daily = df_traffic.merge(cal[['period', 'date', 'base_weight']], on='period', how='inner')
    df_daily['adjusted_coeff'] = df_daily['base_weight']

    df_daily = df_daily.merge(df_vac_daily.rename(columns={'pays': 'pays_depart', 'sum_percent': 'vac_dep'}), 
                              on=['date', 'pays_depart'], how='left')
    df_daily = df_daily.merge(df_evt_daily.rename(columns={'pays': 'pays_depart', 'is_event': 'evt_dep'}), 
                              on=['date', 'pays_depart'], how='left')
    df_daily = df_daily.merge(df_vac_daily.rename(columns={'pays': 'pays_arrivee', 'sum_percent': 'vac_arr'}), 
                              on=['date', 'pays_arrivee'], how='left')
    df_daily = df_daily.merge(df_evt_daily.rename(columns={'pays': 'pays_arrivee', 'is_event': 'evt_arr'}), 
                              on=['date', 'pays_arrivee'], how='left')

    df_daily['vac_dep'] = df_daily['vac_dep'].fillna(0.0)
    df_daily['vac_arr'] = df_daily['vac_arr'].fillna(0.0)
    df_daily['evt_dep'] = df_daily['evt_dep'].fillna(False)
    df_daily['evt_arr'] = df_daily['evt_arr'].fillna(False)

    max_vac_impact = df_daily[['vac_dep', 'vac_arr']].max(axis=1)
    df_daily['adjusted_coeff'] = df_daily['adjusted_coeff'] * (1 + (MULT_VACANCES - 1) * max_vac_impact)

    df_daily.loc[df_daily['evt_dep'] | df_daily['evt_arr'], 'adjusted_coeff'] *= MULT_EVENT

    sum_coeffs = df_daily.groupby(['route', 'period'])['adjusted_coeff'].transform('sum')
    df_daily['norm_coeff'] = df_daily['adjusted_coeff'] / sum_coeffs

    df_daily['value_jour'] = (df_daily['value_mensuelle'] * df_daily['norm_coeff']).round().astype(int)

    cols_order = [
        'period', 'date', 'route', 'pays_depart', 'aeroport_depart', 
        'pays_arrivee', 'aeroport_arrivee', 'value_mensuelle', 'norm_coeff', 'value_jour'
    ]
    df_daily = df_daily[[c for c in cols_order if c in df_daily.columns]]
    df_daily.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    split_monthly_to_daily_2023_2024()