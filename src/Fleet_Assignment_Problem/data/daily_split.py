import pandas as pd
from pathlib import Path

def expand_date_ranges(df, country_col):
    """
    Désagrège une plage de dates (date_debut -> date_fin) en n lignes (1 par jour).
    Permet des jointures strictes de type "date-to-date" par la suite.
    """
    df_exp = df.copy()
    # Création d'une liste de dates pour chaque intervalle
    df_exp['date'] = df_exp.apply(lambda row: pd.date_range(row['date_debut'], row['date_fin']), axis=1)
    # Éclatement des listes en lignes individuelles
    df_exp = df_exp.explode('date')
    # Création d'un marqueur booléen (True) pour identifier la présence de l'événement
    df_exp['flag'] = True
    return df_exp[['date', country_col, 'flag']].drop_duplicates()

def split_monthly_to_daily_2024():
    # 1. Déclaration des chemins relatifs absolus et des variables d'environnement
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_RAW = BASE_DIR / "data" / "raw"
    TRAFFIC_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_network_traffic.parquet"
    VACANCES_FILE = DATA_RAW / "Evenement_pays" / "Vacances_scolaire_2024_2027.csv"
    EVENTS_FILE = DATA_RAW / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2024.parquet"

    # 2. Paramétrage des chocs de demande (Multiplicateurs d'impact calendaire)
    MULT_WEEKEND = 1.40
    MULT_VACANCES = 1.25
    MULT_EVENT = 1.40

    # 3. Ingestion et nettoyage du référentiel de trafic (Contrainte hiérarchique Macro)
    df_traffic = pd.read_parquet(TRAFFIC_FILE)
    df_traffic = df_traffic[df_traffic['period'].str.startswith('2024')].copy()
    if 'date' in df_traffic.columns:
        df_traffic = df_traffic.drop(columns=['date'])
    df_traffic = df_traffic.rename(columns={'value': 'value_mensuelle'})

    # 4. Ingestion et vectorisation des référentiels conjoncturels (Vacances et Événements)
    df_vac = pd.read_csv(VACANCES_FILE, sep=';', encoding='utf-8')
    df_vac['date_debut'] = pd.to_datetime(df_vac['date_debut'], format='%d/%m/%Y')
    df_vac['date_fin'] = pd.to_datetime(df_vac['date_fin'], format='%d/%m/%Y')
    df_vac_daily = expand_date_ranges(df_vac, 'pays').rename(columns={'flag': 'is_vacance'})

    df_evt = pd.read_csv(EVENTS_FILE, sep=';', encoding='utf-8')
    df_evt['date_debut'] = pd.to_datetime(df_evt['date_debut'], format='%d/%m/%Y')
    df_evt['date_fin'] = pd.to_datetime(df_evt['date_fin'], format='%d/%m/%Y')
    df_evt_daily = expand_date_ranges(df_evt, 'pays').rename(columns={'flag': 'is_event'})

    # 5. Instanciation du calendrier isotrope annuel
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    cal = pd.DataFrame({'date': dates})
    cal['period'] = cal['date'].dt.strftime('%Y-%m') # Clé de jointure pour le trafic mensuel
    cal['is_weekend'] = cal['date'].dt.dayofweek >= 5 # 5=Samedi, 6=Dimanche
    
    # 6. Intégration du signal de saisonnalité structurel (Week-end)
    cal['base_weight'] = 1.0 # Poids unitaire de base
    cal.loc[cal['is_weekend'], 'base_weight'] *= MULT_WEEKEND # Application du choc hebdo

    # 7. Distribution du trafic mensuel sur la matrice journalière (Jointure 1 à n)
    df_daily = df_traffic.merge(cal[['period', 'date', 'base_weight']], on='period', how='inner')
    df_daily['adjusted_coeff'] = df_daily['base_weight']

    # 8. Enrichissement géospatial : Détection des événements pour chaque segment (O/D)
    # Point d'Origine (Départ)
    df_daily = df_daily.merge(df_vac_daily.rename(columns={'pays': 'pays_depart', 'is_vacance': 'vac_dep'}), 
                              on=['date', 'pays_depart'], how='left')
    df_daily = df_daily.merge(df_evt_daily.rename(columns={'pays': 'pays_depart', 'is_event': 'evt_dep'}), 
                              on=['date', 'pays_depart'], how='left')
    # Point de Destination (Arrivée)
    df_daily = df_daily.merge(df_vac_daily.rename(columns={'pays': 'pays_arrivee', 'is_vacance': 'vac_arr'}), 
                              on=['date', 'pays_arrivee'], how='left')
    df_daily = df_daily.merge(df_evt_daily.rename(columns={'pays': 'pays_arrivee', 'is_event': 'evt_arr'}), 
                              on=['date', 'pays_arrivee'], how='left')

    # Traitement des valeurs nulles (Absence d'événement)
    cols_to_fill = ['vac_dep', 'evt_dep', 'vac_arr', 'evt_arr']
    df_daily[cols_to_fill] = df_daily[cols_to_fill].fillna(False)

    # 9. Application des chocs conjoncturels géolocalisés
    # Si vacances au départ OU à l'arrivée, on applique le multiplicateur
    df_daily.loc[df_daily['vac_dep'] | df_daily['vac_arr'], 'adjusted_coeff'] *= MULT_VACANCES
    df_daily.loc[df_daily['evt_dep'] | df_daily['evt_arr'], 'adjusted_coeff'] *= MULT_EVENT

    # 10. Normalisation Top-Down sous contrainte de mois et de route
    # Somme des coefficients bruts par route et par mois
    sum_coeffs = df_daily.groupby(['route', 'period'])['adjusted_coeff'].transform('sum')
    # Poids relatif de chaque jour dans le mois
    df_daily['norm_coeff'] = df_daily['adjusted_coeff'] / sum_coeffs

    # 11. Calcul final de la demande journalière discrète
    df_daily['value_jour'] = (df_daily['value_mensuelle'] * df_daily['norm_coeff']).round().astype(int)

    # 12. Formatage et Exportation
    cols_order = [
        'period', 'date', 'route', 'pays_depart', 'aeroport_depart', 
        'pays_arrivee', 'aeroport_arrivee', 'value_mensuelle', 'norm_coeff', 'value_jour'
    ]
    df_daily = df_daily[[c for c in cols_order if c in df_daily.columns]]
    df_daily.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    split_monthly_to_daily_2024()