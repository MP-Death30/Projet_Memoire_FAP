import pandas as pd
import numpy as np
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

def split_monthly_to_daily_2023_2024():
    # 1. Déclaration des chemins relatifs absolus et des variables d'environnement
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_RAW = BASE_DIR / "data" / "raw"
    TRAFFIC_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_network_traffic.parquet"
    COEFF_FILE = DATA_RAW / "Coefficients_repartition" / "coefficient_repartition_jour.csv"
    VACANCES_FILE = DATA_RAW / "Evenement_pays" / "Vacances_scolaire_2024_2027.csv"
    EVENTS_FILE = DATA_RAW / "Evenement_pays" / "Evenement_sport_tech_business_2023_2027.csv"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_daily_2023_2024.parquet"

    # 2. Paramétrage des chocs de demande (Multiplicateurs d'impact calendaire)
    MULT_WEEKEND = 1.40
    MULT_VACANCES = 1.25
    MULT_EVENT = 1.40

    # 3. Ingestion et nettoyage du référentiel de trafic (Contrainte hiérarchique Macro)
    df_traffic = pd.read_parquet(TRAFFIC_FILE)
    df_traffic = df_traffic[df_traffic['period'].str[:4].astype(int).between(2023, 2024)].copy()  #df = df[df['period'].str[:4].astype(int).between(2023, 2024)]
    if 'date' in df_traffic.columns:
        df_traffic = df_traffic.drop(columns=['date'])
    df_traffic = df_traffic.rename(columns={'value': 'value_mensuelle'})

    # 4. Ingestion et vectorisation des référentiels conjoncturels (Vacances et Événements)
    
    # Dictionnaire de standardisation (à mapper sur le format de pays_depart/arrivee de df_traffic)
    COUNTRY_MAPPING = {
        'DE': 'Allemagne'
        'EL': 'Grèce',
        'ES': 'Espagne',
        'FR': 'France',
        'IE': 'Irlande',
        'IT': 'Italie',
        'MT': 'Malte',
        'PT': 'Portugal'
    }

    df_vac = pd.read_csv(VACANCES_FILE, sep=';', encoding='utf-8')
    df_vac['pays'] = df_vac['pays'].replace(COUNTRY_MAPPING) # Alignement de la nomenclature
    df_vac['date_debut'] = pd.to_datetime(df_vac['date_debut'], format='%d/%m/%Y')
    df_vac['date_fin'] = pd.to_datetime(df_vac['date_fin'], format='%d/%m/%Y')
    df_vac_daily = expand_date_ranges(df_vac, 'pays').rename(columns={'flag': 'is_vacance'})

    df_evt = pd.read_csv(EVENTS_FILE, sep=';', encoding='utf-8')
    df_evt['pays'] = df_evt['pays'].replace(COUNTRY_MAPPING) # Alignement de la nomenclature
    df_evt['date_debut'] = pd.to_datetime(df_evt['date_debut'], format='%d/%m/%Y')
    df_evt['date_fin'] = pd.to_datetime(df_evt['date_fin'], format='%d/%m/%Y')
    df_evt_daily = expand_date_ranges(df_evt, 'pays').rename(columns={'flag': 'is_event'})

    # 5. Instanciation du calendrier isotrope annuel et extraction des caractéristiques
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    cal = pd.DataFrame({'date': dates})
    cal['period'] = cal['date'].dt.strftime('%Y-%m') 
    cal['jour_du_mois'] = cal['date'].dt.day
    cal['type_jour'] = np.where(cal['date'].dt.dayofweek >= 5, 'w', 's') # Mapping de nomenclature
    cal['days_in_month'] = cal['date'].dt.days_in_month

    # 6. Intégration du signal de saisonnalité structurel via matrice de coefficients
    df_coeffs = pd.read_csv(COEFF_FILE, sep=';', encoding='utf-8')
    cal = cal.merge(df_coeffs, on=['jour_du_mois', 'type_jour'], how='left')

    # Sélection vectorisée de la colonne appropriée selon la longueur du mois
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
    
    # Assignation du poids de base. Les valeurs nulles issues des jours inexistants sont forcées à 0.0
    cal['base_weight'] = np.select(conditions, choices, default=0.0)

    # Nettoyage de l'espace mémoire
    cols_to_drop = [
        'jour_du_mois', 'type_jour', 'days_in_month', 
        'coefficient_28_29', 'coefficient_30', 'coefficient_31'
    ]
    cal = cal.drop(columns=cols_to_drop)

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
    split_monthly_to_daily_2023_2024()