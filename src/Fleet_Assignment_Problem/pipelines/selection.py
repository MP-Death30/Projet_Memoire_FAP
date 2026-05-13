import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def select_viable_network():
    # 1. Configuration des chemins
    # CORRECTION ICI : parents[3] car nous sommes dans src/Fleet/data/selection.py
    # parents[0] = data
    # parents[1] = Fleet_Assignment_Problem
    # parents[2] = src
    # parents[3] = RACINE DU PROJET
    BASE_DIR = Path(__file__).resolve().parents[3]

    INPUT_FILE = BASE_DIR / "data" / "processed" / "consolidated_traffic.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "gold_network_traffic.parquet"
    
    # Vérification de sécurité avant chargement
    if not INPUT_FILE.exists():
        logging.error(f"Fichier introuvable : {INPUT_FILE}")
        return
        
    logging.info(f"Chargement de {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Conversion date pour l'analyse temporelle
    df['date'] = pd.to_datetime(df['period'])

    # 2. Calcul des métriques par route (L'ADN de la route)
    logging.info("Calcul des métriques de viabilité par route...")
    route_stats = df.groupby('route').agg(
        total_pax=('value', 'sum'),           # Volume total historique
        months_active=('period', 'count'),    # Profondeur historique
        last_seen=('date', 'max')             # Dernière activité connue
    ).reset_index()

    initial_count = len(route_stats)
    logging.info(f"Total routes candidates : {initial_count}")

    # 3. Application des filtres "Business" (Entonnoir)
    
    # CRITÈRE A : Activité Récente (La route doit exister en 2023)
    # On rejette les lignes fermées avant le COVID ou pendant la crise
    cutoff_date = pd.Timestamp('2023-01-01')
    active_routes = route_stats[route_stats['last_seen'] >= cutoff_date]
    logging.info(f"Filtre 1 (Actives en 2023) : {len(active_routes)} routes restantes")

    # CRITÈRE B : Densité Historique (Au moins 5 ans de données)
    # Nécessaire pour entrainer un modèle de prédiction (Saisonnalité)
    robust_routes = active_routes[active_routes['months_active'] >= 60]
    logging.info(f"Filtre 2 (Historique > 5 ans) : {len(robust_routes)} routes restantes")

    # CRITÈRE C : Volume Minimum (Éviter l'aviation générale/privée)
    # 10 000 passagers cumulés sur la période (seuil bas mais filtre le bruit)
    final_routes_list = robust_routes[robust_routes['total_pax'] >= 10000]['route'].tolist()
    logging.info(f"Filtre 3 (Volume significatif) : {len(final_routes_list)} routes qualifiées")

    # 4. Filtrage du Dataset Principal
    # On ne garde que les données correspondant à ces routes d'élite
    df_gold = df[df['route'].isin(final_routes_list)].copy()
    
    # 5. Sauvegarde
    df_gold.to_parquet(OUTPUT_FILE, index=False)
    
    ratio = (len(final_routes_list) / initial_count) * 100
    logging.info(f"SUCCÈS : Dataset Gold généré ({len(df_gold)} lignes).")
    logging.info(f"Réduction du réseau : {ratio:.2f}% des routes conservées (Le 'Coeur de Réseau').")

if __name__ == "__main__":
    select_viable_network()