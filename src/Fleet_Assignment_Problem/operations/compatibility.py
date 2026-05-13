# Résout le produit cartésien entre la flotte et le réseau. Élimine les affectations physiquement impossibles via la contrainte du rayon d'action.

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def build_compatibility_matrix():
    BASE_DIR = Path(__file__).resolve().parents[3]
    DIST_FILE = BASE_DIR / "data" / "processed" / "distances.parquet"
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "compatibility.parquet"

    if not DIST_FILE.exists() or not FLEET_FILE.exists():
        logging.error("Fichiers manquants. Lancez d'abord operations.network et operations.fleet")
        return

    logging.info("Chargement des données...")
    df_dist = pd.read_parquet(DIST_FILE)
    df_fleet = pd.read_parquet(FLEET_FILE)

    logging.info(f"Routes à couvrir : {len(df_dist)}")
    logging.info(f"Flotte disponible : {len(df_fleet)} types d'avions")

    # --- CROSS JOIN (Produit Cartésien) ---
    # On teste chaque avion sur chaque route
    df_dist['key'] = 1
    df_fleet['key'] = 1
    
    logging.info("Génération des combinaisons (peut prendre quelques secondes)...")
    df_cross = pd.merge(
        df_dist[['route', 'distance_km', 'key']],
        df_fleet[['fleet_id', 'range_km', 'speed_kmh', 'capacity', 'key']],
        on='key'
    ).drop('key', axis=1)

    # --- RÈGLES DE PHYSIQUE ---
    logging.info("Application des contraintes opérationnelles...")
    
    # 1. Contrainte de Rayon d'Action (Range)
    # L'avion doit pouvoir voler la distance.
    df_cross['feasible'] = df_cross['range_km'] >= df_cross['distance_km']

    # 2. Calcul du Temps de Vol (Block Time)
    # Formule : (Distance / Vitesse) + 45 min de temps au sol/approche (standard IATA simplifié)
    # Résultat en heures décimales (ex: 1.5 = 1h30)
    TAXI_TIME_HOURS = 0.75 # 45 minutes de marge fixe
    df_cross['flight_time_hours'] = (df_cross['distance_km'] / df_cross['speed_kmh']) + TAXI_TIME_HOURS
    
    # 3. Filtrage
    # On ne garde que les paires possibles
    df_compat = df_cross[df_cross['feasible']].copy()
    
    # Nettoyage final
    final_cols = ['route', 'fleet_id', 'distance_km', 'flight_time_hours', 'capacity']
    df_final = df_compat[final_cols]

    # --- STATISTIQUES ---
    routes_covered = df_final['route'].nunique()
    total_routes = df_dist['route'].nunique()
    orphans = total_routes - routes_covered
    
    logging.info(f"SUCCÈS : {len(df_final)} affectations possibles générées.")
    logging.info(f"Couverture Réseau : {routes_covered}/{total_routes} routes faisables.")
    
    if orphans > 0:
        logging.warning(f"Attention : {orphans} routes sont trop longues pour TOUS vos avions (ou données manquantes).")

    # Sauvegarde
    df_final.to_parquet(OUTPUT_FILE, index=False)
    logging.info(f"Matrice de compatibilité sauvegardée : {OUTPUT_FILE}")

if __name__ == "__main__":
    build_compatibility_matrix()