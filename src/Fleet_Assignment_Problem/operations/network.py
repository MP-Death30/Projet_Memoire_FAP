# Spatialise le réseau. Calcule la distance orthodromique (Haversine) entre chaque paire d'aéroports ICAO.

import pandas as pd
import numpy as np
import airportsdata
from pathlib import Path
import logging
from math import radians, cos, sin, asin, sqrt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def haversine(lon1, lat1, lon2, lat2):
    """
    Calcule la distance grand cercle en km entre deux points (lon, lat).
    Vectorisé pour numpy.
    """
    # Conversion décimal -> radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Formule de Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Rayon de la Terre en km
    return c * r

def build_distance_matrix():
    BASE_DIR = Path(__file__).resolve().parents[3]
    TRAFFIC_FILE = BASE_DIR / "data" / "processed" / "gold_network_traffic.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "distances.parquet"

    if not TRAFFIC_FILE.exists():
        logging.error(f"Fichier trafic introuvable : {TRAFFIC_FILE}")
        return

    logging.info("Chargement du réseau Gold...")
    df_traffic = pd.read_parquet(TRAFFIC_FILE)
    
    # Extraction des routes uniques pour ne pas recalculer 1000 fois la même chose
    unique_routes = df_traffic['route'].unique()
    logging.info(f"Calcul des distances pour {len(unique_routes)} routes uniques...")

    # Chargement de la DB aéroports (Codes ICAO : LFPG, KJFK...)
    airports = airportsdata.load('ICAO')

    routes_data = []
    missing_airports = set()

    for route in unique_routes:
        # Format Eurostat attendu : "PAYS_ICAO1_PAYS_ICAO2" (ex: FR_LFPG_US_KJFK)
        parts = route.split('_')
        
        # Robustesse : Parfois le format varie. On cherche les éléments de 4 lettres.
        icaos = [p for p in parts if len(p) == 4]
        
        if len(icaos) >= 2:
            origin_code = icaos[0]
            dest_code = icaos[1]
            
            if origin_code in airports and dest_code in airports:
                orig = airports[origin_code]
                dest = airports[dest_code]
                
                routes_data.append({
                    'route': route,
                    'origin_icao': origin_code,
                    'dest_icao': dest_code,
                    'origin_lat': orig['lat'],
                    'origin_lon': orig['lon'],
                    'dest_lat': dest['lat'],
                    'dest_lon': dest['lon']
                })
            else:
                if origin_code not in airports: missing_airports.add(origin_code)
                if dest_code not in airports: missing_airports.add(dest_code)

    # Création DataFrame
    df_dist = pd.DataFrame(routes_data)
    
    if df_dist.empty:
        logging.error("Aucune distance calculée. Vérifiez le format des routes.")
        return

    # Calcul vectorisé
    df_dist['distance_km'] = haversine(
        df_dist['origin_lon'], df_dist['origin_lat'],
        df_dist['dest_lon'], df_dist['dest_lat']
    )

    # Nettoyage
    final_df = df_dist[['route', 'origin_icao', 'dest_icao', 'distance_km']]
    
    # Sauvegarde
    final_df.to_parquet(OUTPUT_FILE, index=False)

    logging.info(f"SUCCÈS : {len(final_df)} distances calculées.")
    if missing_airports:
        logging.warning(f"Aéroports inconnus ({len(missing_airports)}) : {list(missing_airports)[:10]}...")
    
    # Exemple de validation
    if 'FR_LFPG_US_KJFK' in final_df['route'].values: # CDG -> JFK
        dist = final_df[final_df['route'] == 'FR_LFPG_US_KJFK']['distance_km'].iloc[0]
        logging.info(f"Check CDG->JFK : {dist:.0f} km (Attendu ~5800 km)")

if __name__ == "__main__":
    build_distance_matrix()