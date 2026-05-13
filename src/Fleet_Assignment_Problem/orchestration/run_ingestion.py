# Ingère, décompresse et nettoie les données brutes Eurostat. Consolide les déclarations miroirs pour générer le référentiel de trafic historique global.

import pandas as pd
from pathlib import Path
import logging
import time
from src.Fleet_Assignment_Problem.pipelines.loader import load_and_melt_data

# Configuration du logging (Traceability)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_etl_pipeline():
    # 1. Définition des chemins (Absolu ou Relatif robuste)
    # Remonte de 3 niveaux : main.py -> Fleet_Assignment_Problem -> src -> RACINE DU PROJET
    BASE_DIR = Path(__file__).resolve().parents[3] 
    RAW_DIR = BASE_DIR / "data" / "raw" / "UE_air_passenger_between_airport"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    # Création du dossier processed s'il n'existe pas
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Récupération des fichiers
    files = list(RAW_DIR.glob("*.tsv.gz"))
    if not files:
        logging.error(f"Aucun fichier trouvé dans {RAW_DIR}")
        return

    logging.info(f"Début du traitement de {len(files)} fichiers...")
    
    all_chunks = []
    start_time = time.time()

    # 3. Boucle de traitement
    for i, file_path in enumerate(files):
        try:
            # Appel de votre logique validée
            df_chunk = load_and_melt_data(str(file_path))
            
            if not df_chunk.empty:
                all_chunks.append(df_chunk)
                # Log réduit pour ne pas spammer la console (tous les 5 fichiers)
                if (i + 1) % 5 == 0:
                    logging.info(f"[{i+1}/{len(files)}] Traité : {file_path.name} ({len(df_chunk)} lignes)")
            else:
                logging.warning(f"[{i+1}/{len(files)}] Fichier vide après filtrage : {file_path.name}")
                
        except Exception as e:
            logging.error(f"Echec sur {file_path.name} : {e}")

    # 4. Consolidation, Déduplication et Sauvegarde
    if all_chunks:
        logging.info("Consolidation des données brutes...")
        full_df = pd.concat(all_chunks, ignore_index=True)
        
        logging.info(f"Lignes brutes récupérées : {len(full_df)}")

        # --- ETAPE DE CORRECTION (DEDUPLICATION) ---
        logging.info("Fusion des déclarations miroirs (Max Aggregation)...")
        # On groupe par 'period' et 'route', et on prend le Max de la valeur 'value'.
        # Cela résout le conflit où le pays de départ et le pays d'arrivée déclarent tous deux le vol.
        clean_df = full_df.groupby(['period', 'route'], as_index=False)['value'].max()
        
        logging.info(f"Lignes après nettoyage : {len(clean_df)} (Doublons éliminés)")

        output_path = PROCESSED_DIR / "consolidated_traffic.parquet"
        
        # Sauvegarde en Parquet (Rapide et compressé)
        clean_df.to_parquet(output_path, index=False)
        
        duration = time.time() - start_time
        logging.info(f"SUCCÈS : {len(clean_df)} lignes sauvegardées dans {output_path}")
        logging.info(f"Temps total : {duration:.2f} secondes")
    else:
        logging.warning("Aucune donnée valide récupérée.")

if __name__ == "__main__":
    run_etl_pipeline()