import pandas as pd
from pathlib import Path

# Définition du chemin vers les données transformées
# (En se basant sur la structure de votre projet : src/Fleet_Assignment_Problem/data/...)
processed_dir = "data\processed"

# Récupération de tous les fichiers .parquet
parquet_files = list(processed_dir.glob("*.parquet"))

if not parquet_files:
    print(f"Aucun fichier .parquet trouvé dans {processed_dir}")
else:
    for file_path in parquet_files:
        print(f"\n--- Fichier : {file_path.name} ---")
        try:
            # Chargement et affichage des 2 premières lignes
            df = pd.read_parquet(file_path)
            print(df.head(2))
        except Exception as e:
            print(f"Erreur lors de la lecture de {file_path.name} : {e}")