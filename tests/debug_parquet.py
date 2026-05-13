import pandas as pd
from pathlib import Path

def lire_premieres_lignes():
    # Remonte de tests/ à la racine du projet, puis pointe vers data/processed
    dossier_processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    
    fichiers_parquet = list(dossier_processed.glob("*.parquet"))
    
    if not fichiers_parquet:
        print(f"Aucun fichier .parquet trouvé dans {dossier_processed}")
        return

    for fichier in fichiers_parquet:
        print(f"\n=== Fichier : {fichier.name} ===")
        try:
            df = pd.read_parquet(fichier)
            print(df.head(2))
        except Exception as e:
            print(f"Erreur lors de la lecture : {e}")

if __name__ == "__main__":
    lire_premieres_lignes()