import pandas as pd
from pathlib import Path

def export_to_csv():
    BASE_DIR = Path(__file__).resolve().parents[3]
    INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.csv"

    if not INPUT_FILE.exists():
        print(f"Erreur : Fichier introuvable - {INPUT_FILE}")
        return

    df = pd.read_parquet(INPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
    
    print(f"Export CSV terminé ({len(df)} lignes).")
    print(f"Fichier disponible : {OUTPUT_FILE}")

if __name__ == "__main__":
    export_to_csv()