# Isole le périmètre opérationnel. Restreint le jeu de données aux liaisons bidirectionnelles strictes opérées vers et depuis le hub d'Orly (LFPO).

import pandas as pd
from pathlib import Path

def filter_transavia_orly_network():
    BASE_DIR = Path(__file__).resolve().parents[3]
    INPUT_FILE = BASE_DIR / "data" / "processed" / "gold_network_traffic.parquet"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "transavia_orly_network_traffic.parquet"

    if not INPUT_FILE.exists():
        print(f"Erreur : Fichier introuvable - {INPUT_FILE}")
        return

    transavia_destinations = {
        "LFMN", "LFBO", "LPPR", "LPPT", "LIRF", "LEMD", "LEBL", "LEZL", "LGIR",
        "EDDB", "LFTH", "LPFR", "LGAV", "LFBZ", "LEMG", "LFMP", "LEPA", "LFMT",
        "LICJ", "LFML", "LTFM", "LIPZ", "LEMH", "LIBD", "LIRN", "LGSR",
        "LGMK", "LOWW", "LGKR", "LEAL", "EIDW", "LICC", "LATI", "LPMA",
        "LGRP", "LIMC", "LEVC", "LIBR", "GCTS", "LGSA", "LGTS", "LMML", "LIEO",
        "LIRP", "LIEE", "GCRR", "GCLP", "LTAI", "LTBJ", "EGPH"
    }

    hub = "LFPO"

    df = pd.read_parquet(INPUT_FILE)

    # Restriction temporelle : 2023 à 2024
    df = df[df['period'].str[:4].astype(int).between(2023, 2024)]

    def is_orly_transavia_route(route_str):
        parts = str(route_str).split('_')
        icaos = [p for p in parts if len(p) == 4]
        if len(icaos) >= 2:
            orig, dest = icaos[0], icaos[1]
            return (orig == hub and dest in transavia_destinations) or \
                   (dest == hub and orig in transavia_destinations)
        return False

    mask = df['route'].apply(is_orly_transavia_route)
    df_filtered = df[mask].copy()

    df_filtered[['pays_depart', 'aeroport_depart', 'pays_arrivee', 'aeroport_arrivee']] = df_filtered['route'].str.split('_', n=3, expand=True)

    df_filtered['pair'] = df_filtered.apply(
        lambda x: "_".join(sorted([
            f"{x['pays_depart']}_{x['aeroport_depart']}",
            f"{x['pays_arrivee']}_{x['aeroport_arrivee']}"
        ])), axis=1
    )
    
    valid_pairs = df_filtered.groupby(['period', 'pair'])['route'].transform('nunique') == 2
    df_filtered = df_filtered[valid_pairs].drop(columns=['pair'])

    df_filtered.to_parquet(OUTPUT_FILE, index=False)
    
    print(f"Réseau Transavia Orly (2023 & 2024, bidirectionnel strict) : {len(df_filtered)} enregistrements conservés.")
    print(f"Fichier généré : {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_transavia_orly_network()