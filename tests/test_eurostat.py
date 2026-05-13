import pandas as pd
import matplotlib.pyplot as plt

# Chargement du fichier TSV
# Note : Eurostat utilise souvent des tabulations ou des virgules suivies d'espaces
df = pd.read_csv('/dataset/UE_air_passenger_between_airport/estat_avia_par_fr.tsv', sep='\t')
df.columns = [c.strip() for c in df.columns]

# 2. Définition du code de départ souhaité
# Exemple : 'FR_FMCZ' pour l'aéroport de départ
depart_cible = 'FR_LFPG'

# 3. Filtrage du départ
# On découpe la première colonne : la route est après la dernière virgule
def est_depart_cible(row_label):
    parts = row_label.split(',')
    # La route est généralement le dernier élément de la liste avant les données
    route = parts[-1].strip() 
    # Le départ est composé des deux premiers segments (ex: FR + FMCZ)
    route_parts = route.split('_')
    if len(route_parts) >= 2:
        return f"{route_parts[0]}_{route_parts[1]}" == depart_cible
    return False

df_filtered = df[df[df.columns[0]].apply(est_depart_cible)]

if df_filtered.empty:
    print(f"Aucune donnée trouvée pour le départ : {depart_cible}")
else:
    # 4. Transformation (Melt)
    df_long = df_filtered.melt(id_vars=[df.columns[0]], var_name='period', value_name='value')

    # CORRECTION DE L'ERREUR ICI : Utilisation de .str pour les méthodes de texte
    df_long['value'] = df_long['value'].astype(str).str.replace(':', '').str.strip()
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
    
    df_long = df_long.dropna(subset=['value'])

    # 5. Sélection de la période et Agrégation
    # Filtrage des années (4 chiffres)
    df_plot = df_long[df_long['period'].str.match(r'^\d{4}$')]
    
    # On somme toutes les destinations pour ce départ par année
    df_final = df_plot.groupby('period')['value'].sum().reset_index()

    # 6. Graphique
    plt.figure(figsize=(12, 6))
    plt.bar(df_final['period'], df_final['value'], color='navy')
    plt.title(f"Volume total au départ de : {depart_cible}")
    plt.xlabel("Année")
    plt.ylabel("Valeur cumulée")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()