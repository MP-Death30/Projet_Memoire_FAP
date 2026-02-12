import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Chargement optimisé
# Notez la vitesse de lecture du Parquet vs le CSV
DATA_PATH = Path("../data/processed/consolidated_traffic.parquet")
print(f"Chargement de {DATA_PATH}...")
df = pd.read_parquet(DATA_PATH)

# 2. Conversion Temporelle & Tri
# Indispensable pour que les graphiques ne soient pas chaotiques
df['date'] = pd.to_datetime(df['period'])
df = df.sort_values('date')

print(f"Période couverte : {df['date'].min()} -> {df['date'].max()}")
print(f"Nombre total de routes uniques : {df['route'].nunique()}")

# 3. Analyse de la complétude (Sparsity check)
# Dans le FAP, une route avec trop de trous est inutilisable.
route_counts = df['route'].value_counts()
print("\n--- Top 5 Routes les plus fréquentes (mois de données) ---")
print(route_counts.head())

print("\n--- Bottom 5 Routes (Données fragmentaires) ---")
print(route_counts.tail())

# 4. Aggrégation Globale (Vision Macro)
# Permet de détecter les ruptures de données systémiques (ex: changement de norme Eurostat)
global_trend = df.groupby('date')['value'].sum()

plt.figure(figsize=(12, 6))
plt.plot(global_trend.index, global_trend.values, label='Trafic Total UE')
plt.title("Audit de continuité temporelle (Trafic Mensuel Aggregé)")
plt.grid(True, alpha=0.3)
plt.axvline(pd.to_datetime('2020-03-01'), color='r', linestyle='--', label='COVID Impact')
plt.legend()
plt.show()

# 5. Zoom sur un Hub Critique (ex: CDG)
# Le FAP se joue souvent sur les hubs. Vérifions la cohérence locale.
hub_target = 'FR_LFPG' # Charles de Gaulle
mask_hub = df['route'].str.contains(hub_target)
df_hub = df[mask_hub]

print(f"\n--- Audit Hub {hub_target} ---")
print(f"Volume total passagers : {df_hub['value'].sum():,.0f}")
print(f"Routes connectées : {df_hub['route'].nunique()}")

# Visualisation des top routes du Hub
top_hub_routes = df_hub.groupby('route')['value'].sum().nlargest(5).index
df_hub_top = df_hub[df_hub['route'].isin(top_hub_routes)]

pivot_hub = df_hub_top.pivot(index='date', columns='route', values='value')
pivot_hub.plot(figsize=(12, 6), title=f"Top 5 Routes depuis/vers {hub_target}")
plt.show()