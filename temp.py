import pandas as pd
from pathlib import Path

# Ajustez le nom du fichier s'il est différent (ex: .csv, .xlsx)
# J'assume que c'est le premier fichier du dossier
base_dir = Path("data/raw/Plane_caracteristic")
files = list(base_dir.glob("*"))

if files:
    f = files[0]
    print(f"Fichier trouvé : {f.name}")
    
    if f.suffix == '.csv':
        df = pd.read_csv(f)
    elif f.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(f)
    else:
        print("Format non reconnu, lecture texte brute :")
        with open(f, 'r') as txt:
            print(txt.readlines()[:5])
            exit()
            
    print(df.head())
    print(df.columns.tolist())
else:
    print("Dossier vide.")