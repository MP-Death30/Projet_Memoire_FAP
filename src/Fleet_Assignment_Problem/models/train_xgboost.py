import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import joblib

def train_xgb_model():
    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    MODEL_DIR = BASE_DIR / "models"
    
    print("Chargement du jeu de données historique (dataset_lstm.parquet)...")
    df = pd.read_parquet(DATA_FILE)
    
    # 1. Reconstitution spatiale alignée sur le format Pays_Aero_Pays_Aero
    routes_split = df['route'].str.split('_', expand=True)
    
    # Extraction stricte des codes ICAO (index 1 = Départ, index 3 = Arrivée)
    # Ex: "DE_EDDB_FR_LFPO" -> From: "EDDB", To: "LFPO"
    df['From'] = routes_split[1]
    df['To'] = routes_split[3]
    
    airports = pd.concat([df['From'], df['To']]).unique()
    airport_to_idx = {apt: i for i, apt in enumerate(airports)}
    df['Origin_Idx'] = df['From'].map(airport_to_idx)
    df['Dest_Idx'] = df['To'].map(airport_to_idx)
    
    # 2. Sécurisation des variables temporelles (au cas où elles manqueraient)
    if 'jour_semaine' not in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['jour_semaine'] = df['date'].dt.dayofweek
        df['mois'] = df['date'].dt.month

    # 3. Sélection stricte des caractéristiques incluant les covariables (Vacances/Évènements)
    features = [
        'Origin_Idx', 'Dest_Idx', 
        'jour_semaine', 'mois', 
        'evenement_depart', 'evenement_arrivee', 
        'vacances_depart', 'vacances_arrivee'
    ]
    
    # La cible originelle est le volume journalier consolidé
    target = 'value_jour'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entraînement de l'arbre Gradient Boosting avec covariables...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        device='cuda' # Utilise le GPU. Remplacer par 'cpu' si erreur matérielle.
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Précision Terminale - RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    
    model.save_model(MODEL_DIR / "xgboost_demand_model.json")
    joblib.dump(airport_to_idx, MODEL_DIR / "xgb_airport_mapping.pkl")
    print("Modèle XGBoost et Mapping sauvegardés avec succès.")

if __name__ == "__main__":
    train_xgb_model()