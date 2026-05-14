import matplotlib.pyplot as plt
import pandas as pd
from src.Fleet_Assignment_Problem.operations.inference_ppo import generate_ppo_demand_state

def plot_horizon_prevision(route="FR_LFPO_FR_LFMN", start_date="2025-01-01"):
    df_pred, _, _ = generate_ppo_demand_state(route, start_date)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_pred['date'], df_pred['predicted_demand'], marker='o', color='navy', linewidth=2)
    
    # Annotations des jours de la semaine pour lecture de la saisonnalité
    jours_semaine = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    labels = [f"{d.strftime('%Y-%m-%d')}\n({jours_semaine[d.dayofweek]})" for d in df_pred['date']]
    
    plt.title(f"Dynamique de la demande intra-hebdomadaire : {route}", fontweight='bold')
    plt.ylabel("Volume passagers prédit")
    plt.xticks(df_pred['date'], labels, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_horizon_prevision("FR_LFPO_FR_LFMN", "2025-01-01")