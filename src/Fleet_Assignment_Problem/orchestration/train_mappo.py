import torch
import pandas as pd
from pathlib import Path
import tensorflow as tf
import joblib
from src.Fleet_Assignment_Problem.environments.fap_ma_env import FAPParallelEnv
from src.Fleet_Assignment_Problem.orchestration.mappo_trainer import MAPPOTrainer, collect_trajectories
from src.Fleet_Assignment_Problem.operations.generate_schedule import generate_dynamic_schedule
from src.Fleet_Assignment_Problem.operations.inference_ppo import predict_demand_for_schedule

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

def run_training():
    # Restriction VRAM TensorFlow pour PyTorch
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    BASE_DIR = Path(__file__).resolve().parents[3]
    FLEET_FILE = BASE_DIR / "data" / "processed" / "fleet_data.parquet"
    DATA_LSTM_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
    MODEL_DIR = BASE_DIR / "models" / "mappo_fap"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Préchargement strict en mémoire
    lstm_model = tf.keras.models.load_model(BASE_DIR / "models" / "lstm_multi_input.keras")
    lstm_scaler = joblib.load(BASE_DIR / "models" / "scaler_target.pkl")
    df_history = pd.read_parquet(DATA_LSTM_FILE)
    fleet_types_df = pd.read_parquet(FLEET_FILE)

    inventory_map = {'737': 12, 'A320': 8, 'Embraer190': 8}
    physical_fleet = []
    for _, row in fleet_types_df.iterrows():
        count = inventory_map.get(row['fleet_id'], 5)
        for _ in range(count):
            physical_fleet.append({
                'position': 0.0, 'capacity': float(row['capacity']), 
                'speed': float(row['speed_kmh']), 'cost': float(row['cost'])
            })

    trainer = MAPPOTrainer(flight_dim=6, agent_dim=5, embed_dim=128)

    epochs = 1000
    for epoch in range(epochs):
        raw_schedule = generate_dynamic_schedule(len(physical_fleet))
        schedule_df = predict_demand_for_schedule(raw_schedule, lstm_model, lstm_scaler, df_history, "2025-01-01")
        
        airports = pd.concat([schedule_df['From'], schedule_df['To']]).unique()
        num_airports = len(airports)
        max_flights = len(schedule_df)
        
        env = FAPParallelEnv(num_airports=num_airports, max_flights=max_flights)
        
        rollouts = collect_trajectories(env, trainer, physical_fleet, schedule_df, steps=max_flights)
        trainer.train_step(rollouts)
        
        if epoch % 10 == 0:
            avg_reward = torch.stack([r['rewards'] for r in rollouts]).mean().item()
            print(f"Epoch {epoch} | Récompense moyenne : {avg_reward:.2f} | Vols: {max_flights}")

    torch.save(trainer.policy.state_dict(), MODEL_DIR / "mappo_policy.pth")
    print(f"Modèle sauvegardé : {MODEL_DIR / 'mappo_policy.pth'}")

if __name__ == "__main__":
    run_training()