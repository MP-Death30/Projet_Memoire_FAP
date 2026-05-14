import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.Fleet_Assignment_Problem.models.model_lstm import prepare_data

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "lstm_multi_input.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler_target.pkl"

def evaluate_lstm():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    _, _, _, X_seq_v, X_exo_v, y_v, _, _ = prepare_data()

    preds_scaled = model.predict([X_seq_v, X_exo_v], verbose=0)

    preds = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_v)

    print("--- MÉTRIQUES GLOBALES (Horizon 7 jours) ---")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, preds)):.2f}")
    print(f"MAE  : {mean_absolute_error(y_true, preds):.2f}\n")

    # t+1 correspond à l'indice 0 de l'horizon de prédiction
    y_true_t1, preds_t1 = y_true[:, 0], preds[:, 0]
    print("--- MÉTRIQUES t+1 (Lendemain) ---")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true_t1, preds_t1)):.2f}")
    print(f"MAE  : {mean_absolute_error(y_true_t1, preds_t1):.2f}\n")

    # t+7 correspond à l'indice 6 de l'horizon de prédiction
    y_true_t7, preds_t7 = y_true[:, 6], preds[:, 6]
    print("--- MÉTRIQUES t+7 (Fin de semaine) ---")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true_t7, preds_t7)):.2f}")
    print(f"MAE  : {mean_absolute_error(y_true_t7, preds_t7):.2f}")

    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(y_true_t1[:150], label="Réel t+1", color="black")
    plt.plot(preds_t1[:150], label="Prédiction t+1", color="red", linestyle="--")
    plt.title("Prédiction à t+1")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y_true_t7[:150], label="Réel t+7", color="black")
    plt.plot(preds_t7[:150], label="Prédiction t+7", color="blue", linestyle="--")
    plt.title("Prédiction à t+7")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_lstm()