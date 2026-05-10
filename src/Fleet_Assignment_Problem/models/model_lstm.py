import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Stratégie validée : {len(gpus)} GPU(s) détecté(s). Calcul accéléré.")
else:
    print("Avertissement : Aucun GPU détecté. Repli sur le CPU. Les temps de calcul exploseront lors du passage à l'échelle.")


# Configuration
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
SEQ_LEN = 21

def prepare_data():
    df = pd.read_parquet(INPUT_FILE)
    
    # Séparation des caractéristiques
    col_target = 'value_jour'
    cols_exo = [
        'evenement_depart', 'vacances_depart', 
        'evenement_arrivee', 'vacances_arrivee', 
        'jour_semaine', 'mois'
    ]
    
    # Normalisation : Le scaler doit être conservé pour l'inversion des prédictions
    scaler = MinMaxScaler()
    df[col_target] = scaler.fit_transform(df[[col_target]])
    
    X_seq_list, X_exo_list, y_list = [], [], []
    
    # Génération des séquences par route isolée
    for route, group in df.groupby('route'):
        group = group.sort_values('date').reset_index(drop=True)
        
        vals_target = group[col_target].values
        vals_exo = group[cols_exo].values
        
        for i in range(len(group) - SEQ_LEN):
            # Séquence de T-SEQ_LEN à T (inclus)
            seq = vals_target[i : i + SEQ_LEN].reshape(-1, 1)
            
            # Variables exogènes à T+1 (Jour de la prédiction)
            exo = vals_exo[i + SEQ_LEN]
            
            # Cible à T+1
            target = vals_target[i + SEQ_LEN]
            
            X_seq_list.append(seq)
            X_exo_list.append(exo)
            y_list.append(target)
            
    X_seq = np.array(X_seq_list)
    X_exo = np.array(X_exo_list)
    y = np.array(y_list)
    
    return X_seq, X_exo, y, scaler, len(cols_exo)

def build_model(seq_len, exo_dim):
    # Branche 1 : Historique (LSTM)
    input_seq = Input(shape=(seq_len, 1), name="past_sequence")
    lstm_out = LSTM(64, return_sequences=False)(input_seq)
    lstm_out = Dropout(0.4)(lstm_out) # Régularisation agressive pour dataset court
    
    # Branche 2 : Contexte futur (Dense)
    input_exo = Input(shape=(exo_dim,), name="future_context")
    exo_out = Dense(16, activation='relu')(input_exo)
    
    # Fusion
    merged = Concatenate()([lstm_out, exo_out])
    dense_1 = Dense(32, activation='relu')(merged)
    dense_1 = Dropout(0.3)(dense_1)
    output = Dense(1, activation='linear')(dense_1)
    
    model = Model(inputs=[input_seq, input_exo], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train():
    X_seq, X_exo, y, scaler, exo_dim = prepare_data()
    
    # Split chronologique (Pas de mélange aléatoire sur séries temporelles)
    split_idx = int(len(X_seq) * 0.8)
    
    X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
    X_exo_train, X_exo_val = X_exo[:split_idx], X_exo[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = build_model(SEQ_LEN, exo_dim)
    
    # Mécanisme de prévention du surapprentissage
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    
    model.fit(
        x=[X_seq_train, X_exo_train], 
        y=y_train,
        validation_data=([X_seq_val, X_exo_val], y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Sauvegarde (Le scaler doit également être exporté via joblib/pickle en production)
    model.save(BASE_DIR / "models" / "lstm_multi_input.keras")

if __name__ == "__main__":
    train()