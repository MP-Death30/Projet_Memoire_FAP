import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib


BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset_lstm.parquet"
SEQ_LEN = 21

def prepare_data():
    df = pd.read_parquet(INPUT_FILE)
    
    col_target = 'value_jour'
    
    # Encodage cyclique trigonométrique
    df['jour_sin'] = np.sin(2 * np.pi * df['jour_semaine'] / 7.0)
    df['jour_cos'] = np.cos(2 * np.pi * df['jour_semaine'] / 7.0)
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12.0)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12.0)
    
    cols_exo = [
        'evenement_depart', 'vacances_depart', 
        'evenement_arrivee', 'vacances_arrivee', 
        'jour_sin', 'jour_cos', 'mois_sin', 'mois_cos'
    ]
    
    scaler_target = MinMaxScaler()
    df[col_target] = scaler_target.fit_transform(df[[col_target]])
    
    X_seq_list, X_exo_list, y_list = [], [], []
    
    for route, group in df.groupby('route'):
        group = group.sort_values('date').reset_index(drop=True)
        
        vals_target = group[col_target].values
        vals_exo = group[cols_exo].values
        
        for i in range(len(group) - SEQ_LEN):
            seq = vals_target[i : i + SEQ_LEN].reshape(-1, 1)
            exo = vals_exo[i + SEQ_LEN]
            target = vals_target[i + SEQ_LEN]
            
            X_seq_list.append(seq)
            X_exo_list.append(exo)
            y_list.append(target)
            
    X_seq = np.array(X_seq_list)
    X_exo = np.array(X_exo_list)
    y = np.array(y_list)
    
    return X_seq, X_exo, y, scaler_target, len(cols_exo)

def build_model(seq_len, exo_dim):
    input_seq = Input(shape=(seq_len, 1), name="past_sequence")
    lstm_out = LSTM(64, return_sequences=False)(input_seq)
    lstm_out = Dropout(0.4)(lstm_out)
    
    input_exo = Input(shape=(exo_dim,), name="future_context")
    exo_out = Dense(16, activation='relu')(input_exo)
    
    merged = Concatenate()([lstm_out, exo_out])
    dense_1 = Dense(32, activation='relu')(merged)
    dense_1 = Dropout(0.3)(dense_1)
    output = Dense(1, activation='linear')(dense_1)
    
    model = Model(inputs=[input_seq, input_exo], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train():
    X_seq, X_exo, y, scaler_target, exo_dim = prepare_data()
    
    split_idx = int(len(X_seq) * 0.8)
    
    X_seq_train, X_seq_val = X_seq[:split_idx], X_seq[split_idx:]
    X_exo_train, X_exo_val = X_exo[:split_idx], X_exo[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    model = build_model(SEQ_LEN, exo_dim)
    
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
    
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(model_dir / "lstm_multi_input.keras")
    joblib.dump(scaler_target, model_dir / "scaler_target.pkl")

if __name__ == "__main__":
    train()