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
HORIZON = 7

def prepare_data():
    df = pd.read_parquet(INPUT_FILE)
    col_target = 'value_jour'
    
    df['jour_sin'] = np.sin(2 * np.pi * df['jour_semaine'] / 7.0)
    df['jour_cos'] = np.cos(2 * np.pi * df['jour_semaine'] / 7.0)
    df['mois_sin'] = np.sin(2 * np.pi * df['mois'] / 12.0)
    df['mois_cos'] = np.cos(2 * np.pi * df['mois'] / 12.0)
    
    cols_exo = ['evenement_depart', 'vacances_depart', 'evenement_arrivee', 'vacances_arrivee', 'jour_sin', 'jour_cos', 'mois_sin', 'mois_cos']
    
    # Division temporelle avant ajustement statistique
    df['date'] = pd.to_datetime(df['date'])
    split_date = df['date'].quantile(0.8)
    
    scaler_target = MinMaxScaler()
    train_mask = df['date'] < split_date
    scaler_target.fit(df.loc[train_mask, [col_target]])
    df[col_target] = scaler_target.transform(df[[col_target]])
    
    X_seq_list, X_exo_list, y_list, is_val_list = [], [], [], []
    
    for route, group in df.groupby('route'):
        group = group.sort_values('date').reset_index(drop=True)
        vals_target = group[col_target].values
        vals_exo = group[cols_exo].values
        dates = group['date'].values
        
        for i in range(len(group) - SEQ_LEN - HORIZON + 1):
            X_seq_list.append(vals_target[i : i + SEQ_LEN].reshape(-1, 1))
            X_exo_list.append(vals_exo[i + SEQ_LEN : i + SEQ_LEN + HORIZON].flatten())
            y_list.append(vals_target[i + SEQ_LEN : i + SEQ_LEN + HORIZON])
            is_val_list.append(dates[i + SEQ_LEN] >= np.datetime64(split_date))
            
    X_seq = np.array(X_seq_list)
    X_exo = np.array(X_exo_list)
    y = np.array(y_list)
    is_val = np.array(is_val_list)
    
    X_seq_train, X_seq_val = X_seq[~is_val], X_seq[is_val]
    X_exo_train, X_exo_val = X_exo[~is_val], X_exo[is_val]
    y_train, y_val = y[~is_val], y[is_val]
    
    return X_seq_train, X_exo_train, y_train, X_seq_val, X_exo_val, y_val, scaler_target, len(cols_exo)

def build_model(seq_len, exo_dim, horizon):
    input_seq = Input(shape=(seq_len, 1), name="past_sequence")
    lstm_out = Dropout(0.4)(LSTM(64, return_sequences=False)(input_seq))
    
    input_exo = Input(shape=(exo_dim * horizon,), name="future_context")
    exo_out = Dense(32, activation='relu')(input_exo)
    
    dense_1 = Dropout(0.3)(Dense(64, activation='relu')(Concatenate()([lstm_out, exo_out])))
    output = Dense(horizon, activation='linear', name="multi_step_output")(dense_1)
    
    model = Model(inputs=[input_seq, input_exo], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def train():
    X_seq_t, X_exo_t, y_t, X_seq_v, X_exo_v, y_v, scaler, exo_dim = prepare_data()
    model = build_model(SEQ_LEN, exo_dim, HORIZON)
    
    model.fit(
        x=[X_seq_t, X_exo_t], y=y_t,
        validation_data=([X_seq_v, X_exo_v], y_v),
        epochs=100, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
        verbose=1
    )
    
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "lstm_multi_input.keras")
    joblib.dump(scaler, model_dir / "scaler_target.pkl")

if __name__ == "__main__":
    train()