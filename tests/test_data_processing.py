import pytest
import pandas as pd
from src.Fleet_Assignment_Problem.data.loader import load_and_melt_data

def test_load_and_melt_pipeline(tmp_path):
    # 1. Création fausse donnée (Wide)
    data = {
        'freq,unit,tra_meas,airp_pr\\TIME_PERIOD': [
            'M,PAS,PAS_CRD,FR_LFPG_BE_BRU', 
            'A,PAS,PAS_CRD,FR_LFPG_BE_BRU'
        ],
        '2023':    [10000, 50000],
        '2023-Q1': [2500, 12000],
        '2023-01': [100, 500],
        '2023-02': [':', 600]
    }
    
    # 2. Écriture fichier temporaire
    d_file = tmp_path / "fake_data.tsv.gz"
    pd.DataFrame(data).to_csv(d_file, sep='\t', compression='gzip', index=False)

    # 3. Exécution Pipeline
    df_result = load_and_melt_data(str(d_file))

    # 4. Vérifications
    assert len(df_result) == 1 
    row = df_result.iloc[0]
    assert row['period'] == '2023-01'
    assert row['value'] == 100.0
    assert row['route'] == 'FR_LFPG_BE_BRU'