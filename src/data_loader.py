# src/data_loader.py
from pathlib import Path
import pandas as pd 

def load_bank_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';') if str(csv_path).endswith('.csv') else pd.read_csv(csv_path)
    # Chuẩn hoá tên cột (tuỳ chọn)
    df.columns = [c.strip().lower().replace('.', '_') for c in df.columns]
    return df
