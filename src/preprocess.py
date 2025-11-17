# src/preprocess.py
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLS = ["duration"]  # theo outline
TARGET_COL = "y"          # 'yes'/'no'

def split_feature_types(df: pd.DataFrame, target_col: str = TARGET_COL) -> Tuple[List[str], List[str]]:
    features = [c for c in df.columns if c != target_col]
    categorical = [c for c in features if df[c].dtype == 'object']
    numeric = [c for c in features if c not in categorical]
    return numeric, categorical

def build_preprocess(df: pd.DataFrame):
    df = df.copy()
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    num_cols, cat_cols = split_feature_types(df)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return df, preprocessor
