# src/train_baseline.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from joblib import dump

from src.data_loader import load_bank_data
from src.preprocess import build_preprocess, TARGET_COL
from src.metrics_utils import evaluate_proba

RANDOM_STATE = 42

def main():
    data_path = Path("data/bank-full.csv")  # đổi tên nếu khác
    out_models = Path("models")
    out_reports = Path("reports")

    df = load_bank_data(data_path)
    # chuẩn hoá nhãn: 'yes' -> 1, 'no' -> 0
    if df[TARGET_COL].dtype == 'object':
        df[TARGET_COL] = (df[TARGET_COL].str.lower() == 'yes').astype(int)

    # tách X, y và build preprocessor
    df, preprocessor = build_preprocess(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    # split 60/20/20 stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    # baseline LR
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=None,          # dùng liblinear/lbfgs không cần n_jobs
        solver="lbfgs"        # nếu nhiều features từ one-hot, lbfgs thường ổn
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", lr)
    ])

    # train trên train; kiểm tra valid để snapshot baseline
    pipe.fit(X_train, y_train)
    yv_proba = pipe.predict_proba(X_valid)[:, 1]
    eval_valid, cm_valid = evaluate_proba(y_valid, yv_proba, outdir=out_reports, prefix="baseline_valid")
    print("VALID metrics:\n", eval_valid)
    print("VALID confusion matrix:\n", cm_valid)

    # cuối cùng chấm trên test
    yt_proba = pipe.predict_proba(X_test)[:, 1]
    eval_test, cm_test = evaluate_proba(y_test, yt_proba, outdir=out_reports, prefix="baseline_test")
    print("TEST metrics:\n", eval_test)
    print("TEST confusion matrix:\n", cm_test)

    # lưu model
    out_models.mkdir(parents=True, exist_ok=True)
    dump(pipe, out_models / "baseline_logreg.joblib")
    print("Saved model to models/baseline_logreg.joblib")

if __name__ == "__main__":
    main()
