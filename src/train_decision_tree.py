from pathlib import Path
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

from data_loader import load_bank_data  # dùng hàm bạn đã có sẵn


# =====================================
# 1. Load & prepare data
# =====================================
def load_and_split_data():
    csv_path = Path("data/bank-full.csv")
    df = load_bank_data(csv_path)

    # Bỏ cột duration (vì không biết trước khi gọi khách hàng)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Chuẩn hoá nhãn
    y = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])

    # Phân loại kiểu dữ liệu
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    num_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Tách tập train/valid/test (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test, num_features, cat_features


# =====================================
# 2. Build preprocessing pipeline
# (Scaler không bắt buộc với Tree, nhưng giữ nguyên cho đồng nhất pipeline)
# =====================================
def build_preprocess(num_features, cat_features):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])
    return preprocessor


# =====================================
# 3A. Train Decision Tree (fixed params) – optional
# =====================================
def train_decision_tree(X_train, y_train, preprocessor):
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=50,
        class_weight="balanced",
        random_state=42
    )
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# =====================================
# 3B. Grid Search để tune hyperparameters (khuyên dùng)
# =====================================
def train_decision_tree_grid(X_train, y_train, preprocessor):
    base_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", base_model)
    ])

    param_grid = {
        "model__max_depth": [4, 6, 8, 10, 12],
        "model__min_samples_split": [10, 30, 50, 100]
    }

    # đa mục tiêu: log thêm f1/recall, nhưng chọn mô hình theo AUC-ROC
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=5,
        scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"},
        refit="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    print("🔍 Running GridSearchCV...")
    grid.fit(X_train, y_train)
    print("Done. Best params:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Lưu bảng kết quả grid để report
    Path("reports").mkdir(exist_ok=True)
    import pandas as pd
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_df.to_csv("reports/dt_grid_results.csv", index=False)
    print("Saved grid results to reports/dt_grid_results.csv")

    return grid.best_estimator_  # pipeline tốt nhất


# =====================================
# 4. Evaluate & visualize
# =====================================
def evaluate_model(pipeline, X, y, prefix="valid"):
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y, y_proba)
    ap = average_precision_score(y, y_proba)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    print(f"\n[{prefix.upper()}] ROC-AUC = {auc:.4f}, PR-AUC = {ap:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve ({prefix}) - AUC={auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    Path("reports").mkdir(exist_ok=True)
    plt.savefig(f"reports/decision_tree_{prefix}_roc.png")
    plt.close()

    # PR Curve
    prec, rec, _ = precision_recall_curve(y, y_proba)
    plt.plot(rec, prec)
    plt.title(f"Precision-Recall Curve ({prefix}) - AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(f"reports/decision_tree_{prefix}_pr.png")
    plt.close()


# =====================================
# 5. Main
# =====================================
def main(use_grid=True):
    X_train, X_valid, X_test, y_train, y_valid, y_test, num_f, cat_f = load_and_split_data()
    preprocess = build_preprocess(num_f, cat_f)

    if use_grid:
        pipeline = train_decision_tree_grid(X_train, y_train, preprocess)
        model_path = "models/decision_tree_best.joblib"
    else:
        pipeline = train_decision_tree(X_train, y_train, preprocess)
        model_path = "models/decision_tree.joblib"

    evaluate_model(pipeline, X_valid, y_valid, prefix="valid")
    evaluate_model(pipeline, X_test, y_test, prefix="test")

    Path("models").mkdir(exist_ok=True)
    dump(pipeline, model_path)
    print(f"\nModel saved at {model_path}")


if __name__ == "__main__":
    # True: chạy Grid Search; False: train nhanh với tham số cố định
    main(use_grid=True)
