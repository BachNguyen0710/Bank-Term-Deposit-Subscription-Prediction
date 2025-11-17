# Data Mining Assignment – Bank Marketing

Assignment cho môn **Data Mining**.  
Mục tiêu: xây dựng pipeline cơ bản với **scikit-learn**, train Logistic Regression làm baseline và lưu kết quả đánh giá (ROC-AUC, PR-AUC, confusion matrix).

---

## 📂 Cấu trúc thư mục

```text
assignment/
├─ data/            # chứa dataset (CSV)
├─ notebooks/       # Jupyter notebooks cho EDA & baseline
├─ src/             # source code (data_loader, preprocess, train_baseline, metrics_utils, …)
├─ models/          # (tạo sau khi train) lưu .joblib model
├─ reports/         # (tạo sau khi train) lưu metrics.csv, ROC/PR plots
├─ requirements.txt
└─ README.md
```

---

## 🚀 Cách cài đặt & chạy

Bạn có thể chọn 1 trong 2 cách bên dưới:

### **Option A – KHÔNG dùng môi trường ảo (dễ nhất)**
> Dùng khi bạn chỉ có 1 project Python trên máy.

1. Cài dependencies trực tiếp:
   ```bash```
   ```pip install -r requirements.txt```

2. Kiểm tra phiên bản scikit-learn:

    ```python -c "import sklearn; print(sklearn.__version__)"```

3. Train baseline model:
    ```python -m src.train_baseline```

### **Option B – Dùng môi trường ảo (venv)**

1. Tạo & kích hoạt venv:

    ```python -m venv .venv```
    ```source .venv/bin/activate     # macOS/Linux```
    ```.venv\Scripts\activate        # Windows```


2. Cài dependencies:

    ```pip install -r requirements.txt```


3. Train baseline model:

    ```python -m src.train_baseline```


4. Thoát venv khi xong:

    ```deactivate```

---

### **▶️ Cách chạy file (3 lựa chọn)**
1. Cách 1 — Chạy ở chế độ module (khuyến nghị)

Chạy từ thư mục gốc assignment/ (nơi có thư mục src/):

    ```python -m src.train_baseline```

Nếu máy bạn có nhiều Python, có thể chỉ định đường dẫn interpreter rõ ràng, ví dụ macOS (Python.org):

    ```/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m src.train_baseline```

2. Cách 2 — Chạy trực tiếp file không sửa code

Dùng biến môi trường PYTHONPATH để Python “nhìn thấy” package src:

macOS/Linux:

```PYTHONPATH=. python src/train_baseline.py```


Windows (PowerShell):

```$env:PYTHONPATH="."; python src/train_baseline.py```


Windows (CMD):

```set PYTHONPATH=.```
```python src\train_baseline.py```

3. Cách 3 — Chạy trực tiếp file có thêm 4 dòng bootstrap (giải pháp code-side)

Thêm 4 dòng sau lên đầu file src/train_baseline.py, rồi chạy python src/train_baseline.py bình thường:

```
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))```

🧪 (Tuỳ chọn) Chạy bằng Notebook
jupyter notebook notebooks/02_baseline_lr.ipynb


Notebook đã có: load dữ liệu, EDA ngắn, pipeline, train LR, đánh giá (ROC/PR), lưu metrics/model.
