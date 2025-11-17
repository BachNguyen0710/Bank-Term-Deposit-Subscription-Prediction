# Data Mining Assignment: Bank Marketing Prediction

This project is for a Data Mining course, focusing on the analysis, construction, and evaluation of machine learning models to predict whether a client will subscribe to a term deposit, based on the "Bank Marketing" dataset.

## Implemented Models

This project implements and compares the following models:

1.  **Logistic Regression (Baseline):** Used as the baseline model for comparison.
2.  **Decision Tree:** A highly interpretable model.
3.  **Random Forest:** An ensemble model to improve accuracy (analyzed in `03_rf_dt.ipynb`).
4.  **K-Means Clustering:** Used to discover potential customer segments (analyzed in `04_Kmeans_clustering.ipynb`).

## How to Run the Project

### 1. Set up the Environment

Ensure you have Python installed. Then, install the required libraries:

```bash
pip install -r requirements.txt

2. View the Analysis
The best way to understand the project is to open and run the Jupyter Notebooks in the notebooks/ directory, in order from 01 to 04.

3. Retrain Models
You can retrain the models by running the scripts in the src/ directory:

Bash

# Train the baseline model
python src/train_baseline.py

# Train the Decision Tree model
python src/train_decision_tree.py
Trained models will be saved in the models/ directory, and reports (metrics, plots) will be saved in the reports/ directory.