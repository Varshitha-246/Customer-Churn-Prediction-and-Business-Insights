# Customer Churn Prediction and Business Insights

## Project Structure

- data/
  - churn.csv
- notebooks/
  - customer_churn_analysis.ipynb
- visuals/
- customer_churn_project.py
- requirements.txt

## Project Goal

Build an end-to-end analytics and machine learning workflow to predict customer churn and extract actionable business insights.

## Features Implemented

- Data loading and inspection (head, shape, info)
- Data cleaning (TotalCharges conversion, missing values, ID drop)
- EDA visualizations (churn distribution, MonthlyCharges vs Churn, tenure vs Churn, correlation heatmap)
- Preprocessing with one-hot encoding and train-test split
- Model training with Logistic Regression and Random Forest
- Evaluation with accuracy, confusion matrix, and classification report
- Feature importance analysis for business insights
- Best model export using joblib

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python customer_churn_project.py
```

## Outputs

Running the script creates:

- visuals/ with saved charts and confusion matrices
- models/best_churn_model.joblib with the best-performing model
