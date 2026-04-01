# Customer Churn Prediction and Business Insights

# Business Problem

Customer churn is a critical challenge for telecom companies because losing customers directly impacts revenue. Identifying customers who are likely to churn allows companies to take proactive retention actions and reduce customer loss.

This project builds an end-to-end data analytics and machine learning workflow to predict churn and generate business insights that support retention strategies.

# Project Objective

Predict customers who are likely to churn
Identify key drivers influencing churn behavior
Provide actionable business insights to improve retention strategies

# Dataset
The dataset contains telecom customer information including:

Customer demographics
Contract type
Service subscriptions
Tenure with the company
Monthly and total charges
These attributes help analyze patterns that influence churn behavior.

# Project Structure
Customer-Churn-Prediction-and-Business-Insights
│
├── data
│   └── churn.csv
│
├── notebooks
│   └── customer_churn_analysis.ipynb
│
├── visuals
│   └── saved charts and evaluation plots
│
├── customer_churn_project.py
├── requirements.txt

# Key Steps in the Analysis

# 1. Data Preparation

Loaded and inspected dataset
Converted TotalCharges to numeric format
Handled missing values
Removed irrelevant columns such as customer ID

# 2. Exploratory Data Analysis
Visualizations were created to understand churn patterns:

Churn distribution
Monthly Charges vs Churn
Tenure vs Churn
Feature correlation heatmap

# 3. Data Preprocessing

One-hot encoding for categorical variables
Train-test split for model validation

# 4. Machine Learning Models
Two classification models were implemented:
Logistic Regression
Random Forest

# Model Evaluation

Models were evaluated using:
Accuracy score
Confusion Matrix
Classification Report (precision, recall, F1-score)
The best-performing model was exported for future deployment using joblib.

# Key Business Insights

Analysis of the dataset revealed several patterns related to customer churn:

Customers with month-to-month contracts are more likely to churn
Higher monthly charges increase the probability of churn
Customers with short tenure show higher churn rates
Customers without additional services show increased churn risk

# Business Recommendations

Based on the insights generated:
Offer incentives for customers to switch to long-term contracts
Provide retention offers for high monthly charge customers
Improve onboarding strategies for new customers
Promote bundled services to improve customer engagement

# Tools & Technologies

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

# Setup

Create a virtual environment and install dependencies:

pip install -r requirements.txt

# Run the Project

python customer_churn_project.py

# Output

Running the script generates:
Visualizations saved in visuals/
Confusion matrix and model evaluation charts
Best performing churn prediction model saved as:

models/best_churn_model.joblib

# Future Improvements
Build an interactive Power BI dashboard
Deploy the model with Streamlit or Flask
Implement real-time churn prediction pipeline

