# churn-prediction-telco-customer
A machine learning project that predicts customer churn for a telecommunications company using logistic regression.

# Project Overview
This project analyzes customer data to predict which customers are likely to churn using python. It includes:
- Exploratory Data Analysis (EDA) with visualizations
- Data Preprocessing
- Machine Learning Model (Logistic Regression)
- Model Evaluation with comprehensive metrics

# Dataset

This project uses the Telco Customer Churn dataset from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
About the dataset:
- 7,043 customers with 21 features
- Target variable: whether the customer churned or not

# Machine Learning Pipeline
- Data cleaning and preprocessing
- One-hot encoding for categorical variables
- Feature scaling (StandardScaler)
- Logistic regression with class balancing
- Comprehensive model evaluation

# Hyperparameter Tuning
- Used GridSearchCV to find optimal model parameters
- Tested different regularization strengths (`C`) and types (`L1`, `L2`)
- 5-fold cross-validation for robust performance evaluation

