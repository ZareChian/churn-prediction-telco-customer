# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:14:22 2025

@author: Mohammad Reza Zarechian
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

# Load data and data exploration
df = pd.read_csv(r"C:\Users\mreza\Downloads\archive\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("First 5 rows:\n")
print(df.head())
print("\nData Info:\n")
df.info()
print("\nChurn Distribution:\n", df['Churn'].value_counts(normalize=True))

plt.bar(df['Churn'].unique(), df['Churn'].value_counts(), color=['skyblue', 'salmon'])
plt.title("Class Distribution (Churn vs Non-Churn)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# Data Preprocessing

# Convert TotalCharges to numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()  

# Numeric Features Visualization
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numeric_features].hist(bins=30, figsize=(10, 4), layout=(1, 3))
plt.suptitle("Distributions of Numeric Features")
plt.tight_layout()
plt.show()

# Categorical Features Visualization
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
for feature in categorical_features:
    counts = df.groupby(feature)['Churn'].value_counts().unstack()
    plt.figure(figsize=(10, 4))
    plt.bar(counts.index, counts['No'], label='No')
    plt.bar(counts.index, counts['Yes'], bottom=counts['No'], label='Yes')
    plt.title(f"Churn Rate by {feature}")
    plt.ylabel("Numbers")
    plt.xlabel(feature)
    plt.xticks(rotation=45)
    plt.legend(title='Churn')
    plt.tight_layout() 
    plt.show()

# Prepare Data for ML
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}) # Convert target variable to binary

# Select features for ML (some excluded accordinf to plots)
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features_ml = ['SeniorCitizen', 'Dependents',
                       'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaperlessBilling', 'PaymentMethod'] 
                                                                       
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=float)

# Split features and target & Train-Test Split & Scale numeric features
X = df_encoded.drop(['Churn', 'customerID'], axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("\nPreprocessed Data Shape:", X_train.shape)

# Hyperparameter Tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10],      # Regularization strength
    'penalty': ['l1', 'l2']        # Regularization type
}

grid_search = GridSearchCV(
    LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
    param_grid, 
    cv=5,                          # 5-fold cross-validation
    scoring='roc_auc'              # Optimize for ROC-AUC score
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Use the best model
best_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': best_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))