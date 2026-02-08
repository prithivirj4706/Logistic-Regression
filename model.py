import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv('framingham_heart_disease (1).csv')

# Data Cleaning
df = df.drop(['education'], axis=1)

# Handling missing values by replacing with mean
cigsPerDay_mean = round(df["cigsPerDay"].mean())
BPMeds_mean = round(df["BPMeds"].mean())
totChol_mean = round(df["totChol"].mean())
BMI_mean = round(df["BMI"].mean())
heartRate_mean = round(df["heartRate"].mean())
glucose_mean = round(df["glucose"].mean())

df['cigsPerDay'] = df['cigsPerDay'].fillna(cigsPerDay_mean)
df['BPMeds'] = df['BPMeds'].fillna(BPMeds_mean)
df['totChol'] = df['totChol'].fillna(totChol_mean)
df['BMI'] = df['BMI'].fillna(BMI_mean)
df['heartRate'] = df['heartRate'].fillna(heartRate_mean)
df['glucose'] = df['glucose'].fillna(glucose_mean)

# Splitting features and target
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Train/Test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling the data (Recommended for Logistic Regression and provides the 2nd pickle file)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Saving the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and Scaler saved successfully as model.pkl and scaler.pkl")
