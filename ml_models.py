# Machine Learning Models for Patel Brothers BI Project

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("PatelBrothers.csv")  # replace with actual dataset filename

# ----------------------------
# 1. Sales Forecasting (Regression)
# ----------------------------
X = data[['Month', 'Year']]  # Example predictors
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Sales Forecasting RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----------------------------
# 2. Customer Segmentation (KMeans)
# ----------------------------
features = data[['Age', 'AnnualIncome', 'SpendingScore']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled)
data['Cluster'] = clusters
print("Customer clusters created successfully.")

# ----------------------------
# 3. Customer Attrition (Classification)
# ----------------------------
X = data[['Purchases', 'Visits', 'SpendingScore']]  # Example predictors
y = data['Churn']  # Binary column 0/1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Churn Model Accuracy:", accuracy_score(y_test, y_pred))
