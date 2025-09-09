# src/anomaly_detection.py
import numpy as np  # ← ADD THIS LINE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import joblib
import pandas as pd

def detect_anomalies(X_scaled, contamination=0.05):
    """
    Detect anomalies in delivery data
    """
    # Train Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)
    
    # Predict anomalies
    anomalies = model.predict(X_scaled)
    
    # Save model
    joblib.dump(model, 'models/anomaly_detection_model.pkl')
    
    return anomalies

def analyze_anomalies(df, anomalies):
    """
    Analyze and summarize detected anomalies
    """
    # Add anomaly labels to dataframe
    df['is_anomaly'] = anomalies
    
    # Analyze by city
    anomaly_by_city = df.groupby('from_city_name')['is_anomaly'].mean().reset_index()
    anomaly_by_city.columns = ['City', 'Anomaly Rate']
    
    # Analyze by service type
    anomaly_by_type = df.groupby('typecode')['is_anomaly'].mean().reset_index()
    anomaly_by_type.columns = ['Service Type', 'Anomaly Rate']
    
    return df, anomaly_by_city, anomaly_by_type