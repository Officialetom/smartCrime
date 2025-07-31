import streamlit as st
import hashlib
import os
import pandas as pd
import pickle

# Dummy admin credentials
def check_credentials(username, password):
    return username == "admin" and password == "1234"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def save_model(model):
    with open("models/global_model.pkl", "wb") as f:
        pickle.dump(model, f)

def load_model():
    with open("models/global_model.pkl", "rb") as f:
        return pickle.load(f)

def log_prediction(station, result, risk_level):
    logs_path = "logs/predictions.csv"
    df = pd.DataFrame([[station, result, risk_level]], columns=["Station", "Prediction", "RiskLevel"])
    if os.path.exists(logs_path):
        df.to_csv(logs_path, mode='a', index=False, header=False)
    else:
        df.to_csv(logs_path, index=False)

def load_logs():
    logs_path = "logs/predictions.csv"
    if os.path.exists(logs_path):
        return pd.read_csv(logs_path)
    return pd.DataFrame(columns=["Station", "Prediction", "RiskLevel"])
