import streamlit as st
from utils import check_credentials, save_model, load_model, log_prediction, load_logs
from federated_learning import train_local_model, aggregate_models
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="Smart Crime Prediction", layout="wide")

REQUIRED_COLUMNS = ["location", "time", "day", "previous_crime", "label"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ------------------------------
# Login Page
# ------------------------------
def login():
    st.title("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid credentials")
