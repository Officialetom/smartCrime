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

# ------------------------------
# Dashboard Page
# ------------------------------
def dashboard():
    st.title("📊 Dashboard")
    logs = load_logs()
    st.metric("Total Predictions", len(logs))
    st.metric("Unique Stations", logs["Station"].nunique())
    st.metric("Avg. Risk", logs["RiskLevel"].mean() if not logs.empty else 0)

# ------------------------------
# Train Model Page
# ------------------------------
def train_model():
    st.title("📤 Train Federated Model")
    uploaded_files = st.file_uploader("Upload Station Training CSVs", type="csv", accept_multiple_files=True)
    if uploaded_files and st.button("Train Federated Model"):
        models = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in REQUIRED_COLUMNS):
                st.error(f"'{file.name}' is missing required columns. Required: {REQUIRED_COLUMNS}")
                return
            model, acc = train_local_model(df)
            st.write(f"Model from `{file.name}` trained with accuracy: {round(acc, 2)}")
            models.append(model)
        global_model = aggregate_models(models)
        save_model(global_model)
        st.success("Global model saved successfully.")

# ------------------------------
# Predict Crime Page
# ------------------------------
def predict():
    st.title("🔍 Predict Crime")
    model = None
    try:
        model = load_model()
    except:
        st.error("Please train a model first.")
        return

    uploaded = st.file_uploader("Upload Station Data for Prediction", type="csv")
    manual_time = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

    time_map = {"Morning": 1, "Afternoon": 2, "Evening": 3, "Night": 4}
    if uploaded and st.button("Predict"):
        df = pd.read_csv(uploaded)
        if not all(col in df.columns for col in REQUIRED_COLUMNS[:-1]):
            st.error(f"'{uploaded.name}' is missing required columns. Required: {REQUIRED_COLUMNS[:-1]}")
            return
        df["time"] = time_map[manual_time]
        predictions = model.predict(df)
        risk_levels = model.predict_proba(df).max(axis=1)
        for i in range(len(predictions)):
            log_prediction(uploaded.name, predictions[i], round(risk_levels[i], 2))
        st.success("Prediction complete.")
        st.write("Predicted Crime Types:", predictions)
        st.write("Risk Levels:", risk_levels)

# ------------------------------
# Risk Map Page
# ------------------------------
def risk_map():
    st.title("🗺️ Risk Severity Map")
    logs = load_logs()
    if logs.empty:
        st.warning("No prediction logs yet.")
        return
    logs["lat"] = 6 + (logs.index % 10) * 0.1
    logs["lon"] = 3 + (logs.index % 10) * 0.1
    fig = px.scatter_mapbox(logs, lat="lat", lon="lon", color="RiskLevel", size="RiskLevel",
                            mapbox_style="open-street-map", zoom=4, hover_name="Station")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Recent Predictions Page
# ------------------------------
def recent_predictions():
    st.title("📄 Recent Predictions")
    logs = load_logs()
    st.dataframe(logs)

# ------------------------------
# Navigation
# ------------------------------
def navigation():
    pages = {
        "Dashboard": dashboard,
        "Train Model": train_model,
        "Predict": predict,
        "Recent Predictions": recent_predictions,
        "Risk Severity Map": risk_map
    }
    st.sidebar.title("🔧 Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

# ------------------------------
# Run App
# ------------------------------
if not st.session_state.logged_in:
    login()
else:
    navigation()
