import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, db

# Firebase init (using your custom Firebase URL and credential file)
FIREBASE_DB_URL = 'https://testingphase1-7b880-default-rtdb.firebaseio.com'

# Initialize Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate("testingphase1-7b880-firebase-adminsdk-fbsvc-c1c4977f14.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': FIREBASE_DB_URL
    })

# Utility Functions
@st.cache_resource
def load_models():
    try:
        with open("rnn_models.pkl", "rb") as f:
            data = pickle.load(f)
        return data["models"], data["scalers"]
    except FileNotFoundError:
        return {}, {}

def fetch_data():
    ref = db.reference("FinalSensorData")
    all_pumps = ref.get()
    data = []
    for pump_id in all_pumps or {}:
        for record_id, record in all_pumps[pump_id].items():
            try:
                timestamp = datetime.strptime(record.get("timestamp", ""), "%d_%m_%Y_%H_%M_%S")
            except Exception:
                continue
            data.append({
                "flowRate": record.get("flowRate", 0),
                "current": record.get("current", 0),
                "timestamp": timestamp
            })
    
    df = pd.DataFrame(data)
    return df.sort_values("timestamp") if not df.empty else pd.DataFrame(columns=["flowRate", "current"])

def prepare_data(series, time_steps=7):
    X, y = [], []
    for i in range(len(series) - time_steps):
        X.append(series[i:i+time_steps])
        y.append(series[i+time_steps])
    return np.array(X), np.array(y)

def predict_future(model, scaler, last_seq, steps):
    preds, seq = [], last_seq.copy()
    for _ in range(steps):
        x = scaler.transform(seq.reshape(-1, 1)).reshape(1, 7, 1)
        pred = model.predict(x, verbose=0)
        inv_pred = scaler.inverse_transform(pred)[0][0]
        preds.append(float(inv_pred))
        seq = np.roll(seq, -1)
        seq[-1] = pred[0][0]
    return preds

# Load models
models, scalers = load_models()

# Streamlit UI
st.title("Pump Data Prediction Dashboard")

menu = st.sidebar.selectbox("Menu", ["Train Model", "Predict"])

df = fetch_data()
if df.empty:
    st.warning("No data found in Firebase.")
    st.stop()

if menu == "Train Model":
    st.subheader("Train RNN Models")
    if st.button("Train"):
        for metric in ["flowRate", "current"]:
            scaler = MinMaxScaler()
            df[metric] = scaler.fit_transform(df[[metric]])
            scalers[metric] = scaler

            X, y = prepare_data(df[metric].values, 7)
            X = X.reshape(X.shape[0], 7, 1)
            model = Sequential([
                SimpleRNN(50, activation="relu", return_sequences=True, input_shape=(7, 1)),
                SimpleRNN(50, activation="relu"),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=50, batch_size=8, verbose=0)
            models[metric] = model

        with open("rnn_models.pkl", "wb") as f:
            pickle.dump({"models": models, "scalers": scalers}, f)
        st.success("Models trained and saved!")

elif menu == "Predict":
    st.subheader("Predict Future Values")
    metric = st.selectbox("Select Metric", ["flowRate", "current"])
    steps = st.slider("Steps to Predict", 1, 50, 10)

    if models.get(metric):
        last_seq = df[metric].values[-7:]
        preds = predict_future(models[metric], scalers[metric], last_seq, steps)
        st.line_chart(preds)
        st.write(preds)
    else:
        st.warning("Model not trained yet.")
