# ml_app.py (ML Related Routes and Helpers)
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
import firebase_admin
from firebase_admin import credentials, db

# Load models and scalers or initialize
try:
    with open("rnn_models.pkl", "rb") as f:
        data = pickle.load(f)
        models = data["models"]
        scalers = data["scalers"]
except FileNotFoundError:
    models = {}
    scalers = {}

# Firebase init (if not already done)
if not firebase_admin._apps:
    cred = credentials.Certificate("testingphase1-7b880-firebase-adminsdk-fbsvc-e9247e0aee.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://testingphase1-7b880-default-rtdb.firebaseio.com/'
    })

app = Flask(__name__)

def prepare_data(series, time_steps=7):
    X, y = [], []
    if len(series) <= time_steps:
        return np.array(X), np.array(y)
    for i in range(len(series) - time_steps):
        X.append(series[i: i + time_steps])
        y.append(series[i + time_steps])
    return np.array(X), np.array(y)

def fetch_and_process_data():
    ref = db.reference("FinalSensorData")
    all_pumps = ref.get()
    data = []
    for pump_id in all_pumps or {}:
        for record in all_pumps[pump_id].values():
            timestamp = pd.to_datetime(record.get("timestamp", "1970-01-01 00:00:00"), format="%d_%m_%Y_%H_%M_%S")
            data.append({
                "flowRate": record.get("flowRate", 0),
                "current": record.get("current", 0),
                "timestamp": timestamp,
            })
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["flowRate", "current"])
    df = df.sort_values("timestamp")
    return df

def predict_future(model, scaler, last_sequence, steps):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        x = current_sequence.reshape(1, 7, 1)
        scaled_x = scaler.transform(x.reshape(7, 1)).reshape(1, 7, 1)
        next_pred = model.predict(scaled_x, verbose=0)
        next_value = scaler.inverse_transform(next_pred)[0][0]
        future_predictions.append(float(next_value))
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0][0]

    return future_predictions

@app.route("/api/train", methods=["GET"])
def train_models():
    global models, scalers
    df = fetch_and_process_data()
    if df.empty or len(df) < 8:
        return jsonify({"error": "Insufficient data for training"}), 400

    for metric in ["flowRate", "current"]:
        scaler = MinMaxScaler()
        df[metric] = scaler.fit_transform(df[[metric]])
        scalers[metric] = scaler

    for metric in ["flowRate", "current"]:
        data_series = df[metric].values
        X, y = prepare_data(data_series, time_steps=7)
        if len(X) == 0:
            return jsonify({"error": f"Insufficient data for {metric} training"}), 400
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential([
            SimpleRNN(50, activation="relu", return_sequences=True, input_shape=(7, 1)),
            SimpleRNN(50, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=50, batch_size=8, verbose=0)
        models[metric] = model

    with open("rnn_models.pkl", "wb") as f:
        pickle.dump({"models": models, "scalers": scalers}, f)

    return jsonify({"message": "Training completed and models updated"}), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    global models, scalers
    data = request.json
    if not data or "metric" not in data or "steps" not in data or not isinstance(data["steps"], int) or data["steps"] <= 0:
        return jsonify({"error": "Invalid input data. Provide 'metric' (flowRate or current) and 'steps' (positive integer)"}), 400

    metric = data["metric"]
    steps = data["steps"]
    if metric not in models:
        return jsonify({"error": f"No model trained for {metric}"}), 400

    df = fetch_and_process_data()
    if df.empty or len(df) < 7:
        return jsonify({"error": "Insufficient data in database for prediction"}), 400

    series = df[metric].values[-7:]
    last_sequence = series.reshape(1, 7)
    scaler = scalers.get(metric, MinMaxScaler())
    scaled_sequence = scaler.transform(last_sequence.reshape(7, 1)).reshape(1, 7, 1)

    predictions = predict_future(models[metric], scaler, scaled_sequence[0], steps)

    return jsonify({f"{metric}_predictions": predictions}), 200

if __name__ == '__main__':
    app.run(debug=True)
