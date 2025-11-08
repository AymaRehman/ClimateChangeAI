import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor

def get_latest_prediction(lags=5):
    """
    Load the trained model and return the latest predicted temperature.
    Compatible with TempPredictor(input_size, hidden_size=64)
    """
    # --- Load data ---
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
        path="data/raw_data.csv", lags=lags
    )

    # --- Load model ---
    input_size = X_train.shape[1]
    model = TempPredictor(input_size=input_size, hidden_size=64)
    model.load_state_dict(torch.load("model/trained_model.pth"))
    model.eval()

    # --- Predict the latest time step ---
    with torch.no_grad():
        last_input = torch.tensor(X_test[-1:].astype(np.float32))
        pred_scaled = model(last_input).numpy()

    # --- Inverse transform to original scale ---
    pred = scaler_y.inverse_transform(pred_scaled)
    return pred.flatten()[0]  # return as single float
if __name__ == "__main__":
    latest_temp = get_latest_prediction()
    print(f"Latest predicted temperature: {latest_temp:.2f} °C")

