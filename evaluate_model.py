# Task 2e: Model Evaluation

# Import necessary libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import load_and_preprocess  # Import data loading function
from model.climate_model import TempPredictor        # Import the model architecture

# ---- Load data ----
print("Loading and preprocessing data...")
X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
    path='data/raw_data.csv', lags=5
)

# ---- Load trained model ----
print("Loading trained model...")
input_dim = X_train.shape[1]  # Number of features
model = TempPredictor(input_size=input_dim, hidden_size=64)
model.load_state_dict(torch.load("model/trained_model.pth"))
model.eval()

# ---- Evaluate ----
print("Evaluating on test data...")
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    preds = model(X_test_tensor).numpy()

# ---- Ensure y_test is 2D for inverse transform ----
y_test_2d = y_test if y_test.ndim == 2 else y_test.reshape(-1, 1)

# ---- Inverse transform predictions ----
y_pred = scaler_y.inverse_transform(preds)
y_true = scaler_y.inverse_transform(y_test_2d)

# ---- Metrics ----
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\n📊 Model Evaluation Metrics:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# ---- Visualization ----
plt.figure(figsize=(10,5))
plt.plot(y_true, label="Actual", color="blue")
plt.plot(y_pred, label="Predicted", color="orange")
plt.xlabel("Time Step")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Model Prediction vs Actual Temperature Anomaly")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
