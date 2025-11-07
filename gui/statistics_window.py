import sys
import os
sys.path.append(os.path.abspath(".."))

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor


class StatisticsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistics Window")
        self.setGeometry(150, 150, 700, 500)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # ---- Load Data ----
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
                path="data/raw_data.csv", lags=5
            )

            # ---- Load Model ----
            model = TempPredictor(input_size=X_train.shape[1], hidden_size=64)
            model.load_state_dict(torch.load("model/trained_model.pth"))
            model.eval()

            # ---- Evaluate ----
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                preds = model(X_test_tensor).numpy()

            y_pred = scaler_y.inverse_transform(preds)
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)

            metrics_text = f"📊 Model Evaluation Metrics:\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}"

        except Exception as e:
            metrics_text = f"Error loading data/model:\n{e}"

        # ---- Display Metrics ----
        self.metrics_label = QLabel(metrics_text)
        layout.addWidget(self.metrics_label)

        # ---- Plot Prediction vs Actual ----
        fig, ax = plt.subplots(figsize=(6, 4))
        if 'y_true' in locals():
            ax.plot(y_true[-200:], label="Actual", color="blue")
            ax.plot(y_pred[-200:], label="Predicted", color="orange")
            ax.set_title("Model Prediction vs Actual Temperature")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        self.setLayout(layout)


# Standalone test
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatisticsWindow()
    window.show()
    sys.exit(app.exec_())
