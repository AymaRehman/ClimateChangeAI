import sys
import os
sys.path.append(os.path.abspath(".."))

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor


class ProcessWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Visualization Window")
        self.setGeometry(180, 180, 800, 500)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # ---- Title ----
        title = QLabel("🌍 Climate Process Visualization")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # ---- Load Data & Model ----
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
                path="data/raw_data.csv", lags=5
            )

            model = TempPredictor(input_size=X_train.shape[1], hidden_size=64)
            model.load_state_dict(torch.load("model/trained_model.pth"))
            model.eval()

            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                preds = model(X_test_tensor).numpy()

            y_pred = scaler_y.inverse_transform(preds)
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        except Exception as e:
            y_true, y_pred = np.zeros((100, 1)), np.zeros((100, 1))
            print(f"Error loading model/data: {e}")

        # ---- Plot 1: Historical trend ----
        fig1, ax1 = plt.subplots(figsize=(6, 3))
        df = pd.read_csv("data/raw_data.csv").dropna(subset=["LandAverageTemperature"])
        df["dt"] = pd.to_datetime(df["dt"])
        ax1.plot(df["dt"].tail(300), df["LandAverageTemperature"].tail(300), color="green")
        ax1.set_title("Historical Temperature Trend")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Average Temperature (°C)")
        ax1.grid(True)

        canvas1 = FigureCanvas(fig1)
        layout.addWidget(canvas1)

        # ---- Plot 2: Predicted vs Actual ----
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(y_true[-200:], label="Actual", color="blue")
        ax2.plot(y_pred[-200:], label="Predicted", color="orange")
        ax2.set_title("Predicted vs Actual Process Values")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Temperature (°C)")
        ax2.legend()
        ax2.grid(True)

        canvas2 = FigureCanvas(fig2)
        layout.addWidget(canvas2)

        self.setLayout(layout)


# Standalone test
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessWindow()
    window.show()
    sys.exit(app.exec_())
