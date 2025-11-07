import sys
import os
import numpy as np
import pandas as pd
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from model.climate_model import TempPredictor
from utils.data_loader import load_and_preprocess

class PLCSimulationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLC Simulation Window")
        self.setGeometry(200, 200, 750, 500)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- Load dataset ---
        self.df = pd.read_csv("data/raw_data.csv")
        self.temps = self.df["LandAverageTemperature"].ffill().values
        self.time_steps = len(self.temps)

        # --- Load model predictions ---
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
            path="data/raw_data.csv", lags=5
        )
        self.model = TempPredictor(input_size=X_train.shape[1], hidden_size=64)
        self.model.load_state_dict(torch.load("model/trained_model.pth"))
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            preds = self.model(X_tensor).numpy()
            self.model_preds = scaler_y.inverse_transform(preds).flatten()

        # Align lengths
        min_len = min(len(self.temps), len(self.model_preds))
        self.temps = self.temps[:min_len]
        self.model_preds = self.model_preds[:min_len]

        # --- Matplotlib Figure ---
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # --- Current temperature label ---
        self.temp_label = QLabel("Current Predicted Temp: -- °C")
        self.temp_label.setAlignment(Qt.AlignCenter)
        self.temp_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; background-color: lightgrey; padding: 6px; border-radius: 5px;"
        )
        self.layout.addWidget(self.temp_label)

        # --- Slider ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(min_len - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_display)
        self.layout.addWidget(self.slider)

        # --- Timer for auto playback ---
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_update)
        self.auto_index = 0
        self.timer.start(150)  # 150ms per update for smoother animation

        self.update_display(0)

    def auto_update(self):
        if self.auto_index < len(self.model_preds):
            self.slider.setValue(self.auto_index)
            self.auto_index += 1
        else:
            self.auto_index = 0

    def update_display(self, index):
        self.ax.clear()

        # Plot actual and predicted temperatures
        self.ax.plot(range(index + 1), self.temps[:index + 1], label="Actual Temp", color="#1f77b4")
        self.ax.plot(range(index + 1), self.model_preds[:index + 1], label="Predicted Temp", color="#ff7f0e")
        self.ax.set_xlabel("Time Step", fontsize=12)
        self.ax.set_ylabel("Temperature (°C)", fontsize=12)
        self.ax.set_title("Temperature vs Time", fontsize=14, fontweight="bold")
        self.ax.grid(alpha=0.3)
        self.ax.legend()
        self.ax.set_facecolor("#f7f7f7")  # light grey background

        current_temp = self.model_preds[index]
        self.temp_label.setText(f"Current Predicted Temp: {current_temp:.2f} °C")

        # --- Adaptive threshold using last 20 steps ---
        window = 20
        start_idx = max(0, index - window + 1)
        recent_vals = self.model_preds[start_idx : index + 1]
        adaptive_threshold = np.mean(recent_vals) + 1.0 * np.std(recent_vals)  # more sensitive

        # Update label color & background
        if current_temp > adaptive_threshold:
            self.temp_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; background-color: red; color: white; padding: 6px; border-radius: 5px;"
            )
        else:
            self.temp_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; background-color: lightgreen; color: black; padding: 6px; border-radius: 5px;"
            )

        self.canvas.draw()

# --- Standalone run ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PLCSimulationWindow()
    window.show()
    sys.exit(app.exec_())
