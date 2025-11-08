# Predicted vs Actual Temperature Visualization Window
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor

class ProcessWindow2(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Window 2 - Predicted vs Actual")
        self.setGeometry(200, 200, 800, 500)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Load data & model
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
            self.y_pred = scaler_y.inverse_transform(preds).flatten()
            self.y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"Error loading model/data: {e}")
            self.y_pred = np.zeros(100)
            self.y_true = np.zeros(100)

        self.time_steps = len(self.y_pred)

        # Plot figure
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.time_steps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_display)
        self.layout.addWidget(self.slider)

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.auto_update)
        self.auto_index = 0
        self.timer.start(100)

        self.update_display(0)

    def auto_update(self):
        if self.auto_index < self.time_steps:
            self.slider.setValue(self.auto_index)
            self.auto_index += 1
        else:
            self.auto_index = 0

    def update_display(self, index):
        self.ax.clear()
        self.ax.plot(range(index + 1), self.y_true[:index + 1], label="Actual Temp", color="blue")
        self.ax.plot(range(index + 1), self.y_pred[:index + 1], label="Predicted Temp", color="orange")
        self.ax.set_title("Predicted vs Actual Temperature")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()


# Standalone test
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessWindow2()
    window.show()
    sys.exit(app.exec_())
