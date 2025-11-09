# Historical Temperature Visualization Window
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ProcessWindow1(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Window 1 - Historical Temperature")
        self.setGeometry(180, 180, 800, 500)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Load data
        self.df = pd.read_csv("data/raw_data.csv").dropna(subset=["LandAverageTemperature"])
        self.df["dt"] = pd.to_datetime(self.df["dt"])
        self.temps = self.df["LandAverageTemperature"].values
        self.time_steps = len(self.temps)

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
        dates = self.df["dt"].iloc[: index + 1]
        self.ax.plot(dates, self.temps[:index + 1], color="green", label="Historical Temp")
        self.ax.set_title("Historical Land Temperature")
        self.ax.set_xlabel("Date")
        self.ax.tick_params(axis='x', rotation=45)
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()


# Standalone test
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessWindow1()
    window.show()
    sys.exit(app.exec_())
