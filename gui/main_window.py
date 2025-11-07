import sys
import os
sys.path.append(os.path.abspath(".."))
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import torch
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor
import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCADA Main Window")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        # Central widget
        central_widget = QWidget()
        layout = QVBoxLayout()

        # --- Load data and model to show latest prediction ---
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_and_preprocess(
                path="data/raw_data.csv", lags=5
            )

            model = TempPredictor(input_size=X_train.shape[1], hidden_size=64)
            model.load_state_dict(torch.load("model/trained_model.pth"))
            model.eval()

            with torch.no_grad():
                X_last = torch.tensor(X_test[-1].reshape(1, -1), dtype=torch.float32)
                pred_last = model(X_last).numpy()
            latest_temp = scaler_y.inverse_transform(pred_last)[0][0]

            info_text = (
                f"Climate AI SCADA Visualization\n"
                f"Model: Trained\n"
                f"Dataset: Raw Data\n"
                f"Latest Predicted Temperature: {latest_temp:.2f} °C"
            )
        except Exception as e:
            info_text = f"Model or data unavailable.\nError: {e}"

        # Project info with latest prediction
        self.info_label = QLabel(info_text)
        layout.addWidget(self.info_label)

        # Buttons to open other windows
        self.stats_btn = QPushButton("Open Statistics Window")
        self.stats_btn.clicked.connect(self.open_statistics)
        layout.addWidget(self.stats_btn)

        self.process_btn = QPushButton("Open Process Window")
        self.process_btn.clicked.connect(self.open_process)
        layout.addWidget(self.process_btn)

        self.plc_btn = QPushButton("Open PLC Simulation Window")
        self.plc_btn.clicked.connect(self.open_plc)
        layout.addWidget(self.plc_btn)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    # -------------------
    # Window Open Methods
    # -------------------
    def open_statistics(self):
        from gui.statistics_window import StatisticsWindow
        self.stats_window = StatisticsWindow()
        self.stats_window.show()

    def open_process(self):
        from gui.process_window import ProcessWindow
        self.process_window = ProcessWindow()
        self.process_window.show()

    def open_plc(self):
        from gui.plc_simulation_window import PLCSimulationWindow
        self.plc_window = PLCSimulationWindow()
        self.plc_window.show()


# ------------
# Launch App
# ------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
