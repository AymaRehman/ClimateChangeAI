import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import torch
from utils.data_loader import load_and_preprocess
from model.climate_model import TempPredictor
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCADA Main Window")
        self.setGeometry(100, 100, 600, 500)  # slightly taller for two process buttons
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout()

        # --- Load latest prediction ---
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

        self.info_label = QLabel(info_text)
        layout.addWidget(self.info_label)

        # --- Buttons to open windows ---
        self.stats_btn = QPushButton("Open Statistics Window")
        self.stats_btn.clicked.connect(self.open_statistics)
        layout.addWidget(self.stats_btn)

        self.process1_btn = QPushButton("Open Process Window 1 (Historical Temp)")
        self.process1_btn.clicked.connect(self.open_process1)
        layout.addWidget(self.process1_btn)

        self.process2_btn = QPushButton("Open Process Window 2 (Predicted vs Actual)")
        self.process2_btn.clicked.connect(self.open_process2)
        layout.addWidget(self.process2_btn)

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

    def open_process1(self):
        from gui.process_window1 import ProcessWindow1
        self.process_window1 = ProcessWindow1()
        self.process_window1.show()

    def open_process2(self):
        from gui.process_window2 import ProcessWindow2
        self.process_window2 = ProcessWindow2()
        self.process_window2.show()

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
