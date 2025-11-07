import sys 
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

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

        # Project info
        self.info_label = QLabel("Climate AI SCADA Visualization\nModel: Trained\nDataset: Raw Data")
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

    def open_statistics(self):
        from statistics_window import StatisticsWindow
        self.stats_window = StatisticsWindow()
        self.stats_window.show()

    def open_process(self):
        from process_window import ProcessWindow
        self.process_window = ProcessWindow()
        self.process_window.show()

    def open_plc(self):
        from plc_simulation_window import PLCSimulationWindow
        self.plc_window = PLCSimulationWindow()
        self.plc_window.show()

# Launch
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
