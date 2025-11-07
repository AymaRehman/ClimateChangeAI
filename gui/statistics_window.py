import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

class StatisticsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Statistics Window")
        self.setGeometry(150, 150, 600, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Metrics info (placeholder values)
        self.metrics_label = QLabel("Model Evaluation Metrics:\nMAE: 0.06\nRMSE: 0.13\nR²: 0.95")
        layout.addWidget(self.metrics_label)

        # Plot training/validation loss (example)
        fig, ax = plt.subplots(figsize=(5,3))
        epochs = np.arange(1, 201)
        train_loss = np.exp(-epochs/100) + 0.01*np.random.randn(200)
        val_loss = np.exp(-epochs/110) + 0.02*np.random.randn(200)
        ax.plot(epochs, train_loss, label='Training Loss')
        ax.plot(epochs, val_loss, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.grid(True)
        ax.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        self.setLayout(layout)

# Standalone testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatisticsWindow()
    window.show()
    sys.exit(app.exec_())
