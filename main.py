"""
----------------------------------------
main.py
----------------------------------------
Project entry point for the SCADA-style AI Climate Dashboard.
Launches the MainWindow GUI.
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
