import sys
from PySide6.QtWidgets import QApplication
from ui.menu import MainMenu

app = QApplication(sys.argv)
win = MainMenu()
win.show()
sys.exit(app.exec())
