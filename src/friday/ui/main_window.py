import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class FridayMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Friday")
        self.resize(1200, 750)

        label = QLabel("Friday is online.")
        label.setStyleSheet(
            """
            font-size: 28px;
            padding: 40px;
            """
        )

        self.setCentralWidget(label)


def run_app() -> None:
    app = QApplication(sys.argv)

    window = FridayMainWindow()
    window.show()

    sys.exit(app.exec())