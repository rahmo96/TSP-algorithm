import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget


class RedirectText:
    """Redirects stdout to a QTextEdit widget."""

    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        cursor = self.widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.widget.setTextCursor(cursor)
        self.widget.ensureCursorVisible()

    def flush(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Redirect Stdout Example")
        self.setGeometry(100, 100, 600, 400)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Text Edit Widget
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        # Redirect stdout
        sys.stdout = RedirectText(self.text_edit)

        self.show()


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
