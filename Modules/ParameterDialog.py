from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
)

class ParameterDialog(QDialog):
    """Dialog for setting algorithm parameters with input validation."""

    def __init__(self, parent, algorithm, callback):
        super().__init__(parent)
        self.setWindowTitle(f"{algorithm} Parameters")
        self.setGeometry(100, 100, 350, 300)
        self.callback = callback
        self.params = {}

        self.fields = {
            "Genetic Algorithm": [
                ("Mutation Rate (0-1)", "0", "Higher values cause more mutations."),
                ("Tours per Generation (100+)", "0", "Defines how many tours per generation."),
                ("Generations (100+)", "0", "More generations improve solutions.")
            ],
            "Simulated Annealing": [
                ("Start Temperature (1000+)", "0", "Higher values slow convergence."),
                ("Iterations per Temperature", "0", "Higher values improve precision.")
            ]
        }

        self.create_fields(self.fields[algorithm])
        self.setup_ui()

    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        for label, default, tooltip in self.fields_list:
            frame = QFrame(self)
            layout.addWidget(frame, alignment=Qt.AlignTop)

            label_widget = QLabel(label, self)
            label_widget.setFont("Arial")
            frame_layout = QVBoxLayout(frame)
            frame_layout.addWidget(label_widget)

            entry = QLineEdit(self)
            entry.setFont("Arial")
            entry.setText(default)
            entry.setToolTip(tooltip)
            object_name = label.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            entry.setObjectName(object_name)  # Set object name here
            frame_layout.addWidget(entry)

            label_tooltip = QLabel(tooltip, self)
            label_tooltip.setFont("Arial")
            label_tooltip.setStyleSheet("color: gray;")
            frame_layout.addWidget(label_tooltip)

            entry.textChanged.connect(self.validate)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.confirm_btn = QPushButton("Confirm", self)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.confirm)
        button_layout.addWidget(self.confirm_btn)

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self.setLayout(layout)

    def validate(self):
        """Validate inputs and enable Confirm button if valid."""
        is_valid = True

        for label, default, tooltip in self.fields_list:
            object_name = label.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            entry = self.findChild(QLineEdit, object_name)
            value = entry.text()  # Get the value from the QLineEdit here

            try:
                value = float(value) if "Mutation" in label or "Temperature" in label else int(value)
                if "Mutation" in label and not (0.0 <= value <= 1.0):
                    is_valid = False
                elif "Tours" in label and value < 100:
                    is_valid = False
                elif ("Generations" in label or "Iterations" in label) and value < 1000:
                    is_valid = False
                elif "Temperature" in label and value < 1000:
                    is_valid = False
            except ValueError:
                is_valid = False

        self.confirm_btn.setEnabled(is_valid)
    def confirm(self):
        """Save parameters and close dialog."""
        for label, default, tooltip in self.fields_list:
            key = label.split()[0].lower()
            object_name = label.replace(" ", "").replace("(", "").replace(")", "").replace("-", "")
            entry = self.findChild(QLineEdit, object_name)
            value = entry.text()
            self.params[key] = float(value) if "Mutation" in label or "Temperature" in label else int(value)

        self.callback(self.params)
        self.accept()

    def reject(self):
        """Handle dialog rejection."""
        self.close()

    def create_fields(self, field_list):
        """Create labeled entry fields with explanatory text."""
        self.fields_list = field_list

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    def callback(params):
        print(f"Parameters selected: {params}")

    app = QApplication(sys.argv)
    dialog = ParameterDialog(None, "Genetic Algorithm", callback)
    dialog.exec()
    sys.exit(app.exec())
