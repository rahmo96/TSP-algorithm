from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFrame, QLabel, QCheckBox, QPushButton, QScrollArea, QMessageBox, QRadioButton
)

class CitySelectorDialog(QDialog):
    """Dialog for selecting up to 10 cities."""

    def __init__(self, parent, cities, callback):
        super().__init__(parent)
        self.setWindowTitle("Select Cities (Max 10)")
        self.setGeometry(100, 100, 400, 400)
        self.selected_cities = []
        self.callback = callback


        layout = QVBoxLayout(self)
        label = QLabel("Select at least 5 cities:", self)
        label.setFont("Arial")
        layout.addWidget(label)

        # Scrollable area for checkboxes
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        inner_frame = QFrame(scroll_area)
        inner_layout = QVBoxLayout(inner_frame)

        # Add checkboxes
        self.checkboxes = []
        for city in cities:
            checkbox = QCheckBox(city, self)
            inner_layout.addWidget(checkbox)
            self.checkboxes.append((city, checkbox))

        scroll_area.setWidget(inner_frame)
        layout.addWidget(scroll_area)

        # Buttons
        button_frame = QFrame(self)
        layout.addWidget(button_frame)

        confirm_button = QPushButton("Confirm", self)
        confirm_button.clicked.connect(self.confirm_selection)
        button_frame_layout = QVBoxLayout(button_frame)
        button_frame_layout.addWidget(confirm_button)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        button_frame_layout.addWidget(cancel_button)

        self.setLayout(layout)

    def confirm_selection(self):
        """Save selected cities and close the dialog."""
        self.selected_cities = [city for city, checkbox in self.checkboxes if checkbox.isChecked()]
        if len(self.selected_cities) < 5:
            QMessageBox.critical(self, "Error", "Please select at least 5 cities.")
            return

        # Pass selected cities to callback
        self.callback(self.selected_cities)
        self.accept()


class StartCityDialog(QDialog):
    """Dialog for selecting the start city with scrolling support."""

    def __init__(self, parent, cities, callback):
        super().__init__(parent)
        self.setWindowTitle("Select Start City")
        self.setGeometry(100, 100, 400, 500)
        self.start_city = None
        self.callback = callback

        layout = QVBoxLayout(self)
        label = QLabel("Select the start city for the algorithm:", self)
        layout.addWidget(label)

        # Scrollable area for radio buttons
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        inner_frame = QFrame(scroll_area)
        inner_layout = QVBoxLayout(inner_frame)

        # Add radio buttons
        self.radio_buttons = []
        for city in cities:
            radio = QRadioButton(city)
            inner_layout.addWidget(radio)  # Add radio button to inner layout
            self.radio_buttons.append(radio)
            radio.toggled.connect(lambda checked,current_city = city: self.on_radio_toggled(checked, city))

        scroll_area.setWidget(inner_frame)
        layout.addWidget(scroll_area)

        # Buttons
        button_frame = QFrame(self)
        layout.addWidget(button_frame)

        confirm_button = QPushButton("Confirm", self)
        confirm_button.clicked.connect(self.confirm_selection)
        button_frame_layout = QVBoxLayout(button_frame)
        button_frame_layout.addWidget(confirm_button)

        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        button_frame_layout.addWidget(cancel_button)

        self.setLayout(layout)

    def on_radio_toggled(self, checked, city):
        """Handle radio button selection."""
        if checked:
            self.start_city = city

    def confirm_selection(self):
        """Confirm the start city and close the dialog."""
        if not self.start_city:
            QMessageBox.critical(self, "Error", "Please select a start city.")
            return

        # Pass the selected city to the callback
        self.callback(self.start_city)
        self.accept()
