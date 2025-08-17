import os
import sys
import threading
from PySide6.QtCore import Qt, Slot, QObject, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QRadioButton, QGroupBox, QMessageBox
)
from Algs.GeneticTSP import GeneticTSP
from Algs.SimulatedAnnealing import SimulatedAnnealing
from Modules.CitySelectorDialog import CitySelectorDialog, StartCityDialog
from Modules.ParameterDialog import ParameterDialog
from Modules.RedirectText import RedirectText

class Communicate(QObject):
    openMapRequest = Signal(str)

class TSPGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver")
        self.setGeometry(100, 100, 500, 700)

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        self.create_widgets()

        # Redirect stdout to results_text
        sys.stdout = RedirectText(self.results_text)

        # Set window icon
        icon_path = "1.webp"  # Replace with the actual path to your icon file
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warning: Icon file not found at '{icon_path}'")

        # Cities list
        self.cities = [
            "Jerusalem", "Tel Aviv", "Haifa", "Be'er Sheva", "Ashdod",
            "Rishon LeTsion", "Petah Tikva", "Netanya", "Ashkelon", "Afula",
            "Eilat", "Nazareth", "Tiberias", "Holon", "Bat Yam",
            "Bnei Brak", "Ramat Gan", "Givatayim", "Herzliya", "Kfar Saba",
            "Rehovot", "Modiin", "Raanana", "Hadera", "Beit Shemesh",
            "Lod", "Ramla", "Kiryat Gat", "Kiryat Malakhi", "Kiryat Shmona",
            "Nahariya", "Arad", "Yavne", "Shoham",
            "Tirat Carmel", "Or Akiva", "Zichron Yaakov", "Karmiel", "Ma'alot-Tarshiha",
            "Migdal HaEmek", "Kiryat Yam", "Kiryat Ata", "Kiryat Bialik", "Kiryat Motzkin",
            "Ness Ziona", "Rosh HaAyin", "Gan Yavne", "Sderot", "Ofakim"
        ]

        # Create communication object
        self.communicate = Communicate()
        self.communicate.openMapRequest.connect(self.ask_to_open_map)

    def create_widgets(self):
        """Create the GUI layout."""
        layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("TSP Solver")
        title_label.setFont("Arial")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # API Key Input
        api_key_label = QLabel("Enter Google Maps API Key:")
        self.api_key_entry = QLineEdit()
        layout.addWidget(api_key_label)
        layout.addWidget(self.api_key_entry)

        # Algorithm Selection
        algorithm_group = QGroupBox("Select Algorithm")
        algorithm_layout = QVBoxLayout()
        self.algorithm_var = "Genetic Algorithm"  # Default selection
        genetic_radio = QRadioButton("Genetic Algorithm")
        genetic_radio.setChecked(True)
        genetic_radio.toggled.connect(lambda: self.set_algorithm("Genetic Algorithm"))
        algorithm_layout.addWidget(genetic_radio)
        sa_radio = QRadioButton("Simulated Annealing")
        sa_radio.toggled.connect(lambda: self.set_algorithm("Simulated Annealing"))
        algorithm_layout.addWidget(sa_radio)
        algorithm_group.setLayout(algorithm_layout)
        layout.addWidget(algorithm_group)

        # Algorithm Notes Section
        notes_group = QGroupBox("Algorithm Notes")
        notes_layout = QVBoxLayout()
        self.notes_text = QTextEdit()
        self.notes_text.setReadOnly(True)
        notes_layout.addWidget(self.notes_text)
        notes_group.setLayout(notes_layout)
        layout.addWidget(notes_group)

        # Results Section
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Buttons
        button_layout = QHBoxLayout()
        start_button = QPushButton("Start Algorithm")
        start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* Green */
                border: none;
                color: white;
                padding: 5px 5px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
        """)
        start_button.clicked.connect(self.select_cities_dialog)
        button_layout.addWidget(start_button)
        button_layout.addWidget(start_button)
        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet("""
                    QPushButton {
                        background-color: #ff0000; /* Red */
                        border: none;
                        color: white;
                        padding: 5px 5px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #3e8e41;
                    }
                """)
        quit_button.clicked.connect(self.close)
        button_layout.addWidget(quit_button)
        layout.addLayout(button_layout)

        self.api_key_entry = QLineEdit()
        layout.addWidget(api_key_label)
        layout.addWidget(self.api_key_entry)

        # Set layout
        self.central_widget.setLayout(layout)

        # Initial notes update
        self.update_notes()

    @Slot()
    def set_algorithm(self, algorithm):
        """Slot to set the selected algorithm."""
        self.algorithm_var = algorithm
        self.update_notes()

    def update_notes(self):
        """Update the notes section based on the selected algorithm."""
        notes = {
            "Genetic Algorithm": (
                "The Genetic Algorithm uses concepts of natural selection to solve optimization problems. "
                "Key parameters include:\n"
                "- Probability of Mutation: Controls the diversity in the population.\n"
                "- Number of Generations: Determines the number of iterations for evolution.\n"
            ),
            "Simulated Annealing": (
                "Simulated Annealing mimics the process of heating and slowly cooling a material to find a low-energy state. "
                "Key parameters include:\n"
                "- Initial Temperature: Sets the starting 'energy' level.\n"
                "- Iterations per Temperature: Controls how thoroughly each temperature is explored.\n"
            ),
        }

        # Clear existing text and insert new notes
        self.notes_text.clear()
        self.notes_text.insertPlainText(notes.get(self.algorithm_var, "No information available."))

    def select_cities_dialog(self):
        """Open the city selection dialog and handle the user's selections."""
        def open_start_city_dialog(selected_cities):
            """Open the Start City dialog after selecting cities."""
            def on_start_city_selected(start_city):
                """Handle the selected start city."""
                self.open_parameter_dialog(selected_cities, start_city)

            # Open Start City Dialog
            start_city_dialog = StartCityDialog(self, selected_cities, callback=on_start_city_selected)
            start_city_dialog.exec()

        def on_cities_selected(selected_cities):
            """Callback for when cities are selected."""
            if not selected_cities:
                QMessageBox.warning(self, "Warning", "No cities selected!")
                return
            open_start_city_dialog(selected_cities)

        # Open CitySelectorDialog
        city_selector_dialog = CitySelectorDialog(self, self.cities, callback=on_cities_selected)
        city_selector_dialog.exec()

    def open_parameter_dialog(self, selected_cities, start_city):
        for city in selected_cities:
            if city == start_city:
                selected_cities.remove(city)
        selected_cities.insert(0, start_city)

        algorithm = self.algorithm_var

        def on_parameters_set(params):
            self.start_algorithm(selected_cities, params)

        dialog = ParameterDialog(self, algorithm, on_parameters_set)
        dialog.exec()

    def start_algorithm(self, selected_cities, params):
        api_key_entry = self.api_key_entry
        if api_key_entry is None:
            QMessageBox.warning(self, "Warning", "API Key field is not initialized!")
            return

        api_key = api_key_entry.text()
        algorithm = self.algorithm_var

        threading.Thread(
            target=self.run_algorithm,
            args=(algorithm, api_key, selected_cities, params),
            daemon=True
        ).start()

    def run_algorithm(self, algorithm, api_key, selected_cities, params):
        map_file = None
        if algorithm == "Genetic Algorithm":
            tsp = GeneticTSP(
                api_key, selected_cities, len(selected_cities),
                prob_mutation=params["mutation"],
                num_generations=params["generations"],
                tours_per_generation=params["tours"],
            )
            tsp.run()
            map_file = tsp.generate_map()
        elif algorithm == "Simulated Annealing":
            sa = SimulatedAnnealing(
                selected_cities, len(selected_cities),
                temperature=params["start"],
                iterations_per_temp=params["iterations"]
            )
            sa.initialize_graph(api_key)
            sa.simulated_annealing()
            map_file = sa.generate_map()

        if map_file:
            self.communicate.openMapRequest.emit(map_file)

    @Slot(str)
    def ask_to_open_map(self, map_file):
        """Ask the user if they want to open the generated map."""
        response = QMessageBox.question(self, "Open Map", "The map has been generated. Do you want to open it?",
                                        QMessageBox.Yes | QMessageBox.No)
        if response == QMessageBox.Yes:
            if sys.platform == "win32":
                os.startfile(map_file)  # Windows
            elif sys.platform == "darwin":
                os.system(f"open {map_file}")  # macOS
            elif sys.platform == "linux":
                os.system(f"xdg-open {map_file}")  # Linux
        else:
            QMessageBox.information(self, "Information", "You chose not to open the map.")

if __name__ == "__main__":
    app = QApplication([])
    window = TSPGUI()
    window.show()
    sys.exit(app.exec())