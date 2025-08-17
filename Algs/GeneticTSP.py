import itertools
import json
import random
import os
import sys
from typing import List
import googlemaps
import numpy as np
from geopy.geocoders import Nominatim
import folium
class GeneticTSP:
    """
        A class to solve the Traveling Salesperson Problem (TSP) using a Genetic Algorithm.

        Attributes:
            api_key (str): Google Maps API key.
            cities (list): List of cities to include in the TSP.
            num_cities (int): Number of cities to include in the solution.
            country (str): Country name for geolocation.
            elite_number (int): Number of elite solutions preserved in each generation.
            tours_per_generation (int): Total tours in each generation.
            num_generations (int): Number of generations to run the genetic algorithm.
            tournament_number (int): Number of solutions participating in tournament selection.
            prob_mutation (float): Probability of mutation in offspring.
            population (list): Current population of solutions.
            new_population (list): Population for the next generation.
            graph (dict): Distance graph between cities.
            best_path (list): Best path found by the algorithm.
        """
    def __init__(self, api_key, cities,num_cities=10,
                 prob_mutation=0.15, num_generations=3000,tours_per_generation=100,country="Israel", elite_number=2,
                   tournament_number=4):

        """
               Initializes the GeneticTSP class.

               Args:
                   api_key (str): Google Maps API key.
                   cities (list): List of city names.
                   num_cities (int): Number of cities to include in the TSP.
                   country (str): Country name for geolocation.
                   elite_number (int): Number of elite solutions preserved in each generation.
                   tours_per_generation (int): Total tours in each generation.
                   num_generations (int): Number of generations to run the algorithm.
                   tournament_number (int): Number of solutions in tournament selection.
                   prob_mutation (float): Probability of mutation.
               """
        self.api_key = api_key
        self.country = country
        self.cities = cities
        self.num_cities = num_cities
        self.elite_number = elite_number
        self.tours_per_generation = tours_per_generation
        self.num_generations = num_generations
        self.tournament_number = tournament_number
        self.prob_mutation = prob_mutation

        self.population = []
        self.new_population = []
        self.graph = {}
        self.best_path = []

    def load_distance_cache(self, filename=None):
        """
        Loads cached distances from a file.

        Args:
            filename (str): The name of the file to load the cache from. If None, uses default path.

        Returns:
            dict: The cached distances.
        """
        if filename is None:
            base_dir = os.path.dirname(__file__)
            filename = os.path.join(base_dir, "distances.json")

        if os.path.exists(filename):
            with open(filename, "r") as file:
                return json.load(file)
        return {}

    def save_distance_cache(self, cache, filename=None):
        """
        Save the distance cache to a file.

        Args:
            cache (dict): The cache to save.
            filename (str): The name of the file to save the cache to. If None, uses default path.
        """
        if filename is None:
            base_dir = os.path.dirname(__file__)
            filename = os.path.join(base_dir, "distances.json")

        with open(filename, "w") as file:
            json.dump(cache, file, indent=4)

    def create_graph(self, cities):
        """
        Creates a graph of distances between cities using Google Maps API.

        Args:
            cities (list): A list of city names.

        Returns:
            dict: A dictionary representing the graph of distances, where each key is a city and
            each value is another dictionary with the distances to all other cities.
        """
        distance_cache = self.load_distance_cache()

        for origin in cities:
            if origin not in self.graph:
                self.graph[origin] = {}
            for destination in cities:
                if origin == destination:
                    self.graph[origin][destination] = 0.0
                elif destination not in self.graph[origin]:
                    if destination not in self.graph:
                        self.graph[destination] = {}
                    distance = self.calculate_distances(origin, destination, distance_cache)
                    self.graph[origin][destination] = distance
                    self.graph[destination][origin] = distance

        self.save_distance_cache(distance_cache)
        return self.graph

    def calculate_distances(self, origin, destination, distance_cache):
        """
        Calculates or retrieves the distance between two cities.

        Args:
            origin (str): The name of the origin city.
            destination (str): The name of the destination city.
            distance_cache (dict): A cache of distances to avoid repeated API calls.

        Returns:
            float: The distance between the two cities in kilometers.
        """
        key = f"{origin}-{destination}"
        reverse_key = f"{destination}-{origin}"

        if key in distance_cache:
            sys.stdout.write(f"Distance between {origin} and {destination} retrieved from cache.\n")
            return distance_cache[key]
        if reverse_key in distance_cache:
            sys.stdout.write(f"Distance between {origin} and {destination} retrieved from cache.\n")
            return distance_cache[reverse_key]

        sys.stdout.write(f"Calculating distance between {origin} and {destination} using Google Maps API.\n")
        gmaps = googlemaps.Client(key=self.api_key)
        result = gmaps.distance_matrix(origin +" ," + self.country, destination + " ," + self.country, units="metric")

        if result["rows"][0]["elements"][0]["status"] == "OK":
            distance = result["rows"][0]["elements"][0]["distance"]["value"] / 1000.0
            distance_cache[key] = distance
            distance_cache[reverse_key] = distance
            return distance
        else:
            return float('inf')

    def create_initial_population(self, selected_cities):
        """
        Creates the initial population of random tours.

        Args:
            selected_cities (list[str]): The list of cities to use for the population.

        The initial population is created by first selecting a random starting city.
        Then, all permutations of the remaining cities are generated, shuffled, and
        the first tours_per_generation permutations are used to create the initial
        population of tours. Each tour is then modified to include the starting city
        as the last stop, ensuring a round trip.
        """
        start_city = selected_cities[0]
        other_cities = selected_cities[1:]

        # Generate completely random tours
        for i in range(self.tours_per_generation // 2):
            tour = [start_city] + random.sample(other_cities, len(other_cities)) + [start_city]
            self.population.append(tour)

        # Generate some greedy-based tours for better initial quality
        for i in range(self.tours_per_generation // 2, self.tours_per_generation):
            # Start with a random city after the start_city
            current = random.choice(other_cities)
            unvisited = set(other_cities) - {current}
            tour = [start_city, current]

            # Greedy selection of next cities
            while unvisited:
                next_city = min(unvisited, key=lambda city: self.graph[current][city])
                tour.append(next_city)
                current = next_city
                unvisited.remove(next_city)

            tour.append(start_city)  # Return to start
            self.population.append(tour)

    def roulette_wheel_selection(self):
        """Implements roulette wheel selection."""
        # Calculate fitness (inverse of cost since we're minimizing)
        fitnesses = [1.0 / self.cost(tour) for tour in self.population]
        total_fitness = sum(fitnesses)

        # Calculate selection probabilities
        probabilities = [f / total_fitness for f in fitnesses]

        # Select individuals based on probabilities
        selected = []
        for _ in range(self.tournament_number):
            selected.append(random.choices(self.population, weights=probabilities)[0])

        return selected

    def cost(self, tour):

        """
        Calculates the total cost of a given tour.

        Args:
            tour (list): List of cities in the tour.

        Returns:
            float: Total cost of the tour.
        """
        distances = np.array([self.graph[tour[i]][tour[i + 1]] for i in range(len(tour) - 1)])
        return np.sum(distances)

    def form_next_generation(self):
        """Creates the next generation with improved selection and diversity maintenance."""
        # Select elite individuals
        self.population.sort(key=lambda tour: self.cost(tour))
        elite = self.population[:self.elite_number]
        self.new_population = elite.copy()

        # Create rest of the population through selection, crossover, and mutation
        while len(self.new_population) < self.tours_per_generation:
            # Use roulette wheel selection instead of tournament
            if random.random() < 0.7:  # 70% chance of using roulette wheel
                parents = self.roulette_wheel_selection()
                parent1, parent2 = parents[0], parents[1]
            else:  # 30% chance of using tournament selection
                group1 = []
                group2 = []

                # Tournament selection code from your original implementation
                indices_chosen = []
                for _ in range(self.tournament_number):
                    while True:
                        random_row = random.randint(0, len(self.population) - 1)
                        if random_row not in indices_chosen:
                            indices_chosen.append(random_row)
                            group1.append(self.population[random_row].copy())
                            break

                indices_chosen = []
                for _ in range(self.tournament_number):
                    while True:
                        random_row = random.randint(0, len(self.population) - 1)
                        if random_row not in indices_chosen:
                            indices_chosen.append(random_row)
                            group2.append(self.population[random_row].copy())
                            break

                group1.sort(key=lambda x: self.cost(x))
                group2.sort(key=lambda x: self.cost(x))

                parent1 = group1[0]
                parent2 = group2[0]

            # Perform crossover
            child1, child2 = self.ordered_crossover(parent1, parent2)

            # Add children to new population
            self.new_population.append(child1)
            if len(self.new_population) < self.tours_per_generation:
                self.new_population.append(child2)

        # Apply adaptive mutation
        self.apply_adaptive_mutation()

        # Update population
        self.population = self.new_population.copy()
        self.new_population = []

    def apply_adaptive_mutation(self):
        """Applies mutation with an adaptive rate based on population diversity."""
        # Calculate population diversity (using standard deviation of costs)
        costs = [self.cost(tour) for tour in self.new_population]
        avg_cost = sum(costs) / len(costs)
        diversity = sum((c - avg_cost) ** 2 for c in costs) / len(costs)

        # Adjust mutation rate based on diversity
        # Lower diversity = higher mutation rate to encourage exploration
        diversity_factor = max(0.1, min(1.0, diversity / avg_cost))
        adjusted_mutation_rate = self.prob_mutation * (1.0 / diversity_factor)

        # Apply mutation with adjusted rate
        for i in range(self.elite_number, len(self.new_population)):  # Don't mutate elites
            if random.random() < adjusted_mutation_rate:
                self.swap_mutation(self.new_population[i])

    def choose_elite_group(self) -> List[List[int]]:
        """
        Chooses the elite group from the current population.

        This function chooses the elite group from the current population. The elite group
        is a subset of the current population and consists of the fittest individuals. The
        elite group is chosen by selecting the fittest individuals from the current population.

        Returns:
            List[List[int]]: The elite group.
        """
        elite = []
        for row in range(self.elite_number):
            elite.append(self.population[row])
        return elite

    def swap_mutation(self, tour):
        """
        Performs a swap mutation on the given tour.
        Ensures that the starting and ending city (tour[0] and tour[-1]) remain the same.

        Args:
            tour (list[int]): The tour to mutate.
        """
        n = len(tour)

        while True:
            index1 = random.randint(1, n - 2)  # Skip the first and last city
            index2 = random.randint(1, n - 2)  # Skip the first and last city

            # Ensure indices are not exactly at a specific distance (e.g., 4)
            if index2 != index1 + 4:
                break

        # Ensure index1 is always less than index2
        if index1 > index2:
            index1, index2 = index2, index1

        # Swap the cities at the selected indices
        tour[index1], tour[index2] = tour[index2], tour[index1]

    def ordered_crossover(self, parent1: List[int], parent2: List[int]):
        """
    Ordered crossover (OX) is a crossover technique used in genetic algorithms
    for permutation representations, such as the traveling salesman problem.
    It was first introduced in 1985 by Davis in the article "Applying Adaptive
    Algorithms to Epistatic Domains". It is a variation of the partially mapped
    crossover (PMX) operator. It works by choosing a random substring from one
    parent and swapping the values of the same positions in the other parent.

    Parameters:
    parent1 (List[int]): The first parent
    parent2 (List[int]): The second parent

    Returns:
        List[int], List[int]: The two children
    """
        n = len(parent1) - 1  # Exclude the return city from crossover

        while True:
            index1 = random.randint(1, n - 1)
            index2 = random.randint(1, n - 1)
            if index1 != index2:
                break
        if index1 > index2:
            index1, index2 = index2, index1
        child1 = self.generate_child(index1, index2, n, parent1, parent2)
        child2 = self.generate_child(index1, index2, n, parent2, parent1)
        return child1, child2

    def generate_child(self, index1: int, index2: int, n: int, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        This function generates a child given two parents and two indices

        Parameters:
        index1 (int): The first index
        index2 (int): The second index
        n (int): The number of elements in the parent
        parent1 (List[int]): The first parent
        parent2 (List[int]): The second parent

        Returns:
            List[int]: The child
        """
        child = [-1] * (n + 1)  # +1 for the return city

        # Directly copy the section from parent1
        child[0:index1] = parent1[0:index1]
        child[index2 + 1:] = parent1[index2 + 1:]

        # Fill in the rest from parent2
        k = index1
        for i in range(1, n):  # Start from 1 to skip the already placed return city
            if parent2[i] not in child:
                child[k] = parent2[i]
                k += 1

        return child

    def genetic_algorithm(self, selected_cities):
        """
    This function performs a genetic algorithm to find the best path that visits all cities once and returns to the origin.

    Parameters:
    selected_cities (List[str]): The list of city names

    Returns:
        Tuple[List[str], float]: A tuple of the best path and the lowest cost
    """
        generation = 0
        lowest_cost_tour = float("inf")
        self.create_initial_population(selected_cities)

        while generation < self.num_generations:
            self.population.sort(key=lambda tour: self.cost(tour))
            best_cost = self.cost(self.population[0])
            if best_cost < lowest_cost_tour:
                lowest_cost_tour = best_cost
                self.best_path = self.population[0]
                sys.stdout.write(f"Lowest cost tour at generation {generation} = {round(best_cost, 3)}km \n ")
            self.form_next_generation()
            generation += 1

        sys.stdout.write(f"Best Path: \n")
        for _ ,city in zip(range(len(self.best_path)), self.best_path):
            sys.stdout.write(f"{_+1}.{city}\n")
        sys.stdout.write(f"Cost: {round(lowest_cost_tour, 3)}km \n")
        return self.best_path, lowest_cost_tour

    def generate_map(self):
        """
        Generate a map of the best tour using Folium.

        Returns:
            str: The path to the saved HTML file.
        """
        geolocator = Nominatim(user_agent="tsp_solver")
        sys.stdout.write("Generating Map...\n")
        locations = []
        for city in self.best_path:
            location = geolocator.geocode(f"{city}, {self.country}")
            if location:
                locations.append((location.latitude, location.longitude))

        tsp_map = folium.Map(location=locations[0], zoom_start=8)

        # Only iterate through locations until the one before the last
        for idx, (lat, lon) in enumerate(locations[:-1]):
            if idx == 0:  # Start and End point
                popup_text = f"1 & {len(locations)}: {self.best_path[idx]}"
                folium.Marker(
                    location=(lat, lon),
                    popup=popup_text,
                    tooltip=popup_text,
                    icon=folium.Icon(color='red')  # Set marker color to red
                ).add_to(tsp_map)
            else:  # Intermediate points
                popup_text = f"{idx + 1}: {self.best_path[idx]}"
                folium.Marker(
                    location=(lat, lon),
                    popup=popup_text,
                    tooltip=popup_text
                ).add_to(tsp_map)

        folium.PolyLine(locations, color="blue", weight=2.5, opacity=1).add_to(tsp_map)

        map_file = os.path.join(os.getcwd(), "tsp_map.html")
        tsp_map.save(map_file)
        return map_file

    def run(self):
        self.create_graph(self.cities)
        best_path, lowest_cost = self.genetic_algorithm(self.cities)



