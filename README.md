# Traveling Salesman Problem (TSP) Solver

This project implements a Genetic Algorithm (GA) approach to solve the Traveling Salesman Problem. The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the origin city.

## Features

- Genetic Algorithm implementation for TSP optimization
- Visualization of routes on an interactive map
- Benchmarking tools to evaluate algorithm performance
- Configurable parameters for the genetic algorithm

## Map Visualization

The solution visualizes routes on an interactive map using Leaflet.js. Cities are marked with pins, and the optimal route is displayed as a connected line between all points.

## Algorithm Parameters

The genetic algorithm can be configured with various parameters:
- Mutation probability
- Population size (tours per generation)
- Number of generations

## Example Results

The project includes benchmark results from various runs with different parameters. For example:

```
🔬 Experiment no. 1 | Parameters: {'prob_mutation': 0.15, 'tours_per_generation': 400, 'num_generations': 6000}
```

## Usage

1. Configure the algorithm parameters
2. Run the solver
3. View the optimized route on the generated HTML map

## Requirements

- Python 3.x
- Required libraries (requirements file forthcoming)

## Project Structure

- TSP solver implementation
- Benchmark results
- Map visualization HTML output

