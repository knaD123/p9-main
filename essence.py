import itertools
import math

import networkx.exception

from prefix_sorting_max_hops import *
from mpls_classes import *
from functools import *
from networkx import shortest_path, diameter, shortest_simple_paths
import networkx as nx
from collections import defaultdict
import os
from ortools.linear_solver import pywraplp
import random
from itertools import islice, cycle
import itertools
from ForwardingTable import ForwardingTable
from typing import Dict, Tuple, List, Callable

import time

def fortz_func(u):
    if u <= 1 / 20:
        return u * 0.1
    if u <= 1 / 10:
        return u * 0.3 - 0.01
    if u <= 1 / 6:
        return u * 1 - 0.08
    if u <= 1 / 3:
        return u * 2 - 0.24666
    if u <= 1 / 2:
        return u * 5 - 1.24666
    if u <= 2 / 3:
        return u * 10 - 3.74666
    if u <= 9 / 10:
        return u * 20 - 10.41333
    if u <= 1:
        return u * 70 - 55.41333
    if u <= 11 / 10:
        return u * 500 - 485.41333
    else:
        return u * 5000 - 5435.41333

def class_selection(population, capacities, loads):
    # Sort the population by fitness
    population.sort(key=lambda x: calculate_fitness(x, capacities, loads))

    # Select the top 50% of the population as parents
    a_class = population[:int(len(population) * 0.2)]
    b_class = population[int(len(population) * 0.2):int(len(population) * 0.9)]
    c_class = population[int(len(population) * 0.9):]

    return a_class, b_class, c_class

def two_point_crossover(individual1, individual2, crossover_probability):
    # Check if crossover should happen
    if random.random() > crossover_probability:
        return individual1, individual2

    # Select two random points in the individuals
    point1 = random.randint(1, len(individual1) - 1)
    point2 = random.randint(point1 + 1, len(individual1))

    # Create the offspring by exchanging the elements between the two points
    offspring1 = {}
    offspring2 = {}
    i = 0
    for (src, tgt), path in individual1.items():
        if i < point1:
            offspring1[(src, tgt)] = path
            offspring2[(src, tgt)] = individual2[(src, tgt)]
        elif i < point2:
            offspring1[(src, tgt)] = individual2[(src, tgt)]
            offspring2[(src, tgt)] = path
        else:
            offspring1[(src, tgt)] = path
            offspring2[(src, tgt)] = individual2[(src, tgt)]
        i += 1

    return offspring1, offspring2

def calculate_fitness(individual, capacities, loads):
    fitness = 0

    # Initialize the utilization of each link to 0
    utilization = {link: 0 for link in capacities.keys()}

    # Calculate the utilization of each link
    for (source, destination), path in individual.items():
        load = loads[source, destination]
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            utilization[link] += load

    # Calculate the fitness using the fortz_func
    for link, capacity in capacities.items():
        u = utilization[link] / capacity
        fitness += fortz_func(u)

    return fitness

def mutate(individual, mutation_rate, viable_paths):
    # Determine if the individual should be mutated
    if random.random() > mutation_rate:
        return individual

    # Choose a random source-destination pair to mutate
    source, destination = random.choice(list(individual.keys()))

    # Choose a new path for the pair from the viable paths
    new_path = random.choice(viable_paths[(source, destination)])

    # Mutate the individual
    individual[(source, destination)] = new_path

    return individual

def genetic_algorithm(viable_paths, capacities, population_size, crossover_rate, mutation_rate, loads, generations, elite_percent):
    # Initialize the population
    population = [{k: random.choice(v) for k, v in viable_paths.items()} for i in range(population_size)]

    # Run the genetic algorithm
    for generation in range(generations):
        # Select parents
        a_class, b_class, c_class = class_selection(population, capacities, loads)
        #print(str(generation) + ": " + str(calculate_fitness(a_class[0], capacities, loads)))
        # Generate the children
        #random_solutions = [{k: random.choice(v) for k, v in viable_paths.items()} for _ in range(int(population_size * 0.1))]
        children = a_class #+ random_solutions
        while len(children) < population_size:
            parent1 = random.choice(a_class)
            parent2 = random.choice(b_class + c_class)
            child1, child2 = two_point_crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate, viable_paths)
            child2 = mutate(child2, mutation_rate, viable_paths)
            children.extend([child1, child2])

        # Replace the population with the children
        population = children

    # Sort the population by fitness
    population.sort(key=lambda x: calculate_fitness(x, capacities, loads))

    # Return the fittest individual
    return population[0]

def remove_duplicates(lst):
    # Create a new list with the unique elements
    unique_lst = []
    for element in lst:
        if element not in unique_lst:
            unique_lst.append(element)
    return unique_lst


def essence(client):
    G = client.router.network.topology.to_directed()
    flow_to_graph = {f: client.router.network.topology.to_directed() for f in client.flows}
    for graph in flow_to_graph.values():
        for src, tgt in graph.edges:
            graph[src][tgt]["weight"] = 0 #1 / client.link_caps[src, tgt]

    pathdict = dict()
    loads = dict()

    for src, tgt, load in client.loads:
        pathdict[(src, tgt)] = []
        loads[(src, tgt)] = load

    for src, tgt, load in client.loads:
        unique_paths = []
        while True:
            path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
            for v1, v2 in zip(path[:-1], path[1:]):
                w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
                w = w * 2 + 1
                flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
            pathdict[(src, tgt)].append(path)
            if path not in unique_paths:
                unique_paths.append(path)
            if len(unique_paths) == client.mem_limit_per_router_per_flow or pathdict[(src, tgt)].count(path) == 3:
                pathdict[(src, tgt)] = unique_paths
                break

    genetic_paths = genetic_algorithm(viable_paths=pathdict, capacities=client.link_caps,
                                      population_size=client.kwargs["population"],
                                      crossover_rate=client.kwargs["crossover"],
                                      mutation_rate=client.kwargs["mutation"], loads=loads,
                                      generations=client.kwargs["generations"], elite_percent=0.2)

    for (src, tgt) in genetic_paths:
        pathdict[src, tgt].remove(genetic_paths[src, tgt])
        pathdict[src, tgt].insert(0, genetic_paths[src, tgt])

    # for (src, tgt) in pathdict:
    #    pathdict[src, tgt] = remove_duplicates(pathdict[src, tgt])

    pathdict = prefixsort(pathdict)

    path_indices = {}
    for src, tgt, load in client.loads:
        path_indices[src, tgt] = 0
        yield ((src, tgt), pathdict[src, tgt][0])

    for i in range(client.mem_limit_per_router_per_flow):
        for src, tgt, load in client.loads:
            path_index = path_indices[src, tgt]
            if path_index >= len(pathdict[src, tgt]):
                continue
            path_indices[src, tgt] += 1
            yield ((src, tgt), pathdict[src, tgt][path_index])


def normalize(value):
    min_value = min(value)
    range_value = max(value) - min_value
    if range_value == 0:
        return value
    else:
        normalized_values = [(x - min_value) / range_value for x in value]
        return normalized_values


def normalize_values(congestion, stretch):
    normalized_congestion = normalize(congestion)
    normalized_stretch = normalize(stretch)
    return normalized_congestion, normalized_stretch


def calculate_fitness_v2(individual, capacities, loads, stretch_dict):
    # Initialize the utilization of each link to 0
    utilization = {link: 0 for link in capacities.keys()}

    # Calculate the utilization of each link
    for (source, destination), path in individual.items():
        load = loads[source, destination]
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            utilization[link] += load

    # Calculate the congestion component of the fitness
    congestion = 0
    for link, capacity in capacities.items():
        u = utilization[link] / capacity
        congestion += fortz_func(u)

    # Calculate the stretch component of the fitness
    stretch = 0
    for (source, destination), paths in individual.items():
        stretch += stretch_dict[tuple(paths)]

    return congestion, stretch


def selection_v2(population, capacities, loads, stretch_dict, congestion_weight, stretch_weight):
    congestion, stretch = zip(*[calculate_fitness_v2(individual, capacities, loads, stretch_dict) for individual in
          population])

    normalized_congestion, normalized_stretch = normalize_values(congestion, stretch)

    fitness_values = [normalized_congestion[i] * congestion_weight + normalized_stretch[i] * stretch_weight for i in range(len(population))]

    # Zip the fitness values and the population together
    fitness_population = zip(fitness_values, population)

    # Sort the list of tuples by the fitness values
    sorted_fitness_population = sorted(fitness_population, key=lambda x: x[0])

    # Extract the individuals from the sorted list of tuples
    #population = [individual for fitness, individual in sorted_fitness_population]

    # Select the top 50% of the population as parents
    #num_parents = int(len(population) * 0.5)
    #parents = population[:num_parents]

    return sorted_fitness_population


def mutate_v2(individual, mutation_rate, viable_paths):
    # Determine if the individual should be mutated
    if random.random() > mutation_rate:
        return individual

    # Choose a random source-destination pair to mutate
    source, destination = random.choice(list(individual.keys()))

    # Choose a new path order for the pair from the viable paths
    new_path_order = random.sample(viable_paths[(source, destination)], len(viable_paths[(source, destination)]))

    # Mutate the individual
    individual[(source, destination)] = new_path_order

    return individual

def class_selection_v2(population, capacities, loads):
    # Sort the population by fitness
    #population.sort(key=lambda x: calculate_fitness(x, capacities, loads))
    population = [individual for fitness, individual in population]

    # Select the top 50% of the population as parents
    a_class = population[:int(len(population) * 0.2)]
    b_class = population[int(len(population) * 0.2):]

    return a_class, b_class


def genetic_algorithm_v2(viable_paths, capacities, population_size, crossover_rate, mutation_rate, loads, generations, stretch_dict, congestion_weight, stretch_weight):
    # Initialize the population
    population = [{k: random.choice(v) for k, v in viable_paths.items()} for i in range(population_size)]

    # Run the genetic algorithm
    for generation in range(generations):
        # Select parents
        parents = selection_v2(population, capacities, loads, stretch_dict=stretch_dict,
                               congestion_weight=congestion_weight,
                               stretch_weight=stretch_weight)

        a_class, b_class = class_selection_v2(parents, capacities, loads)
        #print(str(generation) + ": " + str(calculate_fitness(a_class[0], capacities, loads)))

        # Generate the children
        children = a_class
        while len(children) < population_size:
            parent1 = random.choice(a_class)
            parent2 = random.choice(b_class)
            child1, child2 = two_point_crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate, viable_paths)
            child2 = mutate(child2, mutation_rate, viable_paths)
            children.extend([child1, child2])

        # Replace the population with the children
        population = children

    congestion, stretch = zip(
        *[calculate_fitness_v2(individual, capacities, loads, stretch_dict) for individual in
          population])

    normalized_congestion, normalized_stretch = normalize_values(congestion, stretch)

    fitness_values = [normalized_congestion[i] * congestion_weight + normalized_stretch[i] * stretch_weight for i in range(len(population))]

    # Zip the fitness values and the population together
    fitness_population = zip(fitness_values, population)

    # Sort the list of tuples by the fitness values
    sorted_fitness_population = sorted(fitness_population, key=lambda x: x[0])

    # Extract the individuals from the sorted list of tuples
    population = [individual for fitness, individual in sorted_fitness_population]

    # Return the fittest individual
    return population[0]


def essence_v2(client):
    G = client.router.network.topology.to_directed()
    flow_to_graph = {f: client.router.network.topology.to_directed() for f in client.flows}
    for graph in flow_to_graph.values():
        for src, tgt in graph.edges:
            graph[src][tgt]["weight"] = 0  # 1000 / client.link_caps[src, tgt]

    pathdict = dict()
    loads = dict()
    shortest_paths_len = dict()
    stretch_dict = {}

    for src, tgt, load in client.loads:
        pathdict[(src, tgt)] = []
        loads[(src, tgt)] = load
        shortest_paths_len[(src, tgt)] = len(shortest_path(G, src, tgt))

    for src, tgt, load in client.loads:
        unique_paths = []
        while True:
            path = nx.shortest_path(flow_to_graph[(src, tgt)], src, tgt, weight="weight")
            for v1, v2 in zip(path[:-1], path[1:]):
                w = flow_to_graph[(src, tgt)][v1][v2]["weight"]
                w = w * 2 + 1
                flow_to_graph[(src, tgt)][v1][v2]["weight"] = w
            pathdict[(src, tgt)].append(path)
            if path not in unique_paths:
                unique_paths.append(path)
            if pathdict[(src, tgt)].count(path) == 3 or len(unique_paths) == client.mem_limit_per_router_per_flow:
                break

    # Create stretch dictionary, so it does not have to be recomputed in the genetic algorithm
    for src, tgt, load in client.loads:
        # Get shortest path
        shortest_path_len = shortest_paths_len[src, tgt]

        # Calculate the stretch value for each path between the source and destination
        for path in pathdict[src, tgt]:
            path_tuple = tuple(path)
            path_len = len(path)
            stretch_dict[path_tuple] = (path_len / shortest_path_len)

    # Get the path weights such that backup paths are weighted less than primary path
    #maximum_number_of_paths_for_demand = len(max(pathdict.values(), key=lambda x: len(x)))
    #path_weights = calculate_weights(maximum_number_of_paths_for_demand)
    genetic_paths = genetic_algorithm_v2(viable_paths=pathdict, capacities=client.link_caps,
                                    population_size=client.kwargs["population"],
                                    crossover_rate=client.kwargs["crossover"],
                                    mutation_rate=client.kwargs["mutation"], loads=loads,
                                    generations=client.kwargs["generations"], stretch_dict=stretch_dict,
                                    congestion_weight=client.kwargs["congestion_weight"],
                                    stretch_weight=client.kwargs["stretch_weight"])

    for (src, tgt) in genetic_paths:
        pathdict[src, tgt].remove(genetic_paths[src, tgt])
        pathdict[src, tgt].insert(0, genetic_paths[src, tgt])

    pathdict = prefixsort(pathdict)

    path_indices = {}
    for src, tgt, load in client.loads:
        path_indices[src, tgt] = 0
        yield ((src, tgt), pathdict[src, tgt][0])

    for i in range(client.mem_limit_per_router_per_flow):
        for src, tgt, load in client.loads:
            path_index = path_indices[src, tgt]
            if path_index >= len(pathdict[src, tgt]):
                continue
            path_indices[src, tgt] += 1
            yield ((src, tgt), pathdict[src, tgt][path_index])