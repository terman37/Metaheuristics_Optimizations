import pandas as pd
import re
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

from deap import algorithms, base, creator, tools

def read_tsplib_file(filename):
    if filename is None:
        raise FileNotFoundError('Filename can not be None')
    with open(filename) as file:
        lines = file.readlines()
        data = [line.lstrip() for line in lines if line != ""]
        dimension = re.compile(r'[^\d]+')
        for item in data:
            if item.startswith('DIMENSION'):
                dimension = int(dimension.sub('', item))
                break
        c = [-1.0] * (2 * dimension)
        cities_coord = []
        for item in data:
            if item[0].isdigit():
                j, coordX, coordY = [float(x.strip()) for x in item.split(' ')]
                c[2 * (int(j) - 1)] = coordX
                c[2 * (int(j) - 1) + 1] = coordY
                cities_coord.append([coordX, coordY])
        cities = pd.DataFrame(cities_coord)
        #         cities = cities_coord
        matrix = [[-1] * dimension for _ in range(dimension)]
        for k in range(dimension):
            matrix[k][k] = 0
            for j in range(k + 1, dimension):
                dist = math.sqrt((c[k * 2] - c[j * 2]) ** 2 + (c[k * 2 + 1] - c[j * 2 + 1]) ** 2)
                dist = round(dist)
                matrix[k][j] = dist
                matrix[j][k] = dist
        # mat = np.array(matrix)
        return matrix, dimension, cities

distance_map, nb_cities, cities_coord = read_tsplib_file('./dj38.tsp')

toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, typecode='i', fitness=creator.FitnessMin)

# Attribute generator
toolbox.register("indices", np.random.permutation, nb_cities)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

# toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)

random.seed(169)

pop = toolbox.population(n=100)

# hof = tools.HallOfFame(10)
# stats = tools.Statistics(lambda ind: ind.fitness.values)
# stats.register("avg", np.mean)
# stats.register("std", np.std)
# stats.register("min", np.min)
# stats.register("max", np.max)

# algorithms.eaSimple(pop, toolbox, 0.7, 0.11, 5000, stats=stats, halloffame=hof)

fit_stats = tools.Statistics(lambda ind: ind.fitness.values)
fit_stats.register('min', np.min)

t1 = time.time()
result, log = algorithms.eaSimple(toolbox.population(n=100), toolbox,
                                  cxpb=0.8, mutpb=0.2,
                                  ngen=1000, verbose=False,
                                  stats=fit_stats)

t2 = time.time()
best_individual = tools.selBest(result, k=1)[0]
print('Fitness of the best individual: ', evalTSP(best_individual)[0])
print(t2 - t1)
plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('min'), 'c-')
plt.ylabel('Fitness');
plt.xlabel('generations');
plt.show()
