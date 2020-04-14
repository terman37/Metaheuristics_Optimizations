import numpy as np
import random
import math
import operator
import re
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit


# TSPLIB file reader
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
        mat = np.array(matrix)
        return mat, dimension, cities


@jit(nopython=True)
def routeFitness(pop):
    fitness = 0
    for i in range(pop.shape[0] - 1):
        x = pop[i]
        y = pop[i + 1]
        fitness += dist_matrix[x][y]
    first_city, last_city = pop[0], pop[-1]
    fitness += dist_matrix[first_city][last_city]
    return fitness

@jit(nopython=True)
def createRoute(cityList):
    route = np.random.permutation(cityList)
    return route


@jit(nopython=True)
def initialPopulation(popSize, cityList):
    population = np.empty((popSize, len(cityList)), dtype=np.int8)
    for i in range(0, popSize):
        population[i] = createRoute(cityList)
    return population


@jit(nopython=True)
def rankRoutes(population):
    fitnessResults = np.empty((population.shape[0], 2), dtype=np.int32)
    for i in range(population.shape[0]):
        fitnessResults[i, 0] = i
        fitnessResults[i, 1] = routeFitness(population[i])
    fitnessResults = fitnessResults[np.argsort(fitnessResults[:, 1])]
    return fitnessResults


@jit(nopython=True)
def selection(popRanked, eliteSize):
    selectionResults = np.empty((popRanked.shape[0],), dtype=np.int8)
    select = np.empty((popRanked.shape[0], 4))
    select[:, 0:2] = popRanked
    select[:, 2] = np.cumsum(select[:, 1])
    select[:, 3] = np.cumsum(select[:, 1]) / np.sum(select[:, 1])
    for i in range(0, eliteSize):
        selectionResults[i] = popRanked[i, 0]
    for i in range(eliteSize, popRanked.shape[0]):
        pick = random.random()
        for j in range(0, popRanked.shape[0]):
            if pick <= select[j, 3]:
                selectionResults[i] = popRanked[i, 0]
                break
    return selectionResults


@jit(nopython=True)
def matingPool(population, selectionResults):
    matingpool = np.empty(population.shape, dtype=np.int8)
    for i in range(0, selectionResults.shape[0]):
        index = selectionResults[i]
        matingpool[i] = population[index]
    return matingpool


# @jit(nopython=True)
def breed(parent1, parent2):
    childP1 = np.zeros(parent1.shape, dtype=np.int8)
    childP1.fill(-1)
    geneA = random.randint(0, parent1.shape[0]-1)
    geneB = geneA
    while geneB == geneA:
        geneB = random.randint(0, parent1.shape[0]-1)

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1[i] = parent1[i]

    # childP2 = np.zeros((parent2.shape[0]-(endGene-startGene),), dtype=np.int8)
    # x = 0
    # for i in range(parent2.shape[0]):
    #     if childP1[i]

    # mask = np.in1d(parent2, childP1)
    # childP2 = childP1[-mask]
    childP2 = np.setdiff1d(parent2, childP1)
    if startGene != 0:
        childP1[:startGene] = childP2[:startGene]
    if endGene != 0:
        childP1[endGene:] = childP2[startGene:]

    # TODO REMOVE IF OK
    test = np.zeros(parent1.shape, dtype=np.int8)
    test.fill(-1)

    if childP1.all() == test.all():
        print("here")

    return childP1


# @jit(nopython=True)
def breedPopulation(matingpool, eliteSize):
    children = np.empty(matingpool.shape, dtype=np.int8)
    for i in range(0, eliteSize):
        children[i] = matingpool[i]
    np.random.shuffle(matingpool)
    for i in range(eliteSize, matingpool.shape[0]):
        child = breed(matingpool[i], matingpool[matingpool.shape[0] - i - 1])
        children[i] = child
    return children


@jit(nopython=True)
def mutate(individual, mutationRate):
    for swapped in range(individual.shape[0]):
        if random.random() < mutationRate:
            swapWith = random.randint(0, individual.shape[0]-1)
            city1 = individual[swapped]
            city2 = individual[swapWith]
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


@jit(nopython=True)
def mutatePopulation(population, mutationRate):
    mutatedPop = np.empty(population.shape, dtype=np.int8)
    for i in range(population.shape[0]):
        mutatedInd = mutate(population[i], mutationRate)
        mutatedPop[i] = mutatedInd
    return mutatedPop


# @jit(nopython=True)
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, distances, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)  # ongoing
    print("Initial distance: " + str(rankRoutes(pop)[0][1]))
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    print("Final distance: " + str(rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


# instantiate problem to solve
problem_name = 'Qatar 194 TSP'
optimal_fitness = 9352
dist_matrix, nb_cities, cities_coord = read_tsplib_file('./dj38.tsp')
cityList = cities_coord.index.values

geneticAlgorithm(population=cityList, distances=dist_matrix, popSize=100, eliteSize=20, mutationRate=0.02, generations=5000)
