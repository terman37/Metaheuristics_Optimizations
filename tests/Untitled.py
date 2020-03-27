import os
import time
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from numba import jit
import pygmo as pg

# parameters

func_name = 'rosenbrock'
search_space = (-100, 100)

funcs = {"sphere": 0, "schwefel": 1, "rosenbrock": 2, "rastrigin": 3, "griewank": 4, "ackley": 5}

val_path = os.path.join('../data/', func_name + '.csv')
bias_path = '../data/f_bias.csv'

func_df = pd.read_csv(val_path)
bias_df = pd.read_csv(bias_path)

funcval = func_df.fvalue.values
funcbias = bias_df.fvalue.values[funcs[func_name]]


# function definitions

# @jit(nopython=True)
def eval_fitness(x, dim):
    if func_name == "sphere":
        F = 0
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F += z * z
        result = F + funcbias
    elif func_name == "schwevel":
        F = abs(x[0])
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F = max(F, abs(z))
        result = F + funcbias
    elif func_name == "rosenbrock":
        F = 0
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F += z * z
        result = F + funcbias
    elif func_name == "rastrigin":
        F = 0
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F += z ** 2 - 10 * math.cos(2 * math.pi * z) + 10
        result = F + funcbias
    elif func_name == "griewank":
        F1 = 0
        F2 = 1
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F1 += z ** 2 / 4000
            F2 += math.cos(z / math.sqrt(i + 1))
        result = F1 - F2 + 1 + funcbias
    elif func_name == "ackley":
        Sum1 = 0
        Sum2 = 0
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            Sum1 += z ** 2
            Sum2 += math.cos(2 * math.pi * z)
        result = -20 * math.exp(-0.2 * math.sqrt(Sum1 / dim)) - math.exp(Sum2 / dim) + 20 + math.e + funcbias
    else:
        result = 0
    return result


# Define problem Class
class My_problem:

    def __init__(self, dim, bounds, glob_opt):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.dimension = dim
        self.fitness_per_eval = []
        self.glob_opt = glob_opt

    def fitness(self, x):
        result = eval_fitness(x, self.dimension)
        return [result]

    def get_bounds(self):
        x_min = self.lower_bound * np.ones(self.dimension)
        x_max = self.upper_bound * np.ones(self.dimension)
        return x_min, x_max


def solve_pb(dim, my_algo, bounds, optim, pop_size):
    prob = pg.problem(My_problem(dim, bounds, optim))
    pop = pg.population(prob, pop_size)

    my_algo.set_verbosity(1)

    t1 = time.time()
    pop = algo.evolve(pop)
    t2 = time.time()

    time_diff = t2-t1

    # why get_fevals return 0 ?
    nb_evals = prob.get_fevals()

    return pop, time_diff


def print_solution(my_algo, pop_evolved, duration):

    algorithm_name = my_algo.get_name()
    parameters = my_algo.get_extra_info()
    solution_x = pop_evolved.champion_x
    fitness = pop_evolved.champion_f[0]

    print("Algorithm: %s" % algorithm_name)
    print("Parameters: %s" % parameters)
    print("Solution: ", solution_x)
    print("Fitness: %f" % fitness)

    extract_algo = my_algo.extract(pg.pso)
    log = extract_algo.get_log()
    gbest = [x[2] for x in log]
    plt.plot(gbest)
    plt.show()

    # print("Nb of functions evaluations: %d in %d iterations" % (prob.get_fevals(), 0))
    # print("Stopping criterion: %s" % result.message)

    print("computational time: %.3f seconds" % duration)

    gbest = [x[1] for x in log]
    plt.plot(gbest)


# Solve problem in dimension 50
DIM = 50
# define algorithm and parameters to use
algo = pg.algorithm(pg.pso(gen=200))
pop_size = 100
# run algorithm and print solution
pop_evolv, compute_time = solve_pb(dim=DIM, my_algo=algo, bounds=search_space, optim=funcbias, pop_size=100)
print_solution(my_algo=algo, pop_evolved=pop_evolv, duration=compute_time)


