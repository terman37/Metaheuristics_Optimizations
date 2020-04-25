import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
import pygmo as pg
import pyswarms as ps



funcs = {"sphere": 0, "schwefel": 1, "rosenbrock": 2, "rastrigin": 3, "griewank": 4, "ackley": 5}
funcs_dispname = {"sphere": "F1 : Shifted Sphere Function",
                  "schwefel": "F2 : Schwefel’s Problem 2.21",
                  "rosenbrock": "F3 : Shifted Rosenbrock’s Function",
                  "rastrigin": "F4 : Shifted Rastrigin’s Function",
                  "griewank": "F5 : Shifted Griewank’s Function",
                  "ackley": "F6 : Shifted Ackley’s Function"}


def read_values(func):
    val_path = os.path.join('../data/', func + '.csv')
    bias_path = '../data/f_bias.csv'
    ss_path = '../data/search_space.csv'

    func_df = pd.read_csv(val_path)
    bias_df = pd.read_csv(bias_path)
    searchspace_df = pd.read_csv(ss_path)

    funcval = func_df.fvalue.values
    funcbias = bias_df.fvalue.values[funcs[func]]
    search_space = list(searchspace_df.iloc[funcs[func],])
    return funcval, funcbias, search_space


@jit(nopython=True)
def eval_cost(x, dim):
    if func_name == "sphere":
        F = 0
        for i in range(dim - 1):
            z = x[i] - funcval[i]
            F += z * z
        result = F + funcbias
    elif func_name == "schwefel":
        F = abs(x[0] - funcval[0])
        for i in range(1, dim - 1):
            z = x[i] - funcval[i]
            F = max(F, abs(z))
        result = F + funcbias
    elif func_name == "rosenbrock":
        F = 0
        y = np.empty(dim)
        for i in range(dim - 1):
            y[i] = x[i] - funcval[i] + 1
        for i in range(dim - 2):
            F += 100 * ((y[i] ** 2 - y[i + 1]) ** 2) + (y[i] - 1) ** 2
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


class My_problem:
    def __init__(self, dim, bounds, glob_opt):
        self.lower_bound = bounds[0]
        self.upper_bound = bounds[1]
        self.dimension = dim
        self.fitness_per_eval = []
        self.glob_opt = glob_opt

    def fitness(self, x):
        result = abs(eval_cost(x, self.dimension) - funcbias)
        return [result]

    def get_bounds(self):
        x_min = self.lower_bound * np.ones(self.dimension)
        x_max = self.upper_bound * np.ones(self.dimension)
        return x_min, x_max


def solve_pb(dim, my_algo, bounds, optim, popsize, pop=None):
    prob = pg.problem(My_problem(dim, bounds, optim))
    if pop is None:
        pop = pg.population(prob, popsize)
    my_algo.set_verbosity(100)
    t1 = time.time()
    pop = my_algo.evolve(pop)
    t2 = time.time()

    time_diff = t2 - t1
    if my_algo.get_name().split(":")[0] == "PSO":
        extract_algo = my_algo.extract(pg.pso)
    elif my_algo.get_name().split(":")[0] == "GPSO":
        extract_algo = my_algo.extract(pg.pso_gen)
    elif my_algo.get_name().split(":")[0] == "DE":
        extract_algo = my_algo.extract(pg.de)
    elif my_algo.get_name().split(":")[0] == "saDE":
        extract_algo = my_algo.extract(pg.sade)
    elif my_algo.get_name().split(":")[0] == "sa-DE1220":
        extract_algo = my_algo.extract(pg.de1220)
    elif my_algo.get_name().split(":")[0] == "SGA":
        extract_algo = my_algo.extract(pg.sga)
    elif my_algo.get_name().split(":")[0] == "ABC":
        extract_algo = my_algo.extract(pg.bee_colony)
    elif my_algo.get_name().split(":")[0] == "xNES":
        extract_algo = my_algo.extract(pg.xnes)
    elif my_algo.get_name().split(":")[0] == "CMA-ES":
        extract_algo = my_algo.extract(pg.cmaes)
    elif my_algo.get_name().split(":")[0] == "SA":
        extract_algo = my_algo.extract(pg.simulated_annealing)

    log = extract_algo.get_log()
    curve = [x[2] for x in log]
    niter = log[-1][0]

    return pop, curve, niter, time_diff


def print_solution(dimension, my_algo, pop_evolved, log, niter, duration):
    algorithm_name = my_algo.get_name()
    parameters = my_algo.get_extra_info()
    solution_x = pop_evolved.champion_x
    fitness = pop_evolved.champion_f[0]
    n_evals = pop_evolved.problem.get_fevals()
    print('-' * 60)
    print("Function: %s" % funcs_dispname[func_name])
    print("Problem dimension: %d" % dimension)
    print("Search Space : ", search_space)
    print("Global Optimum: %.2f" % funcbias)
    print('-' * 60)
    print("Algorithm: %s" % algorithm_name)
    print("Parameters: \n%s" % parameters)
    print('-' * 60)
    print("Fitness: %f" % fitness)
    print("Solution: ")
    with pd.option_context('display.max_rows', 8):
        print(pd.DataFrame(solution_x, columns=['X']))
    print('-' * 60)
    print("Nb of functions evaluations: %d" % n_evals)
    print("Stopping criterion: after %d iterations" % niter)
    print("computational time: %.3f seconds" % duration)

    plt.plot(log)
    plt.xlabel("iterations (x100)")
    plt.ylabel("fitness: f(x)-f(x*)")
    plt.show()

def fitness(parts):
    fit = np.zeros(parts.shape[0])
    for x in range(parts.shape[0]):
        fit[x] = abs(eval_cost(parts[x],DIM)-funcbias)

    return fit


# function to choose: sphere, schwefel, rosenbrock, rastrigin, griewank, ackley
func_name = 'rosenbrock'
funcval, funcbias, search_space = read_values(func_name)

DIM = 500


# Create bounds
x_min = search_space[0] * np.ones(DIM)
x_max = search_space[1] * np.ones(DIM)
bounds = (x_min, x_max)

ini = np.zeros((250,DIM))

# Initialize swarm
options = {'c1': 2.05, 'c2': 5, 'w':0.5}

# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=250, dimensions=DIM, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(fitness, iters=4000)



#
# options = {'c1': [1.3, 1.8],
#             'c2': [1.8, 2.2],
#             'w' : [.4, 1],
#             'k' : [3, 10],
#             'p':2}
#
# from pyswarms.utils.search import RandomSearch
# g = RandomSearch(ps.single.LocalBestPSO, n_particles=100, dimensions=DIM,
#                    options=options, bounds=bounds,objective_func=fitness, n_selection_iters=2, iters=5000)
# best_score, best_options = g.search()
# print(best_score,best_options)