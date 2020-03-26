import pandas as pd
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time
from numba import jit

# --------CUSTOMIZE FOR EACH FUNCTION --------
# parameters
func_name = 'sphere'
bias_id = 0
search_space = (-100, 100)


# function definition
@jit(nopython=True)
def eval_fitness(x, dim):
    F = 0
    for i in range(dim - 1):
        z = x[i] - rosenbrock[i]
        F += z * z
    result = F + f_xstar
    return result

# create a file to call for functions
# --------------------------------------------

# get values for function
rosenbrockdf = pd.read_csv('../data/' + func_name + '.csv')
f_biasdf = pd.read_csv('../data/f_bias.csv')

rosenbrock = rosenbrockdf.fvalue.values
f_bias = f_biasdf.fvalue.values

f_xstar = f_bias[bias_id]


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


def solve_pb(dim, myalgo, bounds, optim, pop_size):
    prob = pg.problem(My_problem(dim, bounds, optim))
    pop = pg.population(prob, pop_size)
    myalgo.set_verbosity(1)
    t1 = time.time()
    pop = algo.evolve(pop)
    t2 = time.time()

    # define what to return
    # solution = {algorithm, parameters, solution, fitness, nb_eval, stop_criterion, compute_time, convergence_curve}

    # extract_algo = algo.extract(pg.nlopt)
    # log = extract_algo.get_log()

    solution = 0
    return solution


def print_solution(mysoluc):
    print("Algorithm: %s" % prob.get_name())
    # print("Parameters: %s" % options)
    print("Solution: ", pop.champion_x)
    print("Fitness: %f" % pop.champion_f[0])
    # print("Nb of functions evaluations: %d in %d iterations" % (result.nfev,result.nit))
    print("Nb of functions evaluations: %d in %d iterations" % (prob.get_fevals(), 0))
    # print("Stopping criterion: %s" % result.message)

    duration = t2 - t1
    print("computanional time: %.3f seconds" % duration)

    gbest = [x[1] for x in log]
    plt.plot(gbest)
    print(gbest)
    plt.show()
    print(f_xstar)

# Solve problem in dimension 50
DIM = 50
# algo = pg.algorithm(pg.pso(gen=10))
algo = pg.algorithm(pg.nlopt(solver='neldermead'))
x = solve_pb(DIM, algo, search_space, f_xstar, 1)
