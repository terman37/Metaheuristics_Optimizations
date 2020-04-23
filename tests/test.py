import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
import pygmo as pg

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


# function to choose: sphere, schwefel, rosenbrock, rastrigin, griewank, ackley
func_name = 'schwefel'
funcval, funcbias, search_space = read_values(func_name)

DIM = 500

gen = 40000
pop_size = 250

# algo = pg.algorithm(pg.pso_gen(gen=gen,
#                            omega=0.73,
#                            eta1=0.5, #social
#                            eta2=2.5, #cognitive
#                            max_vel=0.7,
#                            variant=1,
#                            neighb_type=1,
#                            neighb_param=5,
#                            memory=False,
#                            seed=37))

# algo = pg.algorithm(pg.de(gen=gen, F=0.1, CR=0.85, variant=2, ftol=1e-06, xtol=1e-06, seed=37))

# algo = pg.algorithm(pg.xnes(gen=gen, eta_mu=- 1, eta_sigma=- 1, eta_b=- 1, sigma0=- 1, ftol=1e-06, xtol=1e-06, memory=False, force_bounds=False, seed=37))
# algo = pg.algorithm(pg.cmaes(gen=gen, cc=- 1, cs=- 1, c1=- 1, cmu=- 1, sigma0=0.5, ftol=1e-06, xtol=1e-06, memory=False, force_bounds=False, seed=37))
# algo = pg.algorithm(pg.sade(gen=gen,
#                             variant=6,
#                             variant_adptv=1,
#                             ftol=1e-03,
#                             xtol=1e-03,
#                             memory=True,
#                             seed=37))
#
# algo = pg.algorithm(pg.de1220(gen=gen,
#                               allowed_variants=[2, 3, 7, 10, 13, 14, 15, 16,1,4,5,6,8,9,11,12,17,18],
#                               variant_adptv=1,
#                               ftol=1e-03,
#                               xtol=1e-03,
#                               memory=False,
#                               seed=37))

# algo = pg.algorithm(pg.bee_colony(gen=gen))

# algo = pg.algorithm(pg.simulated_annealing(Ts=10.0,
#                                            Tf=0.001,
#                                            n_T_adj=100,
#                                            n_range_adj=10,
#                                            bin_size=20,
#                                            start_range=1.0,
#                                            seed=37))

algo = pg.algorithm(pg.sga(gen=gen,
                           cr=0.85,
                           eta_c=1.0,  # distribution index for sbx crossover
                           m=0.02,
                           param_m=5,
                           # distribution index (polynomial mutation), gaussian width (gaussian mutation) or inactive (uniform mutation)
                           param_s=10,
                           # the number of best individuals to use in “truncated” selection or the size of the tournament in tournament selection.
                           crossover='sbx',  # One of exponential, binomial, single or sbx
                           mutation='polynomial',  # One of gaussian, polynomial or uniform
                           selection='truncated',  # One of tournament, truncated.
                           seed=37))
#
# algo = pg.algorithm(pg.sga(gen=gen,
#                            cr=0.8,
#                            eta_c=5.0, # distribution index for sbx crossover
#                            m=0.01,
#                            param_m=1, # distribution index (polynomial mutation), gaussian width (gaussian mutation) or inactive (uniform mutation)
#                            param_s=10, # the number of best individuals to use in “truncated” selection or the size of the tournament in tournament selection.
#                            crossover='sbx', # One of exponential, binomial, single or sbx
#                            mutation='polynomial', # One of gaussian, polynomial or uniform
#                            selection='truncated', # One of tournament, truncated.
#                            seed=37
#                           ))

print(algo.get_name().split(":")[0])
pop_evolv, logs, nit, compute_time = solve_pb(dim=DIM,
                                              my_algo=algo,
                                              bounds=search_space,
                                              optim=funcbias,
                                              popsize=pop_size)

# gen = 10000
# pop_size = 100
# algo = pg.algorithm(pg.pso(gen=gen,
#                            omega=0.7,
#                            eta1=1.5,
#                            eta2=2,
#                            max_vel=0.5,
#                            variant=6,
#                            neighb_type=3,
#                            neighb_param=4,
#                            memory=False,
#                            seed=37))
# algo = pg.algorithm(pg.de1220(gen=gen))
#
#
#
# pop_evolv, logs, nit, compute_time = solve_pb(dim=DIM,
#                                               my_algo=algo,
#                                               bounds=search_space,
#                                               optim=funcbias,
#                                               popsize=pop_size,
#                                               pop=pop_evolv)

print_solution(dimension=DIM,
               my_algo=algo,
               pop_evolved=pop_evolv,
               log=logs,
               niter=nit,
               duration=compute_time
               )

bestx = pop_evolv.champion_x
fitness = eval_cost(bestx, 500) - funcbias
print("best fitness:", fitness)
# print("\t\t%s" % algo.get_name())
# print("\t\tfitness: %.1f" % fitness)
