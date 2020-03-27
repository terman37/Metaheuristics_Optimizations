import pandas as pd
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
import time
from numba import jit

rosenbrockdf = pd.read_csv('../data/rosenbrock.csv')
f_biasdf = pd.read_csv('../data/f_bias.csv')

rosenbrock = rosenbrockdf.fvalue.values
f_bias = f_biasdf.fvalue.values

f_xstar = f_bias[2]
search_space = (-100, 100)


class shifted_rosen:

    def __init__(self, dim, search_space, glob_opt):
        self.lower_bound = search_space[0]
        self.upper_bound = search_space[1]
        self.dimension = dim
        self.fitness_per_eval = []
        self.glob_opt = glob_opt

    def fitness(self, x):
        F = 0
        z = np.empty(self.dimension)
        for i in range(self.dimension - 1):
            z[i] = x[i] - rosenbrock[i] + 1

        for i in range(self.dimension - 2):
            F += 100 * ((z[i] ** 2 - z[i + 1]) ** 2) + (z[i] - 1) ** 2
        result = F + f_xstar
        return [result]

    def get_bounds(self):
        x_min = self.lower_bound * np.ones(self.dimension)
        x_max = self.upper_bound * np.ones(self.dimension)
        return x_min, x_max

    def get_name(self):
        return "Shifted RosenBrock Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dimension)


DIM = 50
pg.set_global_rng_seed(seed=37)

prob = pg.problem(shifted_rosen(DIM, search_space, f_xstar))
print(prob)

algo = pg.algorithm(pg.pso(gen=100))
# algo = pg.algorithm(pg.nlopt(solver='neldermead'))
algo.set_verbosity(1)

pop = pg.population(prob, 10)

t1 = time.time()
pop = algo.evolve(pop)
t2 = time.time()

extract_algo = algo.extract(pg.pso)
log = extract_algo.get_log()

print("Algorithm: %s" % prob.get_name())
# print("Parameters: %s" % options)
print("Solution: ", pop.champion_x)
print("Fitness: %f" % pop.champion_f[0])
# print("Nb of functions evaluations: %d in %d iterations" % (result.nfev,result.nit))
print("Nb of functions evaluations: %d in %d iterations" % (prob.get_fevals(), 0))
# print("Stopping criterion: %s" % result.message)

duration = t2 - t1
print("computanional time: %.3f seconds" % duration)

gbest = [x[2] for x in log]
plt.plot(gbest)
print(gbest)
plt.show()
print(f_xstar)
