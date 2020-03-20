import pandas as pd
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt
from numba import jit

rosenbrockdf = pd.read_csv('../data/rosenbrock.csv')
f_biasdf = pd.read_csv('../data/f_bias.csv')

rosenbrock = rosenbrockdf.fvalue.values
f_bias = f_biasdf.fvalue.values

f_xstar = f_bias[2]
search_space = (-100, 100)


class shifted_rosen:

    def __init__(self, dim=50, search_space=(-100, 100), glob_opt=0):
        self.lower_bound = search_space[0]
        self.upper_bound = search_space[1]
        self.dimension = dim
        self.fitness_per_eval = []
        self.glob_opt = glob_opt

    def fitness(self, x):
        result = self.eval(x)
        self.fitness_per_eval.append(result - self.glob_opt)
        return [result]

    def eval(self, x):
        F = int(0)
        z = np.empty(50)
        for i in range(self.dimension - 1):
            z[i] = x[i] - rosenbrock[i] + 1

        for i in range(self.dimension - 2):
            F += 100 * ((z[i] ** 2 - z[i + 1]) ** 2) + (z[i] - 1) ** 2
        result = F + f_xstar
        return result

    def get_bounds(self):
        x_min = self.lower_bound * np.ones(self.dimension)
        x_max = self.upper_bound * np.ones(self.dimension)
        return x_min, x_max

    def get_fitness_per_eval(self):
        return self.fitness_per_eval

    def get_name(self):
        return "Shifted RosenBrock Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dimension)


DIM = 50
pg.set_global_rng_seed(seed=37)
# cur_x = np.random.uniform(-100, 100, (DIM))
# print(cur_x)

prob = pg.problem(shifted_rosen(DIM, search_space, f_xstar))
print(prob)
udp = prob.extract(shifted_rosen)

algo = pg.algorithm(pg.pso(gen=100))
algo.set_verbosity(1)

pop = pg.population(prob, 2)
pop = algo.evolve(pop)

log = algo.extract(pg.pso).get_log()
gbest = [x[2] for x in log]
plt.plot(gbest)
print(gbest)
plt.show()
print(f_xstar)
