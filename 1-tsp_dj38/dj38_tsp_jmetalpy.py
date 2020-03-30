from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm

from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.operator.crossover import PMXCrossover

from jmetal.operator import BinaryTournamentSelection
from jmetal.operator import BinaryTournament2Selection
from jmetal.operator import BestSolutionSelection
from jmetal.operator import RankingAndCrowdingDistanceSelection

from jmetal.util.comparator import MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking

from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.termination_criterion import StoppingByQualityIndicator

import math
import random
import re
import pandas as pd
import matplotlib.pyplot as plt

from jmetal.core.problem import PermutationProblem
from jmetal.core.solution import PermutationSolution

eval_fitness = []


class myTSP(PermutationProblem):
    def __init__(self, instance: str = None):
        super(myTSP, self).__init__()

        distance_matrix, number_of_cities = self.__read_from_file(instance)
        self.distance_matrix = distance_matrix
        self.number_of_variables = number_of_cities

        self.obj_directions = [self.MINIMIZE]
        self.number_of_objectives = 1
        self.number_of_constraints = 0

    def __read_from_file(self, filename: str):
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
            for item in data:
                if item[0].isdigit():
                    j, city_a, city_b = [float(x.strip()) for x in item.split(' ')]
                    c[2 * (int(j) - 1)] = city_a
                    c[2 * (int(j) - 1) + 1] = city_b
            matrix = [[-1] * dimension for _ in range(dimension)]
            for k in range(dimension):
                matrix[k][k] = 0
                for j in range(k + 1, dimension):
                    dist = math.sqrt((c[k * 2] - c[j * 2]) ** 2 + (c[k * 2 + 1] - c[j * 2 + 1]) ** 2)
                    dist = round(dist)
                    matrix[k][j] = dist
                    matrix[j][k] = dist
            return matrix, dimension

    def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
        fitness = 0

        for i in range(self.number_of_variables - 1):
            x = solution.variables[i]
            y = solution.variables[i + 1]
            fitness += self.distance_matrix[x][y]

        first_city, last_city = solution.variables[0], solution.variables[-1]
        fitness += self.distance_matrix[first_city][last_city]

        solution.objectives[0] = fitness

        eval_fitness.append(fitness)
        return solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(number_of_variables=self.number_of_variables,
                                           number_of_objectives=self.number_of_objectives)
        new_solution.variables = random.sample(range(self.number_of_variables), k=self.number_of_variables)
        return new_solution

    @property
    def number_of_cities(self):
        return self.number_of_variables

    def get_name(self):
        return 'Symmetric TSP'


myproblem = myTSP(instance='./dj38.tsp')

# print(problem.create_solution())

print('Cities: ', myproblem.number_of_variables)

# parameters

popsize = 10 * myproblem.number_of_variables
offspring = 4 * myproblem.number_of_variables
mut = PermutationSwapMutation(0.1)
cross = PMXCrossover(0.6)
# select = BinaryTournamentSelection(
#     MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()]))
select = BestSolutionSelection()
# select = BinaryTournament2Selection([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])

termin = StoppingByEvaluations(max_evaluations=500*popsize)

algorithm = GeneticAlgorithm(
    problem=myproblem,
    population_size=popsize,
    offspring_population_size=offspring,
    mutation=mut,
    crossover=cross,
    selection=select,
    termination_criterion=termin
)

algorithm.run()
result = algorithm.get_result()

# - The chosen algorithm and a justification of this choice
print('Algorithm: {}'.format(algorithm.get_name()))
print('Problem: {}'.format(myproblem.get_name()))
# - The parameters of the algorithm

# - The final results, both solution and fitness
print('Solution: {}'.format(result.variables))
print('Fitness: {}'.format(result.objectives[0]))
# - The number of function evaluations
print('Nb of evaluations: {}'.format(len(eval_fitness)))
# - The stopping criterion

# - The computational time
print('Computing time: {}'.format(algorithm.total_computing_time))
# - The convergence curve (fitness as a function of time)
plt.plot(eval_fitness, 'r.')
plt.title('Fitness: %d' % result.objectives[0])
plt.show()
