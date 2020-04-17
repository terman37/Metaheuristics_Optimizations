import numpy as np
import pandas as pd

from jmetal.core.problem import FloatProblem, FloatSolution
from jmetal.problem.singleobjective.unconstrained import Rastrigin

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing

from jmetal.operator import BestSolutionSelection, SimpleRandomMutation, SBXCrossover

from jmetal.operator import PolynomialMutation

from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

fitness_per_eval = []
fitness_per_iter = []

rosenbrockdf = pd.read_csv('../data/rosenbrock.csv')
f_biasdf = pd.read_csv('../data/f_bias.csv')

rosenbrock = rosenbrockdf.fvalue.values
f_bias = f_biasdf.fvalue.values

f_xstar = f_bias[2]
search_space = (-100, 100)


class MyRosen(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(MyRosen, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-100. for _ in range(number_of_variables)]
        self.upper_bound = [100. for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        dim = solution.number_of_variables
        F = 0
        z = np.empty(50)
        x = solution.variables
        for i in range(dim - 1):
            z[i] = x[i] - rosenbrock[i] + 1

        for i in range(dim - 2):
            F += 100 * ((z[i] ** 2 - z[i + 1]) ** 2) + (z[i] - 1) ** 2

        result = F + f_xstar

        solution.objectives[0] = result
        return solution

    def get_name(self) -> str:
        return 'Shifted RosenBrock'


problem = MyRosen(5)

max_evaluations = 2500

algorithm = SimulatedAnnealing(
    problem=problem,
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
)

algorithm.run()
result = algorithm.get_result()

# Save results to file
print_function_values_to_file(result, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
print_variables_to_file(result, 'VAR.' + algorithm.get_name() + "." + problem.get_name())

print('Algorithm: ' + algorithm.get_name())
print('Problem: ' + problem.get_name())
print('Solution: ' + str(result.variables[0]))
print('Fitness:  ' + str(result.objectives[0]))
print('Computing time: ' + str(algorithm.total_computing_time))
