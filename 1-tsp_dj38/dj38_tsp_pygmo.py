# install pygmo with conda
# conda config --add channels conda-forge
# conda install pygmo

# from PyGMO.util import tsp as tsputil
# from PYGMO.problem import tsp



# importing the XML file
weights = tsputil.read_tsplib('burma14.xml')

# printing the weights matrix
tsputil.print_matrix(weights)

# creating a tsp problem from the imported weights matrix
tsp_instance = tsp(weights)

# printing the tsp problem details to console
print tsp_instance


class myTSP(base):
    """
    Analytical function to maximize.

    USAGE: my_problem_max()
    """

    def __init__(self):
        super(myTSP,self).__init__(2)
        self.set_bounds(-10, 10)

        # We provide a list of the best known solutions to the problem
        self.best_x = [[1.0, -1.0], ]

    # Reimplement the virtual method that defines the objective function
    def _objfun_impl(self, x):
        f = -(1.0 - x[0]) ** 2 - 100 * (-x[0] ** 2 - x[1]) ** 2 - 1.0
        return (f, )

    # Reimplement the virtual method that compares fitnesses
    def _compare_fitness_impl(self, f1, f2):
        return f1[0] > f2[0]

    # Add some output to __repr__
    def human_readable_extra(self):
        return "\n\tMaximization problem"