from numba import jit
import math

@jit(nopython=True)
def sphere(x, dim, funcval, funcbias):
    F = 0
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        F += z * z
    result = F + funcbias
    return result


@jit(nopython=True)
def schwevel(x, dim, funcval, funcbias):
    F = abs(x[0])
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        F = max(F, abs(z))
    result = F + funcbias[1]
    return result

@jit(nopython=True)
def rosenbrock(x, dim, funcval, funcbias):
    F = 0
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        F += z * z
    result = F + funcbias
    return result


@jit(nopython=True)
def rastrigin(x, dim, funcval, funcbias):
    F = 0
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        F += z ** 2 - 10 * math.cos(2 * math.pi * z) + 10
    result = F + funcbias
    return result


@jit(nopython=True)
def griewank(x, dim, funcval, funcbias):
    F1 = 0
    F2 = 1
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        F1 += z ** 2 / 4000
        F2 += math.cos(z / math.sqrt(i + 1))
    result = F1 - F2 + 1 + funcbias
    return result


@jit(nopython=True)
def ackley(x, dim, funcval, funcbias):
    Sum1 = 0
    Sum2 = 0
    for i in range(dim - 1):
        z = x[i] - funcval[i]
        Sum1 += z ** 2
        Sum2 += math.cos(2 * math.pi * z)
    result = -20 * math.exp(-0.2 * math.sqrt(Sum1 / dim)) - math.exp(Sum2 / dim) + 20 + math.e + funcbias
    return result