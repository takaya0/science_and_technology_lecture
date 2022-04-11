from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from matplotlib import pyplot as plt
import autograd.numpy as np

from utils import get_random_sym_matrix


def RayleighQuotientsOptimization(N):

    A = get_random_sym_matrix(N)

    def cost(x):
        retval = np.dot(x.T, np.dot(A, x))
        return retval
    problem = Problem(Sphere(N), cost=cost)
    solver = SteepestDescent(
        maxiter=8000, logverbosity=2, mingradnorm=2.0 * 1e-1, maxtime=5000)
    Xopt = solver.solve(problem)
    losses = Xopt[1]['iterations']['f(x)']
    times = Xopt[1]['iterations']['time']
    return losses, times


def main():

    experiment1_losses = RayleighQuotientsOptimization(N=10)
    experiment2_losses = RayleighQuotientsOptimization(N=100)
    experiment3_losses = RayleighQuotientsOptimization(N=1000)


if __name__ == "__main__":
    main()
