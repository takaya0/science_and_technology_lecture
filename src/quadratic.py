from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from matplotlib import pyplot as plt
import autograd.numpy as np

import time

from utils import get_random_sym_matrix

image_path = '../Images/'


def RayleighQuotientsOptimization(N):

    A = get_random_sym_matrix(N)

    def cost(x):
        retval = np.dot(x.T, np.dot(A, x))
        return retval
    problem = Problem(Sphere(N), cost=cost)
    now = time.perf_counter()
    solver = SteepestDescent(
        maxiter=8000, logverbosity=2, mingradnorm=0.001, maxtime=5000)
    Xopt = solver.solve(problem)

    print(f'計算時間 : {time.perf_counter() - now }秒')

    losses = Xopt[1]['iterations']['f(x)']
    times = Xopt[1]['iterations']['time']
    return losses, times


def main():
    losses, times = RayleighQuotientsOptimization(N=5000)
    index = np.arange(1, len(losses) + 1)
    plt.plot(index, losses)
    plt.xlabel('Iterations')
    plt.ylabel('function value')

    plt.savefig(image_path + 'quad_5000.png')


if __name__ == "__main__":
    main()
