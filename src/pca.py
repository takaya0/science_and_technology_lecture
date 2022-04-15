from pymanopt.manifolds import Stiefel
from pymanopt.solvers import SteepestDescent
from pymanopt import Problem

from matplotlib import pyplot as plt
from autograd import numpy as np


def BrocketCostOptimization(A, r=3):
    n = len(A)

    #N = np.diag(np.arange(1, r + 1, 1))
    N = np.diag(np.linspace(1, 30, r))
    print(N)

    def cost(x):
        retval = np.trace(np.dot(np.dot(np.dot(x.T, A), x), N))
        #retval = np.sum((A @ x) * (x * N))
        return retval

    prob = Problem(Stiefel(n, r), cost=cost)
    solver = SteepestDescent(maxiter=3000, logverbosity=2)
    Xopt, log = solver.solve(problem=prob)

    losses = log['iterations']['f(x)']
    times = log['iterations']['time']
    return losses


def main():
    N = 100
    A = np.diag([n + 1 for n in range(0, N)])
    losses = BrocketCostOptimization(A, 6)
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
