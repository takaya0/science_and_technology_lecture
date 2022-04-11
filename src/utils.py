import numpy as np


def get_random_sym_matrix(n):
    '''
    Arg:
        n(int, >0) : matrix row and column size
    Return:
        S(ndarray, [n, n]) : Symmetric matrix 
    '''

    def _sym(S):
        return 0.5 * (S + S.T)

    _S = np.random.randn(n, n)
    S = _sym(_S)
    return S
