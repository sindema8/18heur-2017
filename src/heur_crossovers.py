import numpy as np


class Crossover:
    """
    Baseline crossover  - randomly chooses "genes" from parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):
        z = np.array([x[i] if np.random.uniform() < 0.5 else y[i] for i in np.arange(x.size)], dtype=int)
        return z


class UniformMultipoint(Crossover):
    """
    Uniform n-point crossover
    """

    def __init__(self, n):
        self.n = n  # number of crossover points

    def crossover(self, x, y):
        co_n = self.n + 1
        n = np.size(x)
        z = x*0
        k = 0
        p = np.ceil(n/co_n).astype(int)
        for i in np.arange(1, co_n+1):
            ix_from = k
            ix_to = np.minimum(k+p, n)
            z[ix_from:ix_to] = x[ix_from:ix_to] if np.mod(i, 2) == 1 else y[ix_from:ix_to]
            k += p
        return z


class RandomCombination(Crossover):
    """
    Randomly combines parents
    """

    def __init__(self):
        pass

    def crossover(self, x, y):
        z = np.array([np.random.randint(np.min([x[i], y[i]]), np.max([x[i], y[i]]) + 1) for i in np.arange(x.size)],
                     dtype=int)
        return z
