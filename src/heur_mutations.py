from utils import is_integer
import numpy as np


class Correction:
    """
    Baseline mutation correction strategy - "sticks" the solution to domain boundaries
    """

    def __init__(self, of):
        self.of = of

    def correct(self, x):
        return np.minimum(np.maximum(x, self.of.a), self.of.b)


class MirrorCorrection(Correction):
    """
    Mutation correction via mirroring
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        n = np.size(x)
        d = self.of.b - self.of.a
        for k in range(n):
            if d[k] == 0:
                x[k] = self.of.a[k]
            else:
                de = np.mod(x[k] - self.of.a[k], 2 * d[k])
                de = np.amin([de, 2 * d[k] - de])
                x[k] = self.of.a[k] + de
        return x


class ExtensionCorrection(Correction):
    """
    Mutation correction via periodic domain extension
    """

    def __init__(self, of):
        Correction.__init__(self, of)

    def correct(self, x):
        d = self.of.b - self.of.a
        x = self.of.a + np.mod(x - self.of.a, d + (1 if is_integer(x) else 0))
        return x


class Mutation:

    def __init__(self, correction):
        self.correction = correction


class CauchyMutation(Mutation):
    """
    Discrete Cauchy mutation
    """

    def __init__(self, r, correction):
        Mutation.__init__(self, correction)
        self.r = r

    def mutate(self, x):
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r * np.tan(np.pi * (u - 1 / 2))
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected


class GaussMutation(Mutation):
    """
    Discrete Gaussian mutation
    """

    def __init__(self, sigma, correction):
        Mutation.__init__(self, correction)
        self.sigma = sigma

    def mutate(self, x):
        n = np.size(x)
        x_new = np.random.normal(x, self.sigma, size=n)
        if is_integer(x):
            x_new = np.array(np.round(x_new), dtype=int)
        x_new_corrected = self.correction.correct(x_new)
        return x_new_corrected


if __name__ == "__main__":
    pass
