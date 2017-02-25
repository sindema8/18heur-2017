import numpy as np


class ObjFun(object):
    """Generic objective function super-class."""

    def __init__(self, fstar, a, b):
        """
        Default initialization function (for discrete tasks) that sets:
        * f* value to be reached (can be -inf)
        * a: lower bound vector
        * b: upper bound vector
        """
        self.fstar = fstar
        self.a = a
        self.b = b

    def get_fstar(self):
        return self.fstar

    def get_bounds(self):
        return [self.a, self.b]

    def generate_point(self):
        raise NotImplementedError("Objective function must implement its own random point generation")

    def get_neighborhood(self, x):
        return x

    def evaluate(self, x):
        raise NotImplementedError("Objective function must implement its own evaluation")


class AirShip(ObjFun):

    """1-dimensional demo task from the first excercise."""

    def __init__(self):
        fstar = -100
        a = 0
        b = 799
        super().__init__(fstar, a, b)

    def generate_point(self):
        return np.random.randint(0, 800)

    def get_neighborhood(self, x, d):
        left = [x for x in np.arange(x-1, x-d-1, -1, dtype=int) if x >= 0]
        right = [x for x in np.arange(x+1, x+d+1, dtype=int) if x < 800]
        if np.size(left) == 0:
            return right
        elif np.size(right) == 0:
            return left
        else:
            return np.concatenate((left, right))

    def evaluate(self, x):
        px = np.array([0,  50, 100, 300, 400, 700, 799], dtype=int)
        py = np.array([0, 100,   0,   0,  25,   0,  50], dtype=int)
        xx = np.arange(0, 800)
        yy = np.interp(xx, px, py)
        return -yy[x]  # negative altitude, becase we are minimizing (to be consistent with other obj. functions)


class Sum(ObjFun):

    def __init__(self, a, b):
        self.n = np.size(a)  # dimension
        super().__init__(fstar=0, a=a, b=b)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i]) for i in np.arange(self.n)]

    def get_neighborhood(self, x, d):
        assert d == 1, "Sum(x) supports neighbourhood with distance = 1 only"
        nd = []
        for i in np.arange(self.n):
            if x[i] > self.a[i]:
                xx = x.copy()
                xx[i] -= 1
                nd.append(xx)
            if x[i] < self.b[i]:
                xx = x.copy()
                xx[i] += 1
                nd.append(xx)
        return nd

    def evaluate(self, x):
        return np.sum(x)
