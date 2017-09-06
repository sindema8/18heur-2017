import numpy as np
import math

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
        self.n = np.size(a)  # dimension of the task
        super().__init__(fstar=0, a=a, b=b)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i]+1) for i in np.arange(self.n)]

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


class SumSinx(ObjFun):

    def __init__(self, a, b):
        self.n = np.size(a)  # dimension of the task
        super().__init__(fstar=0, a=a, b=b)

    def generate_point(self):
        return [np.random.randint(self.a[i], self.b[i]+1) for i in np.arange(self.n)]

    def get_neighborhood(self, x, d):
        assert d == 1, "SumSinx(x) supports neighbourhood with distance = 1 only"
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
        return np.sum(np.abs(x*np.sin(x)))


class TSPGrid(ObjFun):

    def __init__(self, par_a, par_b, norm=2):
        n = par_a * par_b  # number of cities

        # compute city coordinates
        grid = np.zeros((n, 2), dtype=np.int)
        for i in np.arange(par_a):
            for j in np.arange(par_b):
                grid[i * par_b + j] = np.array([i, j])

        # compute distances based on coordinates
        dist = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(i+1, n):
                dist[i, j] = np.linalg.norm(grid[i, :]-grid[j, :], norm)
                dist[j, i] = dist[i, j]

        self.fstar = n+np.mod(n, 2)*(2**(1/norm)-1)
        self.n = n
        self.dist = dist
        self.a = np.zeros(n-1, dtype=np.int)  # n-1 because the first city is pre-determined
        self.b = np.arange(n-2, -1, -1)

    def generate_point(self):
        return np.array([np.random.randint(self.a[i], self.b[i] + 1) for i in np.arange(self.n-1)], dtype=int)

    def decode(self, x):
        """
        Decodes solution vector into ordered list of visited cities, e.g:
         x = 1 2 2 1 0
        cx = 2 4 5 3 1
        """
        cx = np.zeros(self.n, dtype=np.int)  # the final tour
        ux = np.ones(self.n, dtype=np.int)  # used cities indices
        ux[0] = 0  # first city is used automatically
        c = np.cumsum(ux)  # cities to be included in the tour
        for k in np.arange(1, self.n):
            ix = x[k-1]+1  # order index of currently visited city
            cc = c[ix]  # currently visited city
            cx[k] = cc  # append visited city into final tour
            c = np.delete(c, ix)  # visited city can not be included in the tour any more
        return cx

    def tour_dist(self, cx):
        d = 0
        for i in np.arange(self.n):
            dx = self.dist[cx[i-1], cx[i]] if i>0 else self.dist[cx[self.n-1], cx[i]]
            d += dx
        return d

    def evaluate(self, x):
        cx = self.decode(x)
        return self.tour_dist(cx)

    def get_neighborhood(self, x, d):
        assert d == 1, "TSPGrid supports neighbourhood with distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            # x-lower
            if x[i] > self.a[i]:  # (!) mutation correction .. will be discussed later
                xl = x.copy()
                xl[i] = x[i]-1
                nd.append(xl)

            # x-upper
            if x[i] < self.b[i]:  # (!) mutation correction ..  -- // --
                xu = x.copy()
                xu[i] = x[i]+1
                nd.append(xu)

        return nd


class Zebra3(ObjFun):

    def __init__(self, d):
        self.fstar = 0
        self.d = d
        self.n = d*3
        self.a = np.zeros(self.n, dtype=int)
        self.b = np.ones(self.n, dtype=int)

    def generate_point(self):
        return np.random.randint(0, 1+1, self.n)

    def get_neighborhood(self, x, d):
        assert d == 1, "Zebra3 supports neighbourhood with (Hamming) distance = 1 only"
        nd = []
        for i, xi in enumerate(x):
            xx = x.copy()
            xx[i] = 0 if xi == 1 else 1
            nd.append(xx)
        return nd

    def evaluate(self, x):
        f = 0
        for i in np.arange(1, self.d+1):
            xr = x[(i-1)*3:i*3]
            s = np.sum(xr)
            if np.mod(i,2) == 0:
                if s == 0:
                    f += 0.9
                elif s == 1:
                    f += 0.6
                elif s == 2:
                    f += 0.3
                else:  # s == 3
                    f += 1.0
            else:
                if s == 0:
                    f += 1.0
                elif s == 1:
                    f += 0.3
                elif s == 2:
                    f += 0.6
                else:  # s == 3
                    f += 0.9
        f = self.n/3-f
        return f


class DeJong1(ObjFun):
    """
    De Jong function 1 (sphere)
    http://www.geatbx.com/docu/fcnindex-01.html#P89_3085
    """

    def __init__(self, n, eps=0.01):
        self.fstar = 0 + eps
        self.n = n
        self.a = -5.12*np.ones(self.n, dtype=np.float64)
        self.b = 5.12*np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return np.sum([x[i] ** 2 for i in range(self.n)])

class Rosenbrock(ObjFun):
    """
    Rosenbrock 2-dimensional function
    http://www.geatbx.com/docu/fcnindex-01.html#P129_5426

    """

    def __init__(self, n, eps=0.01):
        self.fstar = 0 + eps
        self.n = n
        self.a = -2.048 * np.ones(self.n, dtype=np.float64)
        self.b = 2.048 * np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return np.sum([(100 * ((x[i + 1]) - x[i] ** 2) ** 2 + (1 - x[i]) ** 2) for i in range(self.n - 1)])

class Schwefel(ObjFun):
    """
    Schwefel's function 2-dimensional
    http://www.geatbx.com/docu/fcnindex-01.html#P150_6749

    """

    def __init__(self, n, eps=0.01):
        self.fstar = -418.9829 * n  + eps
        self.n = n
        self.a = -500 * np.ones(self.n, dtype=np.float64)
        self.b = 500 * np.ones(self.n, dtype=np.float64)

    def generate_point(self):
        return np.array([np.random.uniform(self.a[i], self.b[i]) for i in np.arange(self.n)], dtype=np.float64)

    def evaluate(self, x):
        return np.sum([(-x[i] * math.sin(math.sqrt(math.fabs(x[i])))) for i in range(self.n)])


class HMeans(ObjFun):
    """
    Heuristic clustering inspired by k-means
    """

    def __init__(self, dim=2, n_clu=3, n_pts=15, eps=0.1):
        """
        Pseudo-randomly generates cluster centroids and data point matrix
        :param dim: dimension
        :param n_clu: number of clusters
        :param n_pts: number of data points per cluster
        :param eps: tolerance from f*
        """
        np.random.seed(10)
        C = np.random.uniform(size=(n_clu, dim))  # cluster centers

        # generate points for each cluster
        X = np.zeros((0, dim))  # data point matrix
        sigma = 0.05
        for c in C:
            P = np.zeros((n_pts, dim))
            for i in range(n_pts):
                P[i] = [np.random.normal(cc, sigma) for cc in c]
            X = np.concatenate((X, P), axis=0)

        self.n_clu = n_clu
        self.dim = dim
        a = X.min(axis=0)
        self.a = np.concatenate([a for i in range(n_clu)])  # repeat for each solution centroid
        b = X.max(axis=0)
        self.b = np.concatenate([b for i in range(n_clu)])  # -- // --
        self.X = X
        self.C = C
        x_centroids = self.encode_solution(C)
        self.fstar = self.evaluate(x_centroids) + eps

    def encode_solution(self, C):
        """
        Encodes centroids into solution usable by heuristics
        :param C: matrix with centroids
        :return: array
        """
        return C.flatten()

    def decode_solution(self, x):
        """
        Decodes heuristic solution into centroids 
        :param x: array
        :return: matrix with centroids
        """
        return np.reshape(x, (self.n_clu, self.dim))

    def generate_point(self):
        C = np.random.uniform(size=(self.n_clu, self.dim))  # randomly generates centroids
        return self.encode_solution(C)

    def evaluate(self, x):
        """
        Computes sum of squares of distances from data points to their nearest centroids
        """
        ssq = 0
        C = self.decode_solution(x)
        for x in self.X:  # for each data point:
            d = np.array([np.linalg.norm(c - x) for c in C])  # compute distances to centroids
            ix = d.argmin()  # index of the nearest centroid
            ssq += d[ix] ** 2  # add square of distance to the nearest centroid
        return ssq

    def get_cluster_labels(self, x):
        """
        Returns array with cluster labels [0; n_clu] for a given solution vector 
        """
        C = self.decode_solution(x)
        labels = np.zeros(self.X.shape[0], dtype=int)
        for i, x in enumerate(self.X):  # for each data point:
            d = np.array([np.linalg.norm(c - x) for c in C])  # compute distances to centroids
            ix = d.argmin()  # index of the nearest centroid
            labels[i] = ix
        return labels


if __name__ == "__main__":
    pass
