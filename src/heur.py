import numpy as np
import pandas as pd


class StopCriterion(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Heuristic:

    def __init__(self, of, maxeval):
        self.of = of
        self.maxeval = maxeval
        self.best_y = np.inf
        self.best_x = None
        self.neval = 0
        self.log_data = []

    def evaluate(self, x):
        y = self.of.evaluate(x)
        self.neval += 1
        if y < self.best_y:
            self.best_y = y
            self.best_x = x
        if y <= self.of.get_fstar():
            raise StopCriterion('Found solution with desired fstar value')
        if self.neval == self.maxeval:
            raise StopCriterion('Exhausted maximum allowed number of evaluations')
        return y

    def log(self, data):
        self.log_data.append(data)

    def report_end(self):
        return {
            'best_y': self.best_y,
            'best_x': self.best_x,
            'neval': self.neval if self.best_y <= self.of.get_fstar() else np.inf,
            'log_data': pd.DataFrame(self.log_data)
        }


class ShootAndGo(Heuristic):

    def __init__(self, of, maxeval, hmax=np.inf, random_descent=False):
        Heuristic.__init__(self, of, maxeval)
        self.hmax = hmax
        self.random_descent = random_descent

    def steepest_descent(self, x):
        # Steepest/Random Hill Descent beginning in x
        desc_best_y = np.inf
        desc_best_x = x
        h = 0
        go = True
        while go and h < self.hmax:
            go = False
            h += 1

            neighborhood = self.of.get_neighborhood(desc_best_x, 1)
            if self.random_descent:
                np.random.shuffle(neighborhood)

            for xn in neighborhood:
                yn = self.evaluate(xn)
                if yn < desc_best_y:
                    desc_best_y = yn
                    desc_best_x = xn
                    go = True
                    if self.random_descent:
                        break

    def search(self):
        try:
            while True:
                # Random Shoot...
                x = self.of.generate_point()  # global search
                self.evaluate(x)
                # ...and Go (optional)
                if self.hmax > 0:
                    self.steepest_descent(x)  # local search

        except StopCriterion:
            return self.report_end()
        except:
            raise


class FastSimulatedAnnealing(Heuristic):

    def __init__(self, of, maxeval, T0, n0, alpha, r):
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.r = r

    def mutate(self, x):
        # Discrete Cauchy mutation
        n = np.size(x)
        u = np.random.uniform(low=0.0, high=1.0, size=n)
        r = self.r
        x_new = x + r*np.tan(np.pi * (u-1/2))

        x_new_corrected = np.minimum(np.maximum(x_new, self.of.a), self.of.b)  # trivial mutation correction (for now)
        return np.array(np.round(x_new_corrected), dtype=int)

    def search(self):
        try:
            x = self.of.generate_point()
            f_x = self.evaluate(x)
            while True:
                k = self.neval - 1  # because of the first obj. fun. evaluation
                T0 = self.T0
                n0 = self.n0
                alpha = self.alpha
                T = T0 / (1 + (k / n0) ** alpha) if alpha > 0 else T0 * np.exp(-(k / n0) ** -alpha)

                y = self.mutate(x)
                f_y = self.evaluate(y)
                s = (f_x - f_y)/T
                swap = np.random.uniform() < 1/2 + np.arctan(s)/np.pi
                Heuristic.log(self, {'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, 'T': T, 'swap': swap})
                if swap:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise

