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

    def __init__(self, of, maxeval, T0, n0, alpha, mutation):
        Heuristic.__init__(self, of, maxeval)

        self.T0 = T0
        self.n0 = n0
        self.alpha = alpha
        self.mutation = mutation

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

                y = self.mutation.mutate(x)
                f_y = self.evaluate(y)
                s = (f_x - f_y)/T
                swap = np.random.uniform() < 1/2 + np.arctan(s)/np.pi
                self.log({'step': k, 'x': x, 'f_x': f_x, 'y': y, 'f_y': f_y, 'T': T, 'swap': swap})
                if swap:
                    x = y
                    f_x = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise


class GeneticOptimization(Heuristic):

    def __init__(self, of, maxeval, N, M, Tsel1, Tsel2, mutation, crossover):
        Heuristic.__init__(self, of, maxeval)

        assert M > N, 'M should be larger than N'
        self.N = N  # population size
        self.M = M  # working population size
        self.Tsel1 = Tsel1  # first selection temperature
        self.Tsel2 = Tsel2  # second selection temperature
        self.mutation = mutation
        self.crossover = crossover

    @staticmethod
    def sort_pop(pop_x, pop_f):
        ixs = np.argsort(pop_f)
        pop_x = pop_x[ixs]
        pop_f = pop_f[ixs]
        return [pop_x, pop_f]

    @staticmethod
    def rank_select(temp, n_max):
        u = np.random.uniform(low=0.0, high=1.0, size=1)
        ix = np.minimum(np.ceil(-temp*np.log(u)), n_max)-1
        return ix.astype(int)

    def search(self):
        try:
            # Initialization:
            pop_X = np.zeros([self.N, np.size(self.of.a)], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population fitness (objective) function values
            # a.) generate the population
            for i in np.arange(self.N):
                x = self.of.generate_point()
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # b.) sort according to fitness function
            [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

            # Evolution iteration
            while True:
                # 1.) generate the working population
                work_pop_X = np.zeros([self.M, np.size(self.of.a)], dtype=self.of.a.dtype)
                work_pop_f = np.zeros(self.M)
                for i in np.arange(self.M):
                    parent_a_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # select first parent
                    parent_b_ix = self.rank_select(temp=self.Tsel1, n_max=self.N)  # 2nd --//-- (not unique!)
                    par_a = pop_X[parent_a_ix, :][0]
                    par_b = pop_X[parent_b_ix, :][0]
                    z = self.crossover.crossover(par_a, par_b)
                    z_mut = self.mutation.mutate(z)
                    work_pop_X[i, :] = z_mut
                    work_pop_f[i] = self.evaluate(z_mut)

                # 2.) sort working population according to fitness function
                [work_pop_X, work_pop_f] = self.sort_pop(work_pop_X, work_pop_f)

                # 3.) select the new population
                ixs_not_selected = np.ones(self.M, dtype=bool)  # this mask will prevent us from selecting duplicates
                for i in np.arange(self.N):
                    sel_ix = self.rank_select(temp=self.Tsel2, n_max=np.sum(ixs_not_selected))
                    pop_X[i, :] = work_pop_X[ixs_not_selected][sel_ix, :]
                    pop_f[i] = work_pop_f[ixs_not_selected][sel_ix]
                    ixs_not_selected[sel_ix] = False

                # 4.) sort according to fitness function
                [pop_X, pop_f] = self.sort_pop(pop_X, pop_f)

        except StopCriterion:
            return self.report_end()
        except:
            raise


class DifferentialEvolution(Heuristic):

    def __init__(self, of, maxeval, N, CR, F):
        Heuristic.__init__(self, of, maxeval)
        assert N >= 4, 'N should be at least equal to 4'
        self.N = N
        self.CR = CR
        assert 0 <= F <= 2, 'F should be from [0; 2]'
        self.F = F

    def search(self):
        try:
            # Initialization
            n = np.size(self.of.a)
            pop_X = np.zeros([self.N, n], dtype=self.of.a.dtype)  # population solution vectors
            pop_f = np.zeros(self.N)  # population fitness (objective) function values
            for i in np.arange(self.N):
                x = self.of.generate_point()
                pop_X[i, :] = x
                pop_f[i] = self.evaluate(x)

            # Evolution iteration
            while True:
                for i in range(self.N):
                    x = pop_X[i]
                    agents = np.random.choice(np.delete(np.arange(self.N), i), 3, replace=False)  # selected 3 agents
                    a, b, c = pop_X[agents[0]], pop_X[agents[1]], pop_X[agents[2]]
                    R = np.random.randint(low=0, high=self.N)
                    y = [a[j] + self.F * (b[j] - c[j]) if np.random.rand() < self.CR or j == R else x[j] for j in range(n)]
                    f_y = self.evaluate(y)
                    if f_y < pop_f[i]:
                        pop_X[i] = y
                        pop_f[i] = f_y

        except StopCriterion:
            return self.report_end()
        except:
            raise


if __name__ == "__main__":
    pass
