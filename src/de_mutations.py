# from utils import is_integer
import numpy as np


class Mutation:
    def __init__(self, F, N, CR):
        self.F = F
        self.N = N
        self.CR = CR
        pass


class deRand(Mutation):
    """
    DERAND1
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)

    def mutate(self, i, population, pop_values):
        x = population[i]
        # dimension of the problem
        n = np.size(x)
        # choose 3 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 3, replace=False)
        a, b, c = population[agents[0]], population[agents[1]], population[agents[2]]
        R = np.random.randint(low=0, high=self.N)
        y = [a[j] + self.F * (b[j] - c[j]) if np.random.rand() < self.CR or j == R else x[j] for j in range(n)]
        return y


class deRand2(Mutation):
    """
    DERAND2
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)
        assert self.N > 5

    def mutate(self, i, pop, pop_values):
        x = pop[i]
        # dimension of the problem
        n = np.size(x)
        # choose 5 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 5, replace=False)
        a, b, c, d, e = pop[agents[0]], pop[agents[1]], pop[agents[2]], pop[agents[3]], pop[agents[4]]
        R = np.random.randint(low=0, high=self.N)
        y = [a[j] + self.F * (b[j] - c[j]) + self.F * (d[j] - e[j]) if np.random.rand() < self.CR or j == R else x[j]
             for j in range(n)]
        return y


class deBest(Mutation):
    """
    DEBEST1
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)

    def mutate(self, i, population, pop_values):
        x = population[i]
        # dimension of the problem
        n = np.size(x)
        # choose 3 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 3, replace=False)

        # sort agents, so "a" is best and the order of other 2 does not matter

        agent_values = [pop_values[agents[0]], pop_values[agents[1]], pop_values[agents[2]]]
        agents = [population[agents[0]], population[agents[1]], population[agents[2]]]
        max_idx = np.argmax(agent_values)
        other_idx = np.delete(np.arange(3), max_idx)
        a = agents[max_idx]
        b = agents[other_idx[0]]
        c = agents[other_idx[1]]
        R = np.random.randint(low=0, high=self.N)
        y = [a[j] + self.F * (b[j] - c[j]) if np.random.rand() < self.CR or j == R else x[j] for j in range(n)]
        return y


class deBest2(Mutation):
    """
    DEBEST2
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)
        assert self.N > 5

    def mutate(self, i, pop, pop_f):
        x = pop[i]
        # dimension of the problem
        n = np.size(x)
        # choose 5 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 5, replace=False)

        agent_values = [pop_f[agents[0]], pop_f[agents[1]], pop_f[agents[2]], pop_f[agents[3]], pop_f[agents[4]]]
        agents = [pop[agents[0]], pop[agents[1]], pop[agents[2]], pop[agents[3]], pop[agents[4]]]
        idx = [i[0] for i in sorted(enumerate(agent_values), key=lambda e: e[1])]
        a = agents[idx[0]]
        b = agents[idx[1]]
        c = agents[idx[2]]
        d = agents[idx[3]]
        e = agents[idx[4]]
        R = np.random.randint(low=0, high=self.N)
        y = [a[j] + self.F * (b[j] - c[j]) + self.F * (d[j] - e[j]) if np.random.rand() < self.CR or j == R else x[j]
             for j in range(n)]
        return y


class deCurrentToBest1(Mutation):
    """
    current-to-best 1
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)

    def mutate(self, i, pop, pop_f):
        x = pop[i]
        # dimension of the problem
        n = np.size(x)
        # choose 3 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 3, replace=False)

        agent_values = [pop_f[agents[0]], pop_f[agents[1]], pop_f[agents[2]]]
        agents = [pop[agents[0]], pop[agents[1]], pop[agents[2]]]
        max_idx = np.argmax(agent_values)
        other_idx = np.delete(np.arange(3), max_idx)
        a = agents[max_idx]
        b = agents[other_idx[0]]
        c = agents[other_idx[1]]
        R = np.random.randint(low=0, high=self.N)
        y = [x[j] + self.F * (a[j] - x[j]) + self.F * (b[j] - c[j]) if np.random.rand() < self.CR or j == R else x[j]
             for j in range(n)]
        return y


class deRandToBest1(Mutation):
    """
    rand-to-best 1
    """

    def __init__(self, F, N, CR):
        Mutation.__init__(self, F, N, CR)
        assert self.N >= 5

    def mutate(self, i, pop, pop_f):
        x = pop[i]
        # dimension of the problem
        n = np.size(x)
        # choose 4 agents, but not the one of the x
        agents = np.random.choice(np.delete(np.arange(self.N), i), 4, replace=False)

        agent_values = [pop_f[agents[0]], pop_f[agents[1]], pop_f[agents[2]], pop_f[agents[3]]]
        agents = [pop[agents[0]], pop[agents[1]], pop[agents[2]], pop[agents[3]]]
        max_idx = np.argmax(agent_values)
        other_idx = np.delete(np.arange(4), max_idx)
        a = agents[max_idx]
        b = agents[other_idx[0]]
        c = agents[other_idx[1]]
        d = agents[other_idx[2]]
        R = np.random.randint(low=0, high=self.N)
        y = [
            b[j] + self.F * (a[j] - b[j]) + self.F * (c[j] - d[j]) if np.random.rand() < self.CR or j == R else x[j]
            for j in range(n)]
        return y


if __name__ == "__main__":
    pass
