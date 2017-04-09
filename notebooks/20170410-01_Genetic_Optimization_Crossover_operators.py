
# coding: utf-8

# # Genetic Optimization - Crossover Operator Generaliztion

# In[1]:

# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd, '../src'))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

# Import external libraries
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

# Import our code
from heur import GeneticOptimization
from heur_mutations import Correction, CauchyMutation
from heur_crossovers import Crossover, UniformMultipoint, RandomCombination  # <- new classes!
from objfun import TSPGrid


# **Let's use the well-known ``TSPGrid(3, 3)`` for demonstration purposes**

# In[3]:

tsp = TSPGrid(3, 3)


# # Three different operators:

# **First, let's assume these are our parents:**

# In[4]:

x = np.zeros(10, dtype=int)
x


# In[5]:

y = 9*np.ones(10, dtype=int)
y


# ## 1. Random mix (baseline class)

# In[6]:

co_rnd = Crossover()


# In[7]:

co_rnd.crossover(x, y)


# In[8]:

co_rnd.crossover(x, y)


# In[9]:

co_rnd.crossover(x, y)


# ## 2. Uniform n-point crossover

# In[10]:

co_uni = UniformMultipoint(1)


# In[11]:

co_uni.crossover(x, y)


# ## 3. Random combination

# In[12]:

co_comb = RandomCombination()


# In[13]:

co_comb.crossover(x, y)


# In[14]:

co_comb.crossover(x, y)


# In[15]:

co_comb.crossover(x, y)


# # Demonstration

# In[16]:

NUM_RUNS = 1000
maxeval = 1000


# In[17]:

# prepare battery of crossovers to be tested (with some metadata)
crossovers = [
    {'crossover': Crossover(), 'name': 'mix'},
    {'crossover': UniformMultipoint(1), 'name': 'uni'},  #  test for other n as well!
    {'crossover': RandomCombination(), 'name': 'rnd'},
]


# In[18]:

# traditional testing procedure setup
def experiment_go(of, maxeval, num_runs, N, M, Tsel1, Tsel2, mutation, crossover):
    results = []
    heur_name = 'GO_{}'.format(crossover['name'])
    for i in tqdm_notebook(range(num_runs), 'Testing {}'.format(heur_name)):
        result = GeneticOptimization(of, maxeval, N=N, M=M, Tsel1=Tsel1, Tsel2=Tsel2, mutation=mutation, 
                                     crossover=crossover['crossover']).search()
        result['run'] = i
        result['heur'] = heur_name
        result['crossover'] = crossover['name']
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'crossover', 'best_x', 'best_y', 'neval'])


# In[19]:

results = pd.DataFrame()
for crossover in crossovers:
    res = experiment_go(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, N=5, M=15, Tsel1=1.0, Tsel2=0.2, 
                        mutation=CauchyMutation(r=1.0, correction=Correction(tsp)), crossover=crossover)
    results = pd.concat([results, res], axis=0)


# In[20]:

# from: 20170306_Steepest_descent_vs_Random_descent.ipynb#Overall-statistics
def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[21]:

results_pivot = results.pivot_table(
    index=['heur', 'crossover'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot.sort_values(by='crossover')


# ## Assignment
# 
# * Thoroughly test different kinds of GO setup
# * Could you of any other crossover operator? See e.g. https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm) for inspiration
