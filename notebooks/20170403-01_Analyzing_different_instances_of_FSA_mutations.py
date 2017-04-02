
# coding: utf-8

# # Travelling Salesman Problem (TSP) via Fast Simulated Annealing (FSA)
# ## Using different instances of mutations

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

# Import extrenal librarires
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

# Import our code
from heur import FastSimulatedAnnealing
from heur_mutations import Correction, MirrorCorrection, ExtensionCorrection, CauchyMutation, GaussMutation
from objfun import TSPGrid


# ## Initialize ``TSPGrid(3, 3)``

# In[3]:

tsp = TSPGrid(3, 3)


# ## TSP optimization using FSA and different instances of mutations

# In[4]:

# initialize different corrections
sticky = Correction(tsp)
extend = ExtensionCorrection(tsp)
mirror = MirrorCorrection(tsp)

# prepare battery of mutations to be tested (with some metadata)
mutations = [
    {'mutation': CauchyMutation(r=1.0, correction=sticky), 'name': 'Cauchy-1-Sticky', 'param': 1.0, 'correction': 'sticky'},
    {'mutation': CauchyMutation(r=1.0, correction=extend), 'name': 'Cauchy-1-Extend', 'param': 1.0, 'correction': 'extend'},
    {'mutation': CauchyMutation(r=1.0, correction=mirror), 'name': 'Cauchy-1-Mirror', 'param': 1.0, 'correction': 'mirror'},
    # Test Gaussian mutations as well!
]


# In[5]:

# traditional testing procedure setup
def experiment_fsa(of, maxeval, num_runs, T0, n0, alpha, mutation):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing mutation {}'.format(mutation['name'])):
        result = FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=alpha, mutation=mutation['mutation']).search()
        result['run'] = i
        result['heur'] = 'FSA_{}_{}_{}_{}'.format(T0, n0, alpha, mutation['name'])
        result['mut_param'] = mutation['param']
        result['mut_corr'] = mutation['correction']
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'muta_param', 'mut_corr', 'best_x', 'best_y', 'neval'])


# In[6]:

NUM_RUNS = 1000
maxeval = 1000


# In[7]:

results = pd.DataFrame()
for mutation in mutations:
    res = experiment_fsa(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1.0, n0=5, alpha=2, mutation=mutation)
    results = pd.concat([results, res], axis=0)


# In[9]:

# from: 20170306_Steepest_descent_vs_Random_descent.ipynb#Overall-statistics
def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[10]:

results_pivot = results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
results_pivot = results_pivot.reset_index()
results_pivot


# ## Assignment
# 
# * Analyze roots of this situation
# * Some hints:
#   * Analyze effects of different settings of `r`
#   * Make use of detailed heuristic logs
#   * Compare results with Gaussian mutation instances
