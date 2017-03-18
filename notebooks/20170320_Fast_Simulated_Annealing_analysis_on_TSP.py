
# coding: utf-8

# # Travelling Salesman Problem (TSP) via Fast Simulated Annealing (FSA)
# 
# ## Quick refresher
# 
# * $T_0 \in \mathbb{R}$ initial temperature
# * $ n_0 \in \mathbb{N} $ and $ \alpha \in \mathbb{R} $  - cooling strategy parameters
# * $k$-th step:
#   * Evaluate current temperature:
#     * $ T = \frac{T_0}{1+(k/n_0)^\alpha} $ for $ \alpha > 0 $,
#     * $ T = T_0 \cdot \exp(-(k/n_0)^{-\alpha}) $ otherwise.
#   * **Mutate** the solution ``x`` -> ``y``
#   * $s = \frac{f_x-f_y}{T}$
#   * replace ``x`` if $u < 1/2 + \arctan(s)/\pi$ where $u$ is random (uniform) number
#   
# Cauchy **mutation operator**:
# 
# * mutation perimeter (width) controlled by parameter $r \in \mathbb{R}$
# * $ \boldsymbol{\mathsf{x}}_\mathrm{new} = \boldsymbol{\mathsf{x}} + r \cdot \tan{\left(\pi \left(\boldsymbol{\mathsf{r}} - \dfrac{1}{2}\right)\right)} $ where $\boldsymbol{\mathsf{r}}$ is random uniform vector
# 

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

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Import our code
from heur import ShootAndGo, FastSimulatedAnnealing
from objfun import TSPGrid


# ## Initialize ``TSPGrid(3, 3)``

# In[3]:

# initialization
tsp = TSPGrid(3, 3)


# ## Referential performance: Random Shooting ($\mathrm{SG}_{0}$) and Steepest Descent ($\mathrm{SG}_{\infty}$)

# In[5]:

NUM_RUNS = 1000
maxeval = 1000


# In[6]:

def experiment_sg(of, maxeval, num_runs, hmax):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing hmax={}'.format(hmax)):
        result = ShootAndGo(of, maxeval=maxeval, hmax=hmax).search() # dict with results of one run
        result['run'] = i
        result['heur'] = 'SG_{}'.format(hmax) # name of the heuristic
        result['hmax'] = hmax
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'hmax', 'best_x', 'best_y', 'neval'])


# In[7]:

table_ref = pd.DataFrame()

for hmax in [0, np.inf]:
    res = experiment_sg(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, hmax=hmax)
    table_ref = pd.concat([table_ref, res], axis=0)


# In[8]:

table_ref.head()


# In[9]:

# from: 20170306_Steepest_descent_vs_Random_descent.ipynb#Overall-statistics

def rel(x):
    return len([n for n in x if n < np.inf])/len(x)

def mne(x):
    return np.mean([n for n in x if n < np.inf])

def feo(x):
    return mne(x)/rel(x)


# In[10]:

stats_ref = table_ref.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_ref = stats_ref.reset_index()


# In[11]:

stats_ref


# ## TSP optimization using TSP
# 
# Let's evaluate performance of different temperatures, first.

# In[12]:

def experiment_fsa(of, maxeval, num_runs, T0, n0, alpha, r):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing T0={}, n0={}, alpha={}, r={}'.format(T0, n0, alpha, r)):
        result = FastSimulatedAnnealing(of, maxeval=maxeval, T0=T0, n0=n0, alpha=alpha, r=0.5).search()
        result['run'] = i
        result['heur'] = 'FSA_{}_{}_{}_{}'.format(T0, n0, alpha, r) # name of the heuristic
        result['T0'] = T0
        result['n0'] = n0
        result['alpha'] = alpha
        result['r'] = r
        results.append(result)
    
    return pd.DataFrame(results, columns=['heur', 'run', 'T0', 'n0', 'alpha', 'r', 'best_x', 'best_y', 'neval'])


# In[16]:

table_fsa = pd.DataFrame()

for T0 in [1e-10, 1e-2, 1, np.inf]:
    res = experiment_fsa(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, T0=T0, n0=1, alpha=2, r=0.5)
    table_fsa = pd.concat([table_fsa, res], axis=0)


# In[18]:

stats_fsa = table_fsa.pivot_table(
    index=['heur', 'T0'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['T0'])


# # Analysis
# 
# **Can we improve the best configuration ($T_0=1$)?**
# 
# Let's carefully analyze the data...

# In[42]:

heur = FastSimulatedAnnealing(tsp, maxeval=1000, T0=1, n0=1, alpha=2, r=0.5)
result = heur.search()


# In[43]:

print('neval = {}'.format(result['neval']))
print('best_x = {}'.format(result['best_x']))
print('best_y = {}'.format(result['best_y']))


# In[44]:

log_data = result['log_data'].copy()
log_data = log_data[['step', 'x', 'f_x', 'y', 'f_y', 'T', 'swap']]  # column re-ordering, for better readability
log_data.head(15)


# In[45]:

def plot_compare(step_data, ax1_col, ax1_label, ax2_col, ax2_label):
    fig, ax1 = plt.subplots()

    k = step_data.index.values
    T = step_data[ax1_col]
    ax1.plot(k, T, 'b-')
    ax1.set_xlabel('Step')
    ax1.set_ylabel(ax1_label, color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    s2 = step_data[ax2_col]
    ax2.plot(k, s2, 'r.')
    ax2.set_ylabel(ax2_label, color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.show()


# In[46]:

plot_compare(log_data, 'T', 'Temperature', 'f_x', 'f(x)')


# In[47]:

plot_compare(log_data, 'T', 'Temperature', 'f_y', 'f(y)')


# ### Slower cooling?
# 
# Let's double $n_0$:

# In[51]:

heur = FastSimulatedAnnealing(tsp, maxeval=1000, T0=1, n0=2, alpha=2, r=0.5)
result = heur.search()
print('neval = {}'.format(result['neval']))
print('best_x = {}'.format(result['best_x']))
print('best_y = {}'.format(result['best_y']))


# In[52]:

log_data = result['log_data'].copy()
log_data = log_data[['step', 'x', 'f_x', 'y', 'f_y', 'T', 'swap']]  # column re-ordering, for better readability


# In[53]:

plot_compare(log_data, 'T', 'Temperature', 'f_x', 'f(x)')


# In[54]:

plot_compare(log_data, 'T', 'Temperature', 'f_y', 'f(y)')


# **Thorough testing**:

# In[59]:

for n0 in [2, 3, 5, 10]:
    res = experiment_fsa(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=n0, alpha=2, r=0.5)
    table_fsa = pd.concat([table_fsa, res], axis=0)


# In[60]:

stats_fsa = table_fsa.pivot_table(
    index=['heur', 'T0', 'n0'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['T0', 'n0'])


# ### Is the mutation $r$ adequate?

# In[61]:

log_data['jump_length'] = log_data.apply(lambda r: np.linalg.norm(r['x'] - r['y']), axis=1)
log_data.head(10)


# In[62]:

log_data['jump_length'].describe()


# In[63]:

for r in [.1, .25, .75, 1, 2]:
    res = experiment_fsa(of=tsp, maxeval=maxeval, num_runs=NUM_RUNS, T0=1, n0=5, alpha=2, r=r)
    table_fsa = pd.concat([table_fsa, res], axis=0)


# In[66]:

stats_fsa = table_fsa.pivot_table(
    index=['heur', 'T0', 'n0', 'r'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
stats_fsa = stats_fsa.reset_index()
stats_fsa.sort_values(by=['T0', 'n0', 'r'])


# The best performing instance of FSA, according to $FEO$:

# In[69]:

stats_fsa.sort_values(by=['feo']).head(1)


# ## Conclusion
# 
# When assessing heuristic performance, always try to combine _prior_ knowledge with _posterior_ data collected during experimental phase. On the other hand, exhaustive grid parameter space search can be useful as well, but you will learn much more using this iterative approach.

# ## Assignment
# 
# 1. Could you further improve performance of FSA on this instance of TSP?
# 2. We have performed analyses of the `log_data` contents on a single run of a heuristic. Try to aggregate and make use o fthese statistics on multiple eperimental runs.
# 3. When FSA search is successful, the last objective function call is missing in the `log_data` output variable. Could you improve this behaviour?
