
# coding: utf-8

# # Lecture outline
# 
# 1. Experimental framework implementation
# 1. Performance evaluation
# 1. Assignments

# # 1. Experimental framework implementation
# 
# ## Best practice to implement $n$ heuristics and $m$ objective functions?
# 
# * There are some common characteristics for all objects in our framework
#   * Heuristics - store best found solution, manage stop criterion, etc.
#   * Objective functions - store $f^*$, lower/upper bounds, etc.
# * Every specific heuristic or obj. function implements its own search space exploration or evaluation, neighbourhood generation, etc.
# * Thus, object-oriented design should help us to separate this concerns as much as possible and also to keep us sane
# 
# 
# <img src="img/oop_design.png">
# 

# ## Example: generalized Shoot&Go and two objective functions (AirShip and `sum(x)`)
# 
# 
# ### Generalized Shoot&Go: $\mathrm{SG}_{hmax}$
# 
# * Shoot & Go heuristic (also known as *Iterated Local Search*, *Random-restart hill climbing*, etc)
#     * $hmax \in \{ 0, 1, \ldots, \infty \}$ parameter - maximum number of local searches / hill climbs
#     * note that $\mathrm{SG}_{0}$ is pure Random Shooting (Random Search)
#     
# * implemented as ``class ShootAndGo(Heuristic)`` in ``src/heur.py``    
#     
# ### Objective functions
# 
# #### AirShip
# 
# * Same as on previous lecture, but wee need to **minimize** obj. function values
# * implemented as ``class AirShip(ObjFun)`` in ``src/objfun.py``
# 
# #### `sum(x)`
# 
# * Just as demonstration of vectorized lower/upper bounds
# * implemented as ``class Sum(ObjFun)`` in ``src/objfun.py``
# 
# **Review the code, please!**

# # 2. Performance evaluation
# 
# ## What is the recommended approach to store and analyze results of your experiments?
# 
# 1. Append all relevant statistics from a single run into table (e.g. CSV file in memory or on disk), including all task and heuristic parameters 
# 2. Load the table into analytical tool of your choice (data frames, Excel or Google Docs spreadsheets, etc.)
# 3. Pivot by relevant parameters, visualize in tables or charts

# ## Demonstration
# 
# Neccessary notebook setup first:

# In[1]:

# Import path to source directory (bit of a hack in Jupyter)
import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd,'../src'))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

# Import extrenal librarires
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

# Import our code
from heur import ShootAndGo


# ### General experiment setup
# 
# Runs selected objective function (`of`) using selected heuristic multiple times, stores and returns data (results) in a data frame.

# In[3]:

def experiment(of, num_runs, hmax):
    results = []
    for i in tqdm_notebook(range(num_runs), 'Testing hmax = {}'.format(hmax)):
        result = ShootAndGo(of, maxeval=100, hmax=hmax).search() # dict with results of one run
        result['run'] = i
        result['heur'] = 'SG_{}'.format(hmax) # name of the heuristic
        result['hmax'] = hmax
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run', 'hmax', 'best_x', 'best_y', 'neval'])


# ### Air Ship experiments

# In[4]:

from objfun import AirShip
of = AirShip()


# In[5]:

table = pd.DataFrame()
for hmax in [0, 1, 2, 5, 10, 20, 50, np.inf]:
    res = experiment(of, 10000, hmax)
    table = pd.concat([table, res], axis=0)


# In[6]:

table.info()


# In[7]:

table.head()


# In[8]:

# import visualization libraries
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# #### Quality of solutions based on `hmax`?
# 
# In **tabular** form:

# In[9]:

table.groupby(['hmax'])['best_y'].median()


# In[10]:

table.groupby(['hmax'])['best_y'].mean()


# Feel free to compute other statistics instead of median and mean.
# 
# Directly as **Box-Whiskers plot**:

# In[11]:

ax = sns.boxplot(x="hmax", y="best_y", data=table)


# #### Number of evaluations (when successful), based on `hmax`?
# 
# Let's add another columns, `success`:

# In[12]:

table['success'] = table['neval'] < np.inf


# In[13]:

table[table['success'] == True].head()


# In[14]:

table[table['success'] == False].head()


# Table:

# In[15]:

table[table['success'] == True].groupby(['hmax'])['neval'].mean()


# In[16]:

table[table['success'] == True].groupby(['hmax'])['neval'].median()


# Chart:

# In[17]:

ax = sns.boxplot(x="hmax", y="neval", data=table[table['success'] == True])


# #### Reliability

# In[18]:

rel_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: len([n for n in x if n < np.inf])/len(x)
)


# In[19]:

rel_by_hmax


# In[20]:

ax = rel_by_hmax.plot(kind='bar')


# #### Speed, normalized by reliability?
# 
# * Reliability: $REL = m/q$ where $m$ is number of successful runs and $q$ is total number of runs, $REL \in [0, 1]$
# * Mean Number of objective function Evaluations: $MNE = \frac{1}{m} \sum_{i=1}^m neval_i$
# * Feoktistov criterion: $FEO = MNE/REL$

# In[21]:

feo_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: np.mean([n for n in x if n < np.inf])/(len([n for n in x if n < np.inf])/len(x))
    #                 ^^^   mean number of evaluations ^^^ / ^^^             reliability         ^^^^
)


# In[22]:

feo_by_hmax


# In[28]:

ax = feo_by_hmax.plot(kind='bar')


# ### `sum(x)` experiments
# 
# Let's review this function a little bit:

# In[29]:

from objfun import Sum
of = Sum([0, 0, 0, 0], [10, 10, 10, 10])


# In[30]:

x = of.generate_point()
print(x)
print(of.evaluate(x))


# In[31]:

print(of.get_neighborhood(x, 1))


# In[32]:

print(of.get_neighborhood(x, 2))


# In[33]:

of.get_neighborhood([0, 0, 0, 0], 1)


# In[34]:

of.get_neighborhood([10, 10, 10, 10], 1)


# And now, perform traditional experiments:

# In[35]:

table = pd.DataFrame()
for hmax in [0, 1, 2, 5, 10, 20, 50, np.inf]:
    res = experiment(of, 10000, hmax)
    table = pd.concat([table, res], axis=0)


# #### Quality of solutions based on hmax?

# In[36]:

ax = sns.boxplot(x="hmax", y="best_y", data=table)


# #### Number of evaluations (when successful), based on hmax?

# In[37]:

table['success'] = table['neval'] < np.inf


# In[38]:

ax = sns.boxplot(x="hmax", y="neval", data=table[table['success'] == True])


# #### Reliability?

# In[42]:

rel_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: len([n for n in x if n < np.inf])/len(x)
)


# In[44]:

rel_by_hmax


# In[43]:

ax = rel_by_hmax.plot(kind='bar')


# #### Feoktistov criterion?

# In[40]:

feo_by_hmax = table.pivot_table(
    index=['hmax'],
    values=['neval'],
    aggfunc=lambda x: np.mean([n for n in x if n < np.inf])/(len([n for n in x if n < np.inf])/len(x))
)


# In[41]:

ax = feo_by_hmax.plot(kind='bar')


# # Assignments
# 
# 1. Implement examples in this notebook
# 1. Experiment with **neighbourhood diameter** `d` in `AirShip.get_neighborhood(x, d)`
# 1. Add **Random Descent** heuristic (similar to Shoot & Go, but does not follow steepest descent, chooses direction of the descent randomly instead) into existing framework and analyze its performance.
# 1. Add **Taboo Search** heuristic into existing framework and analyze its performance.
