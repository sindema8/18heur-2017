
# coding: utf-8

# # Mutation and mutation correction strategies

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


# #### ``TSPGrid(3, 3)`` as referential objection function

# In[3]:

tsp = TSPGrid(3, 3)


# ## Mutation correction strategies
# 
# They exists, since we need to "return" any mutated solution back to the domain.
# 
# In this case the domain boundaries are:

# In[4]:

print('a = {}'.format(tsp.a))
print('b = {}'.format(tsp.b))


# Let's assume mutated solution `x`:

# In[5]:

x = np.array([9, 2, 0, 1, 3, 2, 1, 0])


# ### 1. Correction by sticking to domain boundaries
# 
# Implemented by the `Correction` class (in `src/heur_mutations.py`).

# In[6]:

sticky = Correction(tsp)
sticky.correct(x)


# ### 2. Correction by periodic domain extension
# 
# Implemented by the `ExtensionCorrection` class (in `src/heur_mutations.py`).

# In[7]:

extend = ExtensionCorrection(tsp)
extend.correct(x)


# ### 3. Correction by mirroing
# 
# Implemented by the `MirroringCorrection` class (in `src/heur_mutations.py`).

# In[8]:

mirror = MirrorCorrection(tsp)
mirror.correct(x)


# ## Mutation strategies
# 
# Previously, we have used Fast Simulated Annealing directly with the Cauchy mutation, but it should be possible to swap it with any other mutation. And this is true also for other heuristics.

# ### 1. Discrete Cauchy mutation
# 
# Implemented by the `CauchyMutation` class (in `src/heur_mutations.py`).

# In[9]:

cauchy = CauchyMutation(r=0.5, correction=mirror)  # sample initialization
for i in range(10):
    print(cauchy.mutate([0, 2, 0, 1, 3, 2, 1, 0]))


# ### 2. Discrete Gaussian mutation
# 
# Implemented by the `GaussMutation` class (in `src/heur_mutations.py`).

# In[10]:

gauss = GaussMutation(sigma=0.5, correction=mirror)  # sample initialization
for i in range(10):
    print(gauss.mutate([0, 2, 0, 1, 3, 2, 1, 0]))


# # Assignment
# 
# * Evaluate performance of different mutations and mutation corrections using FSA on TSP (see previous lecture notebook for referential data).
