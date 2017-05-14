
# coding: utf-8

# # Generalization for $\mathbb{R}^n$
# 
# Our framework should automatically recognize objective funcion domain and use proper routines for each domain ($\mathbb{Z}^n$ or $\mathbb{R}^n$).

# ### Set up IPython notebook environment first...

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


# ## Testing `numpy.dtype`

# In[3]:

zn = np.ones(10, dtype=int)
zn


# In[4]:

zn.dtype


# In[5]:

rn = np.ones(10)
rn


# In[6]:

rn.dtype


# In[7]:

from utils import is_integer  # new function


# In[8]:

is_integer(rn)


# In[9]:

is_integer(zn)


# ## De Jong 1 objective function
# 
# Source: http://www.geatbx.com/docu/fcnindex-01.html#P89_3085

# In[10]:

from objfun import DeJong1


# In[11]:

dj = DeJong1(n=3, eps=0.1)


# In[12]:

dj.a


# In[13]:

dj.b


# In[14]:

x = dj.generate_point()
x


# In[15]:

dj.evaluate(x)


# In[16]:

# optimum
dj.evaluate(np.zeros(5))


# ## Generalized mutation demo on De Jong 1
# 
# Let's test mutation corrections first:

# In[17]:

from heur_mutations import Correction, MirrorCorrection, ExtensionCorrection


# In[18]:

# sticky correction in R^n (mind x[1])
Correction(dj).correct(np.array([6.12, -4.38,  2.96]))


# In[19]:

# mirror correction in R^n (mind x[1])
MirrorCorrection(dj).correct(np.array([6.12, -4.38,  2.96]))


# In[20]:

# extension correction in R^n (mind x[1])
ExtensionCorrection(dj).correct(np.array([6.12, -4.38,  2.96]))


# Corrections work also in the continuous case, as expected.
# 
# Mutations were modified as follows:
# ```
# if is_integer(x):
#     x_new = np.array(np.round(x_new), dtype=int)
# ```
# i.e. rounding is a conditional step.

# In[21]:

from heur_mutations import CauchyMutation, GaussMutation


# In[22]:

cauchy = CauchyMutation(r=.1, correction=MirrorCorrection(dj))
cauchy.mutate(np.array([6.12, -4.38,  2.96]))


# In[23]:

gauss = GaussMutation(sigma=.1, correction=MirrorCorrection(dj))
gauss.mutate(np.array([6.12, -4.38,  2.96]))


# BTW, integer tasks, like TSP, will continue to work as ususal:

# In[24]:

from objfun import TSPGrid


# In[25]:

tsp = TSPGrid(3, 3)


# In[26]:

cauchy_tsp = CauchyMutation(r=1, correction=MirrorCorrection(tsp))
cauchy_tsp.mutate(np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int))


# ## De Jong 1 optimization via FSA
# 
# Thanks to current state of the framework, no modification to FSA is needed.

# In[27]:

from heur import FastSimulatedAnnealing


# In[28]:

heur = FastSimulatedAnnealing(dj, maxeval=10000, T0=10, n0=10, alpha=2, 
                              mutation=cauchy)
res = heur.search()
print(res['best_x'])
print(res['best_y'])
print(res['neval'])


# ## De Jong 1 optimization via GO
# 
# Let's review modified crossover operators in $\mathbb{R}^n$ first:

# In[29]:

from heur_crossovers import Crossover, UniformMultipoint, RandomCombination


# In[30]:

x = dj.generate_point()
y = dj.generate_point()
print(x)
print(y)


# In[31]:

Crossover().crossover(x, y)


# In[32]:

UniformMultipoint(1).crossover(x, y)


# In[33]:

RandomCombination().crossover(x, y)


# They work as expected.
# 
# Let's make sure they will be compatible with integer tasks:

# In[34]:

x = tsp.generate_point()
y = tsp.generate_point()
print(x)
print(y)


# In[35]:

Crossover().crossover(x, y)


# In[36]:

UniformMultipoint(1).crossover(x, y)


# In[37]:

RandomCombination().crossover(x, y)


# Finally, let's run GO:

# In[38]:

from heur import GeneticOptimization


# In[39]:

heur = GeneticOptimization(dj, maxeval=10000, N=10, M=30, Tsel1=0.5, Tsel2=0.1, 
                           mutation=cauchy, crossover=UniformMultipoint(1))
res = heur.search()
print(res['best_x'])
print(res['best_y'])
print(res['neval'])


# ## Excercises
# * Tune heuristics on other continuous [benchmark functions](http://www.geatbx.com/docu/fcnindex-01.html)
