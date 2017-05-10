
# coding: utf-8

# # _h_-means
# 
# Heuristic clustering inspired by [_k_-means](https://cs.wikipedia.org/wiki/K-means). As another demonstration of how continuous heuristics can be used.

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

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # Objective function demonstration

# In[3]:

from objfun import HMeans


# In[4]:

of = HMeans()  


# In[5]:

print('f* = {}'.format(of.fstar))


# In[6]:

# plot the data proints
X = of.X
ax = plt.scatter(x=X[:, 0], y=X[:, 1])


# In[7]:

print('a = {}'.format(of.a))
print('b = {}'.format(of.b))


# **Bounds are repeated for each centroid, that will be tuned by the heuristic.**

# In[10]:

# some random evaluations
for i in range(10):
    x = of.generate_point()
    print('f({}) = {}'.format(x, of.evaluate(x)))


# In[11]:

# we can get cluster labels (for a random solution)
labels = of.get_cluster_labels(x)
print(labels)


# In[12]:

# auxiliary routine
def visualize_solution(x, of):
    labels = of.get_cluster_labels(x)
    X = of.X
    ax = plt.scatter(x=X[:, 0], y=X[:, 1], c=labels)


# In[13]:

# visualization of a random solution
visualize_solution(x, of)


# # Optimization demonstration

# In[14]:

from heur_mutations import MirrorCorrection, CauchyMutation
from heur import FastSimulatedAnnealing


# In[15]:

heur = FastSimulatedAnnealing(of, maxeval=10000, T0=10, n0=10, alpha=2, 
                              mutation=CauchyMutation(r=0.1, correction=MirrorCorrection(of)))
res = heur.search()
print('x_best = {}'.format(res['best_x']))
print('y_best = {}'.format(res['best_y']))
print('neval = {}'.format(res['neval']))


# In[16]:

visualize_solution(res['best_x'], of)


# ## Excercises
# * Tune heuristics for this objective function
# * Tune this objective function, e.g. by penalization for smaller number of clusters than $h$ (and make sure you understand why this is possible)
