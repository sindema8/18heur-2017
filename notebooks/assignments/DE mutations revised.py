
# coding: utf-8

# # Mutations for differential evolution revised

# In this paper, core operation for Differential Evolution is researched, mutation. For DE, different types of mutation can easily change a chance to susccess for any function, as it strongly defines behavior of heuristics. With non-random changes to algorithm via mutation, it can be adapted to single-modal functions with stress to exploitation of current best solution or to multi-modal functions, encouraging exploration over exploitation through more random-wise mutations.
# 
# Several widely-used mutations are presented and its advantages and disadvantages are described with example of finding solution to one of De Jong testing functions - single-modal Rosenbrock 2D "banana" function. It is expected, that mutations using best candidates will be more successful than the one with random search, as the heuristic can not stuck in local minimum, for the Rosenbrock function has only one, which happens to be global minimum as well.

# In[9]:

import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd, '../../src'))

# Ensure modules are reloaded on any change (very useful when developing code on the fly)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[10]:

# Import external libraries
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


# ### Evaluation
# 
# Heuristics results will be evaluated with standard criteria:
# 
# * Reliability (REL) - statistical probability of successful run
# * Mean Number of Objective Function Evaluations (MNE)
# * Feokristov criterion (FEO) - MNE/REL

# In[11]:

# Define criteria for heuristics evaluation
def rel(x):
    return len([n for n in x if n < np.inf])/len(x)
def mne(x):
    return np.mean([n for n in x if n < np.inf])
def feo(x):
    return mne(x)/rel(x)


# In[12]:

# Import objective function and heuristics dependencies
from objfun import Rosenbrock, Schwefel
from heur import DifferentialEvolution
from de_mutations import deRand, deBest, deRand2, deBest2, deCurrentToBest1, deRandToBest1


# In[13]:

NUM_RUNS = 100
maxeval = 1000


# ### Rosenbrock function
# 
# For testing purposes - different tuning of parameters for given mutations, Rosenbrock function will be used. As the parameters tuning is question of each function it self, there is no certainity, that the same set of parameteres for given mutation will be optimal for different function (i.e. No Free Lunch theorem).
# 
# Definition of function described here (http://www.geatbx.com/docu/fcnindex-01.html#P129_5426) will be used.

# In[14]:

rosenbrock2 = Rosenbrock(n=2, eps=0.1)
fstar = rosenbrock2.evaluate(np.ones(2))


# ### DE parameters

# For Differential evolution three parameteres need to be supplied, all of the are very important setting for all the mutations revised in this notebook.
# 
# * N: population size, to properly test all the mutations, has to be $> 5$ (some mutations need bigger amount of individuals),
# * CR: crossover probability,
# * F: differential weight, has to be in $[0,2]$.

# ### Mutations in short
# 
# Implemented mutations are the ones originally presented by publicators of DE Price and Storn. This subset of known mutations contains both explorative (Rand1,Rand2) as exhaustive (Best1,Best2) approaches and in addition some others, trying to combine both. There is always trade-off between exploration and exhaustion, which has to be handled accordingly and mutations has to be tailored to the needs of objective function (i.e. if the function is multimodal, different mutation has to be used - in favor of exploration - than in case of unimodal.) Following brief list contains all the implemented mutations and their definition.
# 
# Numbers $r_1, \ldots, r_5$ are random numbers from range $1, \ldots, N$. In all cases, $x_{current} \ne x_{r_1} \ne \ldots \ne x_{r_5} \ne x_{best}$
# 
# * **Rand1** - simplest of random mutations, $x_{new} = x_{r_1} + F (x_{r_2} - x_{r_3})$.
# 
# * **Rand2** - similar to deRand, uses more individuals, hence the need for higher value of $N$, $x_{new} = x_{r_1} + F (x_{r_2} - x_{r_3}) + F (x_{r_4} - x_{r_5})$.
# 
# * **Best1** - computationally harder, especially for bigger populations, because of the search for the best individual in population, denoted $x_{best}$, $x_{new} = x_{best} + F (x_{r_1} - x_{r_2})$.
# 
# * **Best2** - similar to deBest, uses more individuals, $x_{new} = x_{best} + F (x_{r_1} - x_{r_2})+ F (x_{r_3} - x_{r_4})$.
# 
# * **Current to Best** - $x_{new} = x_{current} + F (x_{best} - x_{current})+ F (x_{r_1} - x_{r_2})$.
# 
# * **Rand to Best** - $x_{new} = x_{r_1} + F (x_{best} - x_{r_1})+ F (x_{r_2} - x_{r_3})$.

# ### Experiment setup

# In[15]:

def experiment_de(of, maxeval, num_runs, N, CR, F, mutation):
    results = []
    for i in tqdm_notebook(range(num_runs)):
        result = DifferentialEvolution(of, maxeval=maxeval, N=N, CR=CR, F=F, mutation=mutation['algorithm']).search()
        result['run'] = i
        result['heur'] = 'DE_{}_{}_{}_{}'.format(N, CR, F, mutation['name'])
        results.append(result)
    return pd.DataFrame(results, columns=['heur', 'run','best_x', 'best_y', 'neval'])


# ### Initial set of parameters
# Peek at the results with some initial set of parameters.
# 
# * Increase in population size gives better chance to find good solution, but is only useful for mutations with search for the best individual, which are deBest, deBest2, deCurrentToBest1 and deRandToBest1.
# 
# * Increase in F puts more emphasis on the mutation itself and increase in CR raises a chance of mutation.

# In[16]:

def initMutations(F, N, CR):
    mutations = [
    {'algorithm': deRand(F=F, N=N, CR=CR), 'name': 'deRand1'},
    {'algorithm': deBest(F=F, N=N, CR=CR), 'name': 'deBest1'},
    {'algorithm': deRand2(F=F, N=N, CR=CR), 'name': 'deRand2'},
    {'algorithm': deBest2(F=F, N=N, CR=CR), 'name': 'deBest2'},
    {'algorithm': deCurrentToBest1(F=F, N=N, CR=CR), 'name': 'deCurrentToBest1'},
    {'algorithm': deRandToBest1(F=F, N=N, CR=CR), 'name': 'deRandToBest1'},
    ]
    return mutations


# In[17]:

F=1
N=6
CR=0.5


# In[18]:

mutations = initMutations(F, N, CR)


# In[19]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[20]:


de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# From the first test, it can be seen, that rand1 mutation was best. It can be said, the mutations with selection of best individual were worse (best1 and best2). The reason could be small population, with the increase in population, these mutations could yield better results.

# In[70]:

F=1
N=12
CR=0.5


# In[22]:

mutations = initMutations(F, N, CR)


# In[23]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[24]:


de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# The reliability of all the experiments rose with the growth in population, most significantly for best1 and best2. According to Feocristov criterion, best1 was the best for this run, due to small number of evaluation in opposition to rand1, which has high reliability, but also needs more evaluations.

# Lets see, how increase of desired precision of outcome affect these results.

# In[25]:

rosenbrock2 = Rosenbrock(n=2, eps=0.001)


# In[26]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[27]:


de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# Most of the experiments did not finish, because maximal number of evaluations was exhausted. It is logical to increase the maximum if more precise solution is to be found.

# In[28]:

NUM_RUNS = 100
maxeval = 10000


# In[29]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[30]:

de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# Best2 and rand2 are the most reliable, but also takes twice as much evaluations and are therefore slower. To Feocristov, rand1 is far best. Lets lower the precision and tweak N, F and CR parameters for the mutations.

# In[31]:

maxeval = 1000
rosenbrock2 = Rosenbrock(n=2, eps=0.1)


# ### Random-wise mutations

# In[32]:

def initRandMutations(F, N, CR):
    mutations = [
    {'algorithm': deRand(F=F, N=N, CR=CR), 'name': 'deRand1'},
    {'algorithm': deRand2(F=F, N=N, CR=CR), 'name': 'deRand2'},
    ]
    return mutations


# In[33]:

F=0.3
N=20
CR=0.9
mutations = initRandMutations(F, N, CR)


# In[34]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[35]:

de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# After few experiments, this setup seems to be good. With increase of $F$, reliability growths, but number of evaluations growths as well, keeping Feocristov criterion at approximately the same level. It can be seen, that rand2 method is more reliable than rand1, but it takes more evaluations, so there is tradeoff between reliability and time consumption.

# ### Best-finding mutations

# In[36]:

def initBestMutations(F, N, CR):
    mutations = [
    {'algorithm': deBest(F=F, N=N, CR=CR), 'name': 'deBest1'},
    {'algorithm': deBest2(F=F, N=N, CR=CR), 'name': 'deBest2'},
    {'algorithm': deCurrentToBest1(F=F, N=N, CR=CR), 'name': 'deCurrentToBest1'},
    {'algorithm': deRandToBest1(F=F, N=N, CR=CR), 'name': 'deRandToBest1'},
    ]
    return mutations


# It is my expectation, that these mutations will profit from broad population and lower value of $CR$, as it is desired to keep the best solution "on its way".
# 
# Lets try the same setup as for the random-wise mutations:

# In[37]:

F=0.3
N=20
CR=0.9
mutations = initBestMutations(F, N, CR)


# In[38]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[39]:

de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# Best2 is far superior so far, and after other parameters setup, no better result could be achieved with best1 and best2. With growth of population size, reliability grows, but number of evaluations grows more rapidly thus lowering feocristov criterion. Let us now examine currentToBest1 and randToBest1 mutations

# In[40]:

def initLastMutations(F, N, CR):
    mutations = [
    {'algorithm': deCurrentToBest1(F=F, N=N, CR=CR), 'name': 'deCurrentToBest1'},
    {'algorithm': deRandToBest1(F=F, N=N, CR=CR), 'name': 'deRandToBest1'},
    ]
    return mutations


# In[41]:

F=1.0
N=20
CR=0.97
mutations = initLastMutations(F, N, CR)


# In[42]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[43]:

de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# These experiments are strongly reliable for $CR$ value close to $1$ and $F$ value deviationg around $1$, which is usable property, but number of evaluations is much higher than with best2 or rand1/2. Lowering F value leads to satisfyingly small number of evaluations but with the cost of low reliability and lowering CR leads to great reliability but higher number of evaluations (see below), so this setup is balanced.

# In[44]:

F=1
N=20
CR=0.5
mutations = initLastMutations(F, N, CR)


# In[45]:

de_results = pd.DataFrame()
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results = pd.concat([de_results, de_res], axis=0)


# In[46]:

de_results_pivot = de_results.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot = de_results_pivot.reset_index()

de_results_pivot


# ### Best setups
# This table shows results for the best setups of parameters for the mutations revised.

# In[47]:

de_results_rand = pd.DataFrame()
F=0.3
N=20
CR=0.9
mutations = initRandMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_rand = pd.concat([de_results_rand, de_res], axis=0)


# In[48]:

de_results_best= pd.DataFrame()
F=0.3
N=20
CR=0.9
mutations = initBestMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_best = pd.concat([de_results_best, de_res], axis=0)


# In[49]:

de_results_last= pd.DataFrame()
F=1.0
N=20
CR=0.97
mutations = initLastMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=rosenbrock2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_last = pd.concat([de_results_last, de_res], axis=0)


# In[51]:

de_results_final = pd.concat([de_results_rand,de_results_best,de_results_last], axis=0)
de_results_pivot_final = de_results_final.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot_final = de_results_pivot_final.reset_index()

de_results_pivot_final.sort_values('feo')


# ### Schwefel function
# 
# To see, how this parameters setup could be useful for another functions, another objective function can be tested with this set of parameters. Schwefel function, as described here (http://www.geatbx.com/docu/fcnindex-01.html#P150_6749) is good candidate, because of its different nature. This function has a lot of local minimum points and algorithms may tend to stuck in local minimum instead of finding global minimum.

# #### Best setup of Rosenbrock

# In[69]:

schwefel2 = Schwefel(n=2, eps=0.00001)
fstar = schwefel2.evaluate(420.9687 * np.ones(2))


# In[68]:

de_results_rand = pd.DataFrame()
F=0.3
N=20
CR=0.9
mutations = initRandMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=schwefel2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_rand = pd.concat([de_results_rand, de_res], axis=0)
    
de_results_best= pd.DataFrame()
F=0.3
N=20
CR=0.9
mutations = initBestMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=schwefel2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_best = pd.concat([de_results_best, de_res], axis=0)
    
de_results_last= pd.DataFrame()
F=1.0
N=20
CR=0.97
mutations = initLastMutations(F, N, CR)
for mutation in mutations:
    de_res = experiment_de(of=schwefel2, maxeval=maxeval, num_runs=NUM_RUNS, N=N, CR=CR, F=F, mutation=mutation)
    de_results_last = pd.concat([de_results_last, de_res], axis=0)
    
de_results_final = pd.concat([de_results_rand,de_results_best,de_results_last], axis=0)
de_results_pivot_final = de_results_final.pivot_table(
    index=['heur'],
    values=['neval'],
    aggfunc=(rel, mne, feo)
)['neval']
de_results_pivot_final = de_results_pivot_final.reset_index()

de_results_pivot_final.sort_values('feo')


# For this function, the goal was to find optimum with high precision. Despite this precision and previous assumptions about the complexity of global optimum search, DE work better this time than in the case of Rosenbrock function, even only with parameters tuned to Rosenbrock function. For this function, with a lot of global minima, the power of more complex mutations with finding the best individiual in population is emphasized, because these methods was the best according to Feocristov criterion.
# 
# On this example, it can be seen, that random mutations were mediocore and I would say, they should be used as the first candidate for all functions, because the results with random mutations, in my experiments, were never criticaly bad (oposed to other mutations with bad parameters setup) and are the fastest.

# ## Conclusion
# The goal of this notebook was not to find the best parameters setup for optimizing Rosenbrock function with Differential Evolution, but to tweak the parameters for each of implemented mutation. Some assumptions about influence of parameters was made based on the knowledge of the heuristics and some of the parameter tuning was made with aposterior information by experimenting with different setups.
# 
# Rosenbrock function can be characterized more as simple task than hard, because it is unimodal. This unimodality could favor methods using best individual in population when mutating, because there is no risk of stucking in local minimum. It showed up, that best mutation to be used is truly based on finding best individual. It also showed up, that totally random approach is very usefull, because classical random mutation was fast, reliable and with small number of evaluations. If there is need to keep machine time to the lowest, classical random mutation (rand1) should be used, as it was approximately twice as fast as best2, but if time consumption is not so much important, than the mutation is best2 with great reliability and low number of evaluations.
# 
# Other methods showed to be mediocore for this function, but as seen in final example, randToBest1 with $F = 1$ was a good choice for function with a lot of local minima, as well as currentToBest1 with $F = 1$. On this example, with Shwefel function and these two mutations with different parameter setup, it is pretty obvious the impact of NO FREE LUNCH theorem. The mutation itself was good only with some setup, but with another it failed miserably opposed to random mutation.
# 
# This notebook does not show the best parameter setup for DE, but demonstrates some ways of using different mutations for DE on problem of optimization of unimodal 2D function with nontrivial solution.

# In[ ]:



