# Chapter 2 from Sutton and Barto

#%%
import numpy as np
import utils as U
from Environment import environment

# -> RL uses training to evaluate actions, not to instruct
# -> evaluation: how good
# -> CH2 uses nonassociative setting

"A K-armed bandit problem"
# -> Choice among k different options
# -> after each choice get reward
# -> goal = maximize a reward

# %%
'q*(a) = E[Rt|At = a]'
# -> Qt(a) = estimated value
# -> exploiting - greedy, exploring - nongreedy

def sample_average(k = 5, t_end = 10):
    bandit = environment(1, k)
    bandit.R = np.random.rand(bandit.map.shape[0], bandit.map.shape[1])
    t = 0
    R_cum = np.zeros((k,1))
    Qt = np.zeros((k,1))
    Nt = np.zeros((k,1))

    while t != t_end:
        At = np.random.choice(bandit.states_id)
        Rt = bandit.R * U.one_hot(At, bandit)
        R_cum[At] += Rt[Rt > 0]
        Nt[At] += 1 
        Qt = np.divide(R_cum, Nt, where = Nt!=0)
        t+=1
    return Qt
# %%
