# Chapter 2 from Sutton and Barto

#%%
import numpy as np
import utils as U
from Environment import environment

#%%
def sample_average(k = 5, t_end = 10):
    bandit = environment(1, k) #create environment with k levers
    bandit.R = np.random.rand(bandit.map.shape[0], bandit.map.shape[1]) #set random rewards
    
    #lists for bookkeeping
    R_cum = np.zeros((k,1)) #sum of rewards for action 
    Nt = np.zeros((k,1)) #number of times action is taken

    t = 0
    while t != t_end:
        At = np.random.choice(bandit.states_id) #pick random action
        Rt = bandit.R * U.one_hot(At, bandit) #get reward for action
        R_cum[At] += Rt[Rt > 0] #sum rewards for acton
        Nt[At] += 1 #number of times action is taken
        Qt = np.divide(R_cum, Nt, where = Nt!=0) #action value
        
        t+=1
    return Qt
# %%
