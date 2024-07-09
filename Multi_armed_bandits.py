# Chapter 2 from Sutton and Barto
#%%
import numpy as np
import utils as U
from Environment import environment
import matplotlib.pyplot as plt
#todo -> convert to notebook, -> set dictionairy
#create graphs

def e_greedy_selection(Qt, epsilon = 0.5):
    nstates = len(Qt)
    dist = np.full(nstates, epsilon / nstates)
    dist[np.argmax(Qt)] += 1 - epsilon
    return np.random.choice(nstates, p = dist)

action_selection_strategies = {
    'greedy': e_greedy_selection,
    'random': lambda Qt: np.random.choice(k)
}

#initialize
k = 10
runs = 2000
t_end = 1000
bias = 0
epsilon = 0

def sample_average(k = k, epsilon = epsilon, t_end = t_end, selection = 'greedy'):
    bandit = environment(1, 1) #create environment, single state
    bandit.R = np.random.normal(size = k) #q*
    #lists for bookkeeping
    R_cum = np.zeros(k) #sum of rewards for action 
    Nt = np.zeros(k) #number of times action is taken
    Qt = np.full(k, bias)
    reward_log = np.zeros(t_end)

    t = 1
    while t != t_end:
        At = action_selection_strategies[selection](Qt, epsilon = epsilon)
        R = bandit.R * U.one_hot(At, k)  #np.random.choice(bandit.R, p = dist) #get reward for action
        q_star = R[R != 0][0] #actual reward
        Rt = np.random.normal(q_star) #received reward
        R_cum[At] += Rt #sum rewards for acton
        Nt[At] += 1 #number of times action is taken
        Qt = np.divide(R_cum, Nt, where = Nt!=0) #action value
        
        #bookkeeping
        reward_log[t] = Rt
        t+=1
    return Qt, reward_log

def run(runs = runs, epsilon = epsilon, selection = 'greedy'):
    rewards_runs = np.zeros(t_end)
    for i in range(runs):
        rewards_runs += sample_average(epsilon = epsilon, selection = selection)[1]
    average_reward = np.divide(rewards_runs, runs)
    return average_reward


# %%
epsilon = [0, 0.01, 0.1]
fig = plt.figure()
axs = fig.subplots(2,1)
for i in epsilon:
    values = run(epsilon = i)
    axs[0].plot(np.arange(0, t_end), values, label = f'eps = {i}')
    axs[1].plot(np.arange(0, t_end), values/1.54, label = f'{i}')

axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Average reward')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('% Optimal action')
plt.legend()
plt.show()

# %%
