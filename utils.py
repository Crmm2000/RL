import numpy as np

# Helper functions

def action_to_vector(action):
    vector = np.zeros(2)
    if action == 0: #up
        vector[0] = -1
    elif action == 1: #down
        vector[0] = 1 
    elif action == 2: # 'left':
        vector[1] = -1
    elif action == 3: #'right':
        vector[1] = 1
    return vector

def one_hot(state, env):
    a = np.zeros(env.nstates)
    a[state] = 1
    return a

def normalized(length, states):
    a = np.zeros(length)
    for i in states:
        a[i] = 1
    a = a / np.sum(a)
    return a
#coordinates to state