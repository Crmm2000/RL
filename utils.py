import numpy as np

def action_to_vector(action):
    vectors = {
        0: ([-1, 0], '↑'),  # up
        1: ([1, 0], '↓'),   # down
        2: ([0, -1], '←'),  # left
        3: ([0, 1], '→')    # right
    }
    vector, symbol = vectors.get(action, ([0, 0], ''))
    return np.array(vector), symbol

def one_hot(state, nstates):
    a = np.zeros(nstates)
    a[state] = 1
    return a

#only accurate if there is 1 optimal action
def policy_visual(env, policy):
    policies = []
    for state in range(env.nstates):
        policies.append(env.agent.actions.get(np.argmax(policy[state]))[1])
    return print(env.fit_grid(np.array(policies)))
