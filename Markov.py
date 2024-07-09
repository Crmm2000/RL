# Basically Lecture 2 from david silver

import numpy as np
import matplotlib as plt
import seaborn as sns

def markov_chain(n, start_state, goal_state, env, step_horizon = True, visual = False):
    # Initialization
    current_state = start_state 
    terminate = False
    t = 0
    t_end = n

    # Bookkeeping
    trajectory = [env.state_names[current_state]] #set up list for trajectory

    # Run chain
    while (t != t_end-1 and  not terminate):
        #Agent step
        current_state = int(np.random.choice(env.states_id, p = env.P[current_state])) #Thus for the decision we only need the states and P(s'|s)
        trajectory.append(f'{env.state_names[current_state]}')

        #Environment step
        t +=1

        #Termination conditions, if needed/wanted -> changes dynamics, if off just a time horizon
        if step_horizon == True:
            if (current_state == env.absorbing_state) or (current_state == goal_state) : #is techinically already true but nice to break it off earlier
                terminate = True
                trajectory.append('GOAL')
    print(f'Trajectory: {trajectory}')


#todo, occupancy analog -> what does it  what we have access to, so we're for now keeping histories out of it, we don't remember anything
def markov_reward_process(n, start_state, goal_state, env, gamma =0.9, visual = False, step_horizon = True):
    #this markov system has no memory for occupancy, so if it visits the same state twice, it will just override it
    # if visual == True:
    #     %matplotlib
    #     plt.figure(figsize = (2,2))
    #     plt.ion
    #     plt.title('Markov Reward Process')
    
    # Initialization
    current_state = start_state 
    terminate = False
    t = 1
    t_end = n
    
    # General bookkeeping 
    accumulated_reward = 0 #At the end this should be the same as the return Gt for the whole run

    # Loop to take steps
    while (t != t_end-1 and  not terminate):
        # Bookkeeping -> everything that's bookkeeping requires memory or resources
        current_state = int(current_state)
        reward = env.R.flatten()[current_state] #Rt+1
        discounted_rewards = (gamma**(t-1)) * reward  #gamma**t * Rt+1
        accumulated_reward += discounted_rewards #will be the return after the loop

        # Check termination conditions
        if (current_state == env.absorbing_state) or (current_state == goal_state): #is techinically already true but nice to break it off earlier
            terminate = True
            break
        
        # Agent step
        next_state = np.random.choice(env.states_id, p = env.P[current_state]) 

        #Environment step -> so normally you would communicate with the environment a bit earlier, and the environment will give you back a reward
        t += 1
        
        current_state = next_state
        #plt.show()

    # Return
    Gt = accumulated_reward

    #Return to state
    state_return = np.zeros(env.nstates)
    state_return[start_state] = Gt 
    
    #so we create a list for each state and keep track of amount of times it was visited, which we already did with the occumpancy, so we can just accumulate the value and divide it by the occpancy
    #so create an accumulated value, then an accumulated occupancy, divide one by the other and you should get the appropriate predicted value
    return state_return #, expected_rewards, state_values, discounted_occupancy


# Value function
def value_function(episodes, env, visualise = False):
    # if visualise == True:
    #     %matplotlib widget
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection = '3d')
    #     ax.set_title('Value function: v(s) = E(gt | st)')
    
    state_returns = np.zeros(env.nstates)
    for i in range(episodes):
        # Cumulate state returns
        for j in range(env.nstates): #iterate over starting states
            state_returns = state_returns + markov_reward_process(n = 50, start_state = j, goal_state = 4, env = env, gamma = 0.9, visual = False)

        # # Visualise value function for each episode
        # if visualise == True:
        #     plt.cla()
        #     ax.plot_surface(env.X1, env.X2, -env.fit_grid(state_returns/(i+1)))
        #     #sns.heatmap(env.fit_grid(state_returns/(i+1)), cbar = False, annot = True) #expected value
        #     plt.suptitle(f'episode: {i+1}')
        #     plt.draw()
        #     plt.pause(0.1)
        
    # Expected value: State value E(gt | st = 14)
    state_values = state_returns/episodes
    # if visualise == True:
    #     plt.close(fig)
    return env.fit_grid(state_values)