import numpy as np
import utils as u
import matplotlib.pyplot as plt
import seaborn as sns

# Agent -> to do: there is some double code, mainly in the transition matrices and the policy
#get different models for agent: model-free, model-based, valuebased, policybased, actor critic

# Maybe we can make the other agents inherit from this basic agent

#-> fix visuals -> I think the general approach works now
#-> notation, env, start_state, goal_state, belong to the same agent, so they can be part of self?
#-> lot to cover, also fix policy iteration and stuff
# -> some things belong to the env and not the agent

class Agent_MDP():
    # So what does an agent do? Make actions and accumulate information (= Bookkeeping)
    def __init__(self, env, start_state, absorbing_state, goal_state, penalty_steps = -0.1, color = 'r'):
        'Some features'
        self.color = color 
        self.r = 3 #range
        self.x = start_state #np.array(env.coordinates[start_state]) #Set our state equal to a state in the environment

        
        'Model environment'
        self.env = env #Fully observable

        
        'Actions: A (s)'
        self.choices = np.array([0, #up
        1, #down
        2, #right
        3]) #left

        self.P = self.get_transition(env = env, absorbing_state = absorbing_state, actions = self.choices)
        self.policy = np.ones((env.nstates, len(self.choices))) * (1/len(self.choices)) #P(a|s)
        #np.ones(len(self.choices)) / len(self.choices) #initialize random policy -> so technically this is not a policy, or it's just the same for all states which doesnt make it that flexible
        'Bookkeeping - > memory of actions, sense of self, we can also integrate this into the model of the environment?'
        'Reward'
        env.R = np.ones((env.nstates,len(self.choices))) * penalty_steps #each step has same penalty , R(s,a)
        env.R[goal_state] = 0
        self.state_values = np.zeros(env.nstates) #value based
        self.q_values = np.zeros((len(self.choices), env.nstates))
    
    
    #P(s'|s,a)
    def get_transition_action(self, env, absorbing_state, action): # -? again somewhere here it should be easier, the boundarie conditions shouldnt be necessary
        P_matrix = np.zeros((env.nstates, env.nstates))
        for i in env.states_id:
           
            #dist_successors = np.zeros(env.nstates)
            current_state = env.coordinates[i] # s
            a = u.action_to_vector(action) #a
            successor_coords = current_state + a # -> my main problem is in this phase, the change should be in the action to vector thing
            successor_state = env.coords_to_state(successor_coords)
            dist_successors = u.one_hot(successor_state) * env.P[self.x][successor_state] #store probability for each state
            
            #Normalize
            if np.sum(dist_successors) > 0:
                dist_successors = dist_successors / np.sum(dist_successors)
            P_matrix[i] = dist_successors
            if absorbing_state != None:
                P_matrix[absorbing_state] = 0 #add absorbing state
        #P_matrix = P_matrix/ P_matrix.sum(axis=1, keepdims=True)
        return P_matrix 

    #R = E(R' | s, a)
    def agent_reward(self, env, actions):
        current_state = self.x
    
    #P(s'|s) _> can do this shorter by summing over get_transition_actions and normalizing
    def get_transition(self, env, absorbing_state, actions): # -? again somewhere here it should be easier, the boundarie conditions shouldnt be necessary
        P_matrix = np.zeros((env.nstates, env.nstates))
        for i in env.states_id:
           
            dist_successors = np.zeros(env.nstates)
            current_state = env.coordinates[i] # s
            for j in actions:
                a = u.action_to_vector(j) #a
                successor_coords = current_state + a # -> my main problem is in this phase, the change should be in the action to vector thing
                successor_state = env.coords_to_state(successor_coords)
                dist_successors += u.one_hot(successor_state, env = env) * env.P[self.x][successor_state] #store probability for each state
            
            #Normalize
            if np.sum(dist_successors) > 0:
                dist_successors = dist_successors / np.sum(dist_successors)
            P_matrix[i] = dist_successors
            if absorbing_state != None:
                P_matrix[absorbing_state] = 0 #add absorbing state
        return P_matrix

    # Visuals
    def plot_pmatrix_agent(self):
        plt.figure()
        sns.heatmap(self.P)
        plt.title(f"P(s'|s)")
        plt.show()

    def plot_pmatrix_actions(self, env):
        plt.figure()
        for i in range(len(self.choices)):
            sns.heatmap(self.get_transition_action(env, self.choices[i]))
            plt.title(f"P(s'|s,a): a = {self.choices[i]}")
            plt.show()
    
    # Simulate step
    def get_action(self):
        next_action = int(np.random.choice(self.choices, p = self.policy[self.x]))
        return next_action
    
    def take_step(self, env, action): #not really an interaction with the environment -> todo
        location = env.coordinates[self.x] 
        next_location = location + u.action_to_vector(action)
        successor_state = env.coords_to_state(next_location)
        self.x = successor_state
        return successor_state
        #next_step = action sampled from policy
        #self.x = self.x + self.actions[action]

    def q_run(self, env, t_end, start_state, goal_state, start_action, gamma = 0.9):
        #First bookkeeping activity
        accumulated_reward = 0
        state_return = np.zeros(env.nstates)
        t = 1
        terminate = False
        
        self.x = start_state #initize

        # Take fist step
        self.take_step(start_action)

        while t != t_end and terminate == False:  
            action = self.get_action()
            reward = env.R[self.x][action] #Rt+1
            discounted_rewards = (gamma**(t-1)) * reward  #gamma**t * Rt+1
            accumulated_reward += discounted_rewards #will be the return after the loop
            
            self.take_step(action)
            t += 1
            
            if self.x == goal_state:
                terminate = True
                reward = env.R[self.x][action] #Rt+1
                discounted_rewards = (gamma**(t-2)) * reward  #gamma**t * Rt+1
                accumulated_reward += discounted_rewards #will be the return after the loop
        Gt = accumulated_reward #return
        state_return[start_state] = Gt 
        return state_return
    

    def run(self, t_end, env, start_state, goal_state, gamma =0.9, visual = False): #The visual code is quite messy but I really like the visuals so we'll have to figure a way out to deasl wiht it   
        # if visual == True:
        #     %matplotlib
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection = '3d')
        #     plt.ion() #interactive window

        
        #First bookkeeping activity
        accumulated_reward = 0
        state_return = np.zeros(env.nstates)
        t = 1
        terminate = False
        self.x = start_state #initize
        #Step loop
        while t != t_end and terminate == False:
            action = self.get_action()
            reward = env.R[self.x][action] #Rt+1
            discounted_rewards = (gamma**(t-1)) * reward  #gamma**t * Rt+1
            accumulated_reward += discounted_rewards #will be the return after the loop
            
            self.take_step(env, action)
            t += 1
            
            # if visual == True:
            #     ax.cla()
            #     ax.plot_surface(env.X1, env.X2, env.fit_grid(state_return))
            #     ax.set_title(f'MDP episode {t}, state {a.x}')
            #     ax.plot(env.coordinates[self.x][0],env.coordinates[self.x][1], 0.1, 'ko')
            #     ax.plot(np.repeat(env.coordinates[self.x][0], 100) , np.repeat(env.coordinates[self.x][1], 100), np.linspace(start=0.1, stop=0, num=100) , 'k')
            #     plt.draw()
            #     plt.pause(0.01)  # animation
            
            if self.x == goal_state:
                terminate = True
                reward = env.R[self.x][action] #Rt+1
                discounted_rewards = (gamma**(t-2)) * reward  #gamma**t * Rt+1
                accumulated_reward += discounted_rewards #will be the return after the loop
        
        # if visual == True:
        #     ax.set_zlim(0,0.1)
        #     plt.show()

        Gt = accumulated_reward #return
        
        state_return[start_state] = Gt 
        return state_return 

    def value_function_MDP(self, env, episodes, visualise = False):
        # if visualise == True:
        #     %matplotlib
        #     fig = plt.figure()
        #     ax = fig.add_subplot(projection = '3d')
        #     ax.set_title('Value function: v(s) = E(gt | st)')
        state_returns = np.zeros(env.nstates)
        for i in range(episodes):
            # Cumulate state returns
            for j in range(env.nstates): #iterate over starting states
                 #later: can also keep track of all things within agent
                state_returns = state_returns + self.run(100, start_state = j)
    
            # # Visualise value function for each episode
            # if visualise == True:
            #     plt.cla()
            #     ax.plot_surface(env.X1, env.X2, -env.fit_grid(state_returns/(i+1)))
            #     #sns.heatmap(env.fit_grid(state_returns/(i+1)), cbar = False, annot = True) #expected value
            #     plt.suptitle(f'episode: {i+1}')
            #     plt.draw()
            #     plt.pause(0.1)
            
        # Expected value: State value E(gt | st = 14)
        self.state_values = state_returns/episodes
        # if visualise == True:
        #     plt.close(fig)
        return env.fit_grid(self.state_values)

    def q_function_MDP(self, env, episodes, visualise = False):
        state_action_returns = np.zeros((len(self.choices), env.nstates))
        for i in range(episodes):
            # Cumulate state returns
            for j in range(env.nstates): #iterate over starting states
                #cumulate action returns
                for k in range(len(self.choices)):
                    state_action_returns[k] = state_action_returns[k] + self.q_run(100, start_state = j, start_action = k)
            
        # Expected value: State value E(gt | st = 14)
        self.q_values = state_action_returns/episodes
        return self.q_values#env.fit_grid(state_values)