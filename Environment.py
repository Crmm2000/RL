import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from Grid import Grid
import utils as u
from IPython import display

# S, P
class Environment(Grid): 
    """
    The environment obeys the laws/symmetries of the gridworld,
    it has the Markov property, this is what the agent (decision-making entity) interacts with.
    T-increments occur at the environment steps, indicating the point at which a decision is made
    Dynamics => P[s' | s, a]
    
    Interaction:
        Env Gets:
        Action agent At
    
        Env Emits:
        Observation Ot+1 (= St+1 in fully observable MDP)
        Reward Rt+1
    """

    def __init__(self, width, height):
        Grid.__init__(self, width, height) # environment exists in grid

        # Transition dynamics environment, Pss' = P[St+1 = s'| St = s]
        self.P = np.ones((self.nstates, self.nstates)) / self.nstates #Uniform
        self.absorbing_state = [] 
        self.forbidden_states = []

    # Add absorbing state
    def add_absorbing_states(self, states):
        self.absorbing_state.extend(states) 
        for state in states:
            self.P[state] = u.one_hot(state, self.nstates)
    
    # Inaccessible states
    def add_forbidden_states(self, forbidden_states):
        self.forbidden_states.extend(forbidden_states)
        for state in forbidden_states:
            self.P[:, state] = 0  
        self.P = np.divide(self.P, self.P.sum(axis=1, keepdims=True)) 
    
    'Aux'
    def plot_dynamics(self):
        s = sns.heatmap(self.P, annot = True, annot_kws = {'size': 5}, yticklabels = list(self.states.values()), xticklabels = list(self.states.values()))
        s.set(title = f'P(s+1|s) = Transition dynamics', xlabel='S+1', ylabel='S')

    def heatmap(self, values):
        sns.heatmap(self.fit_grid(values), annot = True)
    
    def plot_landscape(self, surface):
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.plot_surface(self.X1, self.X2, self.fit_grid(surface))


# S, P, R, GAMMA
class MRP(Environment, Grid): # 
    def __init__(self, width, height, gamma):
        Environment.__init__(self, width, height) 
        self.R = np.zeros(self.nstates) 
        self.gamma = gamma
    
    'P'
    def add_forbidden_states(self, forbidden_states):
        super().add_forbidden_states(forbidden_states)
        return 

    'R'
    def set_rewards(self, goal_states, goal_reward, penalty_steps = 0):
        goal_states = np.array([goal_states])
        self.R = self.R + penalty_steps 
        for state in goal_states:
            self.R[state] = goal_reward

    # Reward function: E[Rt+1|s] -> immediate reward
    def reward_function(self, state_agent):
        return self.R[int(state_agent)] 

    'Auxilliary functions'
    def plot_rewards(self):
        s = sns.heatmap(self.fit_grid(self.R), annot = True, annot_kws = {'size': 5})
        s.set(title = f'Rewards: R = E[Rt+1|S]') 



# S, P, R, A
class MDP(MRP, Environment, Grid): 
    def __init__(self, width = 4, height = 4, gamma = 0.9):
        MRP.__init__(self, width, height, gamma) 
        
        #create an agent (something that can take actions)
        self.agent = self.Agent(self)

    'A'     
    class Agent: 
        # only the decision-making entity
        def __init__(self, mdp):
            self.actions = { # Assume actions are the same for each state
                0: ([-1, 0], '↑'), # up
                1: ([1, 0], '↓'),  # down
                2: ([0, -1], '←'), # left
                3: ([0, 1], '→')   # right
            } 

            self.n_actions = len(self.actions)
            self.policy = np.ones((mdp.nstates, self.n_actions)) / self.n_actions #explicit intial random policy
            mdp.R = np.zeros((self.n_actions, mdp.nstates)) #Rsa
            mdp.P = self.set_mdp_dynamics(mdp)
            
        def set_mdp_dynamics(self, mdp): #Pssa
            P_matrix = np.zeros((self.n_actions, mdp.nstates, mdp.nstates))
            for j in range(self.n_actions):
                for i in mdp.states_id:
                    P_matrix[j, i] = mdp.transition_matrix(action = j, state = i, absorbing_state = mdp.absorbing_state)
            return P_matrix
        
        def step(self, state):
            state = int(state)
            action = np.random.choice(self.n_actions, p = self.policy[state])
            return action
    
    'P'
    
    # P[s'|s,a] 
    def transition_matrix(self, state, action, absorbing_state = None):
        state = int(state)
        a = u.action_to_vector(action)[0] 
        
        successor_coords = self.coordinates[state] + a #movement, maybe noise can be added here?
        successor_state = self.coords_to_state(successor_coords) #movement to state
        successor_state = int(successor_state)

        if len(self.P.shape) == 2:
            if np.sum(self.P[state][successor_state]) < 0.001:
                successor_state = state
        else:
            if np.sum(self.P[action][state][successor_state]) < 0.001:
                successor_state = state
        
        if len(self.P.shape) == 2:
            dist_successors = u.one_hot(successor_state, self.nstates) * self.P[state][successor_state] #store probability for each state, or noise here?
        else:
            dist_successors = u.one_hot(successor_state, self.nstates) * self.P[action][state][successor_state] #store probability for each state, or noise here?
        
        #Normalize, get transition matrix (p(s'|s, a))
        p_matrix = np.divide(dist_successors, np.sum(dist_successors), where = np.sum(dist_successors) != 0)

        #adjust for absorbing state -> is this necessary?
        if absorbing_state != None:
            p_matrix[absorbing_state] = 0 
        
        return p_matrix

    # P[s',r|s,a], Markov
    def dynamics_environment(self, state, action, absorbing_state = None):
        #get p(s', r|s, a)
        transition_probs = self.transition_matrix(state = state, action = action, absorbing_state = absorbing_state) #p(s'|s,a)
        rewards = [self.R[action][int(next_state)] for next_state in transition_probs]
        probabilities = [prob for prob in transition_probs]
        states = np.arange(self.nstates)
        
        # Stack the rewards and probabilities -> get histogram for joint probability
        a = np.vstack((rewards, probabilities, states))

        #isn't acccurate enough 
        hist, xedges, yedges = np.histogram2d(a[2], a[0], bins = self.nstates-1, weights = a[1]) #get prob: [0][successor][reward]
        #safety check
        if np.sum(hist) != 1:
            raise ValueError("The sum of the array elements must be equal to 1.")
        return hist, xedges, yedges, transition_probs #p(s', r|s, a), p(s'|s,a) #technically you should be able to sum over the reward to get the latter, but haven't figured out how to deal without rounding it and reducing accuracy
    
    # Inaccessible states
    def add_forbidden_states(self, forbidden_states): #still needs work -> q_values don't update properly
        self.forbidden_states.extend(forbidden_states)
        for state in forbidden_states:
            self.P[:, :, state] = 0  
            self.R[:, state] = 0
        for i in range(self.agent.n_actions):
             self.P[i] = np.divide(self.P[i], self.P[i].sum(axis=1, keepdims=True), where = self.P[i].sum(axis=1, keepdims=True) != 0) 
    
    # Add absorbing state
    def add_absorbing_states(self, states):
        self.absorbing_state.extend(states) 
        for state in states:
            self.P[:, state] = u.one_hot(state, self.nstates)


    #P(s', s|a)
    def get_transition_action(self, action, absorbing_state = None): # -? again somewhere here it should be easier, the boundary conditions shouldnt be necessary
        P_matrix = np.zeros((self.nstates, self.nstates))
        for i in range(self.nstates):
            P_matrix[i] = self.dynamics_environment(action = action, state = i, absorbing_state=absorbing_state)[3]
        return P_matrix 

    #next state from p(s'|s, a)    
    def next_state(self, action, state, absorbing_state = None): 
        return np.random.choice(range(self.nstates, p = self.dynamics_environment(action=action, state=state, absorbing_state = absorbing_state)[3]))

    'R'
    #Adjust reward dynamics
    def set_rewards(self, goal_state, goal_reward = 1, penalty_steps = 0):
        self.R = self.R + penalty_steps 
        self.goal_state = goal_state
        self.R[:, goal_state] = goal_reward
    
    def reward_state_action(self): #r, where reward = E[Rt+1|s], thus we make sure it's weighted over the possible states
        # Calculate the rewards for the resulting states
        rewards = np.array([[np.dot(self.transition_matrix(state = j, action = i, absorbing_state = self.absorbing_state) , self.R[i]) 
                            for j in range(self.nstates)] 
                            for i in range(self.agent.n_actions)])
        return rewards
    
    #(r(s,a)) -> not true, successor state should be an input
    def reward_function(self, state, action): # I think now this technically is r(s,a,s')
        return self.reward_state_action()[action][state] 
    

    'Auxilliary functions'
    def render(self, state, surface):
        state = int(state)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X1, self.X2, surface.T, alpha=0.7)
        ax.plot(self.coordinates[state][0], self.coordinates[state][1], [1, 0], 'k-o', label=f'State: {state}')
        ax.plot(self.coordinates[self.goal_state][0], self.coordinates[self.goal_state][1], [0], 'o')
        plt.title(f'Gt: {self.agent.Gt}', fontsize = 12)
        plt.legend()

        display.display(plt.gcf())
        display.clear_output(wait=True)

