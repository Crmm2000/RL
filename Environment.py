#%%
import numpy as np
import seaborn as sns
import matplotlib as plt
from Grid import grid

#%%
class environment(grid):
    'The environment obeys the laws/symmetries of the gridworld, but keeps track of everything thats in it, including the agents, thus this is what the agent interacts with '
    """
    Maybe also internal dynamics: t increments are environment steps, additionally
    Since the environment is markov it is not necessary to keep explicit track of the history

    Interaction:
        Gets:
        Action agent At
    
        Emits:
        Observation Ot+1
        Reward Ot+1

    Note that this also entails that the t-increments happen at the environment steps
    """
    # TO DO: dynamics, how actions change
    
    def __init__(self, width = 4, height = 4):
        super().__init__(width, height) # environment exists in grid

        'Model'
        

        #Reward = scalar feedback signal
        self.R = np.zeros((self.x1, self.x2)) 

        #set possible absorbing states
        self.absorbing_state = None #nstates - 1 

        #Transition dynamics environment P(s', s): Pss' = P[St+1 = s'| St = s] (thus how actions change a state)
        self.P = np.ones((self.nstates, self.nstates)) / self.nstates #initialize
        

        'Dynamics -> how actions change the state'
    def add_absorbing_state(self, state):
        self.absorbing_state = state
        if self.absorbing_state != None:
            self.P[self.absorbing_state] = 0
    
    def add_forbidden_state(self, forbidden_states):
        self.forbidden_states.append(forbidden_states)
        for i in forbidden_states:
            self.states_id[i] = np.nan
            self.R[self.coordinates[i]] = np.nan
            self.P[:, i] = 0 #-> so technically, this would be dependent on the agent, think about later
        self.P = self.P/ self.P.sum(axis=1, keepdims=True)
        self.nstates = (self.x1*self.x2) - len(self.forbidden_states)
    
    # Static objects
    def add_obstacle(self, states_obstacles, obstacle_type = 'wall'):
        obstacle_coordinates = [] #create list with coordinates, only useful for plotting
        for i in states_obstacles:
            obstacle_coordinates.append([self.coordinates[i]])
            
            if obstacle_type == 'wall':
                self.state_names[i] = f's{i}: wall' #make sure there is no state in the obstacles

                # Set obstacle dynamics
                self.P[:, i] = 0 #Chance of getting into a state where the wall is
                
                
            self.P = self.P/ self.P.sum(axis=1, keepdims=True)
        return obstacle_coordinates

    
    'Visuals'
    def plot_dynamics(self):
        s = sns.heatmap(self.P, annot = True, annot_kws = {'size': 5}, yticklabels = self.state_names, xticklabels = self.state_names)
        s.set(title = f'P(s,s+1) = Transition dynamics', xlabel='S+1', ylabel='S')


    def plot_rewards(self):
        s = sns.heatmap(self.R, annot = True, annot_kws = {'size': 5})
        s.set(title = f'Rewards: R = E[Rt+1|S]') # this means you only get the reward after the environment adds time?


    def plot_landscape(self, surface):
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.plot_surface(self.X1, self.X2, self.fit_grid(surface))
        ax.set_zlim(0,1)

    def heatmap(self, surface):
        sns.heatmap(self.fit_grid(surface), annot = True)
