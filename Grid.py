import numpy as np
import seaborn as sns

class grid():
    'the idea of the grid is that even if the information is not accessible to the agent, the agent can not help but do it?'
    def __init__(self, width, height):
        self.map = np.zeros(shape = (width, height)) 
        self.states_id = [i for i in range(width*height)]
        self.state_names = [f's{i}' for i in range(width*height)] 
        self.nstates = width*height

        # Coordinates grid world
        self.x1, self.x2 = width, height
        self.X1, self.X2 = np.meshgrid(np.arange(width), np.arange(height)) #Create meshgrid
        self.grid = [self.X1, self.X2]
        self.coordinates = [(self.X1[i, j], self.X2[i, j]) for j in range(width) for i in range(height)] #coordinates (row, collumn)

        #forbidden_states 
        self.forbidden_states = []

    #-> what happens when we go out of our state space
    def add_forbidden_state(self, forbidden_states):
        self.forbidden_states.append(forbidden_states)
        for i in forbidden_states:
            self.states_id[i] = np.nan
            self.coordinates[i] = np.nan
            self.nstates = self.states - len(self.forbidden_states)

    # To do, align with env
    def coords_to_state(self, coords): #-> can place this in gridwordl
        # Boundaries
        if coords[0] < 0:
            coords[0] = 0
        if coords[1] < 0:
            coords[1] = 0
        if coords[0] > self.x1-1:
            coords[0] = self.x1-1
        if coords[1] > self.x2-1:
            coords[1] = self.x2-1
        new_coords = self.fit_grid(self.states_id)[int(coords[0])][int(coords[1])]
        return self.fit_grid(self.states_id)[int(coords[0])][int(coords[1])]

    # Visualisation
    def fit_grid(self, array):
        return np.reshape(array, (self.x1, self.x2))
    
    # Grid
    def plot_grid(self):
        sns.heatmap(self.fit_grid(self.states_id), annot = self.fit_grid(self.state_names), cbar = False, xticklabels = False, yticklabels= False, fmt = 's')