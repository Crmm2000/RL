import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Grid(): #S
    def __init__(self, width, height):
        self.map = np.zeros(shape = (height, width)) 
        self.nstates = width*height
        self.states = {i: f's{i}' for i in range(width*height)}
        self.states_id = [i for i in range(width * height)]

        # Coordinates grid world
        self.x1, self.x2 = width, height
        self.X1, self.X2 = np.meshgrid(np.arange(width), np.arange(height)) #Create meshgrid
        self.grid = [self.X1, self.X2]
        self.coordinates = [(self.X1[i, j], self.X2[i, j]) for j in range(width) for i in range(height)] #coordinates (row, collumn)
        
    'Aux'
    def fit_grid(self, array):
        return np.reshape(array, (self.x1, self.x2))
    
    def coords_to_state(self, coords): 
        coords[0] = min(max(coords[0], 0), self.x1 - 1)
        coords[1] = min(max(coords[1], 0), self.x2 - 1)
        return self.fit_grid(range(self.nstates))[int(coords[0])][int(coords[1])]
    
    def plot_grid(self):
        sns.heatmap(self.fit_grid(self.states_id), annot = self.fit_grid(list(self.states.values())), cbar = False, xticklabels = False, yticklabels= False, fmt = 's')