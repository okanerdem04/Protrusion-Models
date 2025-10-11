import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

def gen_lattice(lattice, cell_id, size, num_cells): # function to create a lattice
    # assign each cell id a single position on the lattice
    '''for i in range(num_cells-1):
        rand_i = random.randint(3,size-3) # overwrites a zero in a random position
        rand_j = random.randint(3,size-3) # this can overwrite a non-zero value, maybe change?
        lattice[rand_i][rand_j] = cell_id[i]

        for j in range(rand_i-2,rand_i+1):
            for k in range(rand_j-2,rand_j+1):
                lattice[j][k] = cell_id[i]'''
    
    for i in range(5):
        for j in range(5):
            x_pos = i*20
            y_pos = j*20
            for k in range(-8,8):
                for l in range(-8,8):
                    lattice[x_pos+k][y_pos+l] = cell_id[j+i*5]
