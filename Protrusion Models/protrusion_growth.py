import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

from calc_hamiltonian import neighbors

def center_of_mass(lattice,width,height,ind):
    x_lattice = np.array([]) # empty arrays to store all x, y positions
    y_lattice = np.array([]) 
    
    # go through each element in array
    for i in range(width):
        for j in range(height):
            # see if the selected element matches the id of the cell we are looking for
            if lattice[i,j] == ind:
                x_lattice = np.append(x_lattice,i)
                y_lattice = np.append(y_lattice,j)

    com_x = (np.sum(x_lattice) / np.size(x_lattice))
    com_y = (np.sum(y_lattice) / np.size(y_lattice))

    return [com_x, com_y]

