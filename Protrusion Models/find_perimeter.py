import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

from calc_hamiltonian import neighbors

@numba.njit
def find_perimeter(lattice,width,height,ind): # find the perimeter of a single cell with a given id
    perimeter_lattice = np.zeros((width,height),dtype=np.int64) # temporary blank lattice

    # go through each element in array
    for i in range(width):
        for j in range(height):
            # see if the selected element matches the id of the cell we are looking for
            if lattice[i,j] == ind:
                for nx, ny in neighbors(i,j,width,height):
                    if lattice[i,j] != lattice[nx,ny]:
                        perimeter_lattice[nx,ny] = 1

    #print(np.sum(perimeter_lattice == 1))
    return (np.sum(perimeter_lattice == 1))