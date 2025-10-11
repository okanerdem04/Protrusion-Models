import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

@numba.njit
def growth(lattice,cell_id,size,num_cells,ao,temp2): # calculates energy due to departure from targeted area
    # inputs: 2d np array lattice, 1d np array cell_id, int size, int num_cells, int ao
    # returns: int ham

    # for each cell id, store the number of total lattice spaces it occupies
    asig = np.zeros((num_cells+1), dtype=np.int64)
    
    ham = 0

    for i in range(size): # go through each cell
        for j in range(size):
            val = lattice[i][j]
            asig[val] = asig[val] + 1 # increase the energy Asig for each cell id if the lattice space contains said id

    for i in range(1,num_cells):
        ham += (asig[i]-ao)**2 # increase the hamiltonian by the square of the difference of the size of the cell and its preferred size
        
    # print(asig)
    return ham*temp2 # return the value of the hamiltonian

@numba.njit
def tension(lattice,cell_id,size,num_cells,temp): # calculates energy due to surface tension of cell membranes
    # inputs: 2d np array lattice, 1d np array cell_id, int size, int num_cells, int temp
    # returns: int ham

    ham = 0

    for n in range(1,num_cells): # go through each individual cell
        # then go through each cell space, adding 1 to ham for each adjacent space not belonging to cell
        for i in range(size):
            for j in range(size): # in theory we only need to do two comparisons for this to fully work?
                if i < size-1: # in order to not access a thing outside of range, we do this
                   if lattice[i][j] != lattice[i+1][j]: ham += 1
                if j < size-1:
                   if lattice[i][j] != lattice[i][j+1]: ham += 1

    ham = ham * temp # multiply by our temperature, i.e. surface tension coefficient
    return ham

def calc_hamiltonian(lattice,cell_id,size,num_cells,ao,temp,temp2):
    ham = 0
    ham += tension(lattice,cell_id,size,num_cells,temp)
    ham += growth(lattice,cell_id,size,num_cells,ao,temp2)
    return ham



