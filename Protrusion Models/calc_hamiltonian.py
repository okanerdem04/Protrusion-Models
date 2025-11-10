import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

@numba.njit
def find_areas(lattice,cell_id,width,height,num_cells): # find the areas of all cells
    # inputs: 2d np array lattice, 1d np array cell_id, int width, int height, int num_cells, int a0, int alpha
    # returns: 1d np array areas

    # for each cell id, store the number of total lattice spaces it occupies (i.e. its area)
    areas = np.zeros((num_cells+1), dtype=np.int64)

    for i in range(width): # go through each cell
        for j in range(height):
            areas[lattice[i][j]//3] += 1
        
    return areas # return the value of the hamiltonian

@numba.njit
def neighbors(x, y, width, height): # takes an x and y position, returns all x and y positions directly adjacent to it
    return [((x+1)%width, y), ((x-1)%width, y), (x, (y+1)%height), (x, (y-1)%height)]

@numba.njit
def adhesion_energy(lattice,width,height,x,y): # takes the lattice, goes through each cell, sums up how many different adjacents each cell has
    energy = 0

    for nx, ny in neighbors(x,y,width,height):
        energy += (lattice[nx, ny] != lattice[x,y]) # adds to the energy depending on how many adjacent tiles belong to same cell

    return energy

@numba.njit
def calc_hamiltonian(lattice,cell_id,width,height,num_cells,a0,alpha,lambd,x,y):
    ham = 0

    # area constraint energy
    areas = find_areas(lattice,cell_id,width,height,num_cells)
    for i in range(1,num_cells+1):
        ham += lambd * ((areas[i]-a0) ** 2) # hamiltonian increases for each cell depending on how far its area is from the target

    # adhesion energy - just calculated for the swapped cells, since going through the entire grid takes a long time
    adhesion = adhesion_energy(lattice,width,height,x,y)
    ham += alpha*adhesion

    return ham



