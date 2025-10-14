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
    areas = np.zeros((num_cells+1)*2, dtype=np.int64)

    for i in range(width): # go through each cell
        for j in range(height):
            areas[lattice[i][j]] += 1
        
    return areas # return the value of the hamiltonian

@numba.njit
def neighbors(x, y, width, height): # takes an x and y position, returns all x and y positions directly adjacent to it
    return [((x+1)%width, y), ((x-1)%width, y), (x, (y+1)%height), (x, (y-1)%height)]

@numba.njit
def diagonal_neighbors(x, y, width, height): # takes an x and y position, returns all x and y positions diagonal to it
    return [((x+1)%width, y+1), ((x+1)%width, y-1), (x-1, (y+1)%height), (x-1, (y-1)%height)]

@numba.njit
def adhesion_energy(lattice,width,height,x,y): # takes the lattice, goes through each cell, sums up how many different adjacents each cell has
    energy = 0

    for nx, ny in neighbors(x,y,width,height):
        # floor division by two to account for both cells and protrusions
        energy += (lattice[nx,ny] != lattice[x,y]) # adds to the energy depending on how many adjacent tiles belong to different cell

    return energy

@numba.njit
def protrusion_incentive(lattice,width,height):
    energy = 0

    for i in range(width): # slightly disgusting loop through the whole grid
        for j in range(height):
            if lattice[i,j] % 2 == 1: # check to make sure cell at that lattice point is a protrusion
                num_neighbours = 0
                for nx, ny in neighbors(i,j,width,height):
                    num_neighbours += (lattice[nx, ny] == lattice[i,j])

                for nx, ny in diagonal_neighbors(i,j,width,height):
                    num_neighbours += (lattice[nx, ny]//2 == lattice[i,j]//2)

                if num_neighbours <= 2: energy -= num_neighbours # reduces energy if the cell has two neighbours exactly. this should convince protrusions to grown in a line, not a clump
                elif num_neighbours >= 3: energy += num_neighbours # still discourage cells from being adjacent to more than two neighbours

    return energy

@numba.njit
def calc_hamiltonian(lattice,cell_id,width,height,num_cells,a0,alpha,lambd,x,y,target_protrusions):
    ham = 0

    # area constraint energy
    areas = find_areas(lattice,cell_id,width,height,num_cells)
    for i in range(1,num_cells+1):
        ham += lambd * ((areas[i*2]-a0) ** 2) # hamiltonian increases for each cell depending on how far its area is from the target

        if areas[i*2 + 1] >= target_protrusions-1: # only apply to protrusions once they are close to their target area
            ham += lambd * ((areas[i*2 + 1]-target_protrusions) ** 2) # apply the same principle for protrusions and their target area

    # adhesion energy - just calculated for the swapped cells, since going through the entire grid takes a long time
    if lattice[x,y]%2 == 0: # only do adhesion energy for cell bodies, not protrusions
        adhesion = adhesion_energy(lattice,width,height,x,y)
        ham += alpha*adhesion

    # protrusion incentive energy
    protrusion = protrusion_incentive(lattice,width,height)
    ham += protrusion

    return ham



