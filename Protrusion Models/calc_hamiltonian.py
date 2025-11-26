import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

@numba.njit
def find_areas(lattice,width,height,num_cells): # calculates the total number of pixels belonging to each cell on the lattice
    # inputs: 2d npArray lattice, int width, int height, int num_cells

    # create a npArray of zero ints, with the length being equal to the number of unique cells
    areas = np.zeros((num_cells+1), dtype=np.int64)

    # for each pixel on the lattice, increase the value of the index in "areas" that matches the value of the selected pixel
    for i in range(width):
        for j in range(height):
            if lattice[i][j] % 3 != 2: # lattice % 3 != 2 are all pixels that aren't inactive protrusions
                areas[lattice[i][j]//3] += 1
        
    return areas # return the array storing the areas of each cell

@numba.njit
def find_protrusions(lattice,width,height,num_cells): # calculates the total number of pixels belonging to inactive protrusions for each cell
    # inputs: 2d npArray lattice, int width, int height, int num_cells

    # create a npArray of zero ints, with the length being equal to the number of unique cells
    protrusions = np.zeros((num_cells+1), dtype=np.int64)

    # for each pixel on the lattice, increase the value of the index in "protrusions" that matches the value of the selected pixel
    for i in range(width):
        for j in range(height):
            if lattice[i][j] % 3 == 2: # lattice % 3 == 2 are all pixels that are inactive protrusions, i.e. not cell bodies or protrusion tips
                protrusions[lattice[i][j]//3] += 1
        
    return protrusions # return the array storing the total number of inactive protrusions for each cell

@numba.njit
def neighbors(x, y, width, height):
    # inputs: int x, int y, int width, int height

    # return a 2d list containing the indices of the four neighbouring points to [x,y], accounting for the fact that the edges of the lattice are connected
    return [((x+1)%width, y), ((x-1)%width, y), (x, (y+1)%height), (x, (y-1)%height)]

@numba.njit
def adhesion_energy(lattice,width,height,x,y): # for a given lattice point [x,y], calculate how many of its neighbouring points belong to different cells
    # inputs: 2d npArray lattice, int width, int height, int x, int y
    energy = 0

    for nx, ny in neighbors(x,y,width,height):
        energy += (lattice[nx, ny] != lattice[x,y])

    # return the number of neighbours belonging to different cells
    return energy

@numba.njit
def calc_hamiltonian(lattice,width,height,num_cells,area_coeff,a0,adhesion_coeff,protrusion_coeff,p0,x,y): # calculate the overall Hamiltonian function of the lattice
    # inputs: 2d npArray lattice, 1d npArray cell_id, int width, int height, int num_cells, int area_coeff, int a0, int adhesion_coeff, int protrusion_coeff, int p0, int x, int y
    ham = 0

    # find the total area constraint energy component, based on lambda and a0
    areas = find_areas(lattice,width,height,num_cells)
    for i in range(1,num_cells+1):
        ham += area_coeff * ((areas[i]-a0) ** 2)

    # find the total protrusion constraint energy component, based on mu and p0
    protrusions = find_protrusions(lattice,width,height,num_cells)
    for i in range(1,num_cells+1):
        ham += protrusion_coeff * ((protrusions[i]-p0) ** 2)

    # find the adhesion energy around one point, based on alpha
    # this does *not* find the total adhesion energy, as that takes much longer to calculate and we're only interested in the changes of energy when swapping one point
    adhesion = adhesion_energy(lattice,width,height,x,y)
    ham += adhesion_coeff*adhesion

    # return the total Hamiltonian of the system
    return ham



