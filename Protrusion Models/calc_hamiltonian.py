import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba

@numba.njit
def find_areas(spins,compartments,width,height,num_cells): # calculates the total number of pixels belonging to each cell on the lattice
    # inputs: 2d npArray lattice, int width, int height, int num_cells

    # create a npArray of zero ints, with the length being equal to the number of unique cells
    areas = np.zeros((num_cells+1), dtype=np.int64)
    protrusions = np.zeros((num_cells+1), dtype=np.int64)

    # for each pixel on the lattice, increase the value of the index in "areas" that matches the value of the selected pixel
    for i in range(width):
        for j in range(height):
            if compartments[i][j] != 3: # do not count protrusion bodies in area calculations
                areas[spins[i][j]] += 1
            else:                       # any compartment that does not count towards soma area counts towards protrusion area instead
                protrusions[spins[i][j]] += 1
        
    return areas, protrusions # return the arrays storing the areas of each cell and protrusion

@numba.njit
def neighbors(x, y, width, height):
    # inputs: int x, int y, int width, int height

    # return a 2d list containing the indices of the four neighbouring points to [x,y], accounting for the fact that the edges of the lattice are connected
    return [((x+1)%width, y), ((x-1)%width, y), (x, (y+1)%height), (x, (y-1)%height)]

@numba.njit
def adhesion_energy(spins,compartments,width,height,x,y,Jt): # for a given lattice point [x,y], calculate how many of its neighbouring points belong to different cells
    # inputs: 2d npArray lattice, int width, int height, int x, int y
    energy = 0

    for nx, ny in neighbors(x,y,width,height):
        # check to make sure we're comparing different cells, or different compartments of the same cell
        if spins[nx, ny] != spins[x,y] or compartments[nx, ny] != compartments[x,y]:
            # if neither of them are backgrounds...
            if spins[nx, ny] != 0 and spins[x,y] != 0:
                # if the connection is between a body and protrusion, instead decrease energy by a lot
                if (compartments[nx,ny]==1 and compartments[x,y]!=1) or (compartments[nx,ny]==1 and compartments[x,y]!=1):
                    energy -= Jt
                # if it is not, this is a cell-cell or protrusion-protrusion interaction, neither of which we care about yet
                else:
                    energy += 1
            # otherwise, do the adhesion between background and other cells
            else:
                energy += 1

    # return the number of neighbours belonging to different cells
    return energy

@numba.njit
def calc_hamiltonian(spins,compartments,width,height,num_cells,area_coeff,a0,adhesion_coeff,protrusion_coeff,p0,Jt,x,y): # calculate the overall Hamiltonian function of the lattice
    # inputs: 2d npArray lattice, 1d npArray cell_id, int width, int height, int num_cells, int area_coeff, int a0, int adhesion_coeff, int protrusion_coeff, int p0, int x, int y
    ham = 0

    # find the total area constraint energy component, based on lambda and a0
    areas, protrusions = find_areas(spins,compartments,width,height,num_cells)
    for i in range(1,num_cells+1):
        ham += area_coeff * ((areas[i]-a0) ** 2)

    # find the total protrusion constraint energy component, based on mu and p0
    for i in range(1,num_cells+1):
        ham += protrusion_coeff * ((protrusions[i]-p0) ** 2)

    # find the adhesion energy around one point, based on alpha
    # this does *not* find the total adhesion energy, as that takes much longer to calculate and we're only interested in the changes of energy when swapping one point
    adhesion = adhesion_energy(spins,compartments,width,height,x,y,Jt)
    print(f"adhesion energy total {adhesion}")
    ham += adhesion_coeff*adhesion

    # return the total Hamiltonian of the system
    return ham



