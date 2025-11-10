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

# finds the nearest other protrusion's position
def find_nearest_protrusion(lattice,width,height,x,y,protrusion_id):
    found = False
    d = 1
    search_area = lattice[x-1:x+1,y-1:y+1].copy()

    while found == False:
        # get the indices of the lattice spaces in range of d, accounting for the looping nature of the lattice
        rows = np.arange(x-d,x+d+1) % width
        cols = np.arange(y-d,y+d+1) % height
        selection = lattice[np.ix_(rows, cols)].copy()

        # erase all of the data for the original protrusion from the selection
        selection[selection//3 == protrusion_id//3] = 0
        # sum up all of the selected lattice spaces which contain a protrusion that isn't the one initially selected
        in_range = np.count_nonzero(selection % 3 != 0)

        if in_range > 0:
            search_area = selection.copy()
            found = True
        else:
            d += 1
    
    # for now, get the indices of a non-zero item in this lattice area - this technically may not be the closest one
    print(search_area)

    nearest_site = np.nonzero(search_area)
    # we need to translate the relative frame of reference here to the one of the entire lattice

    nearest_x = x + nearest_site[0]-d
    nearest_y = y + nearest_site[1]-d

    print(f"nearest protrusion site x {nearest_x} y {nearest_y}, distance {d}")

    return [nearest_x,nearest_y,d]

