import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba
from calc_hamiltonian import neighbors

# return True if near a different protrusion cell, False otherwise
def near_protrusion(lattice,width,height,x,y,protrusion_id,d):
    # get the indices of the lattice spaces in range of d, accounting for the looping nature of the lattice
    rows = np.arange(x-d,x+d+1) % width
    cols = np.arange(y-d,y+d+1) % height
    selection = lattice[np.ix_(rows, cols)].copy()

    # erase all of the data for the original protrusion from the selection
    selection[selection//3 == protrusion_id//3] = 0
    # sum up all of the selected lattice spaces which contain a protrusion that isn't the one initially selected
    in_range = np.sum(selection[selection % 3 != 0])

    if in_range > 0:
        return True
    
    return False

    

# run a random walk from a starting point
def run_random_walk(lattice,width,height,start_x,start_y,protrusion_id,d):
    num_steps = 1
    found = False
    starting_movement = 0

    walk_x = start_x
    walk_y = start_y

    # repeat until the random walk reaches the target distance from a protrusion
    while found == False:
        num_steps += 1
        Nx, Ny = neighbors(walk_x,walk_y,width,height)[np.random.randint(0, 4)] # select a random adjacent point

        walk_x = Nx
        walk_y = Ny

        if starting_movement == 0: # store the first step
            starting_movement = [walk_x, walk_y]

        found = near_protrusion(lattice,width,height,walk_x,walk_y,protrusion_id,d) # break the while loop if near a different protrusion

        if num_steps >= 100000:
            print(f"Protrusion not found after 100000 steps")
            return starting_movement, num_steps

    # return the first movement and the number of steps
    print(f"Protrusion found in {num_steps} steps")
    return starting_movement, num_steps
