import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba
import math

from calc_hamiltonian import neighbors

# find the center of mass of a cell
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
        elif d < width and d < height:
            d += 1
        else:
            return [0,0]
    
    # for now, get the indices of a non-zero item in this lattice area - this technically may not be the closest one
    nearest_site = np.nonzero(search_area)

    nearest_x = nearest_site[0]-d
    nearest_y = nearest_site[1]-d

    print(f"nearest protrusion site x {nearest_x[0]} y {nearest_y[0]}, distance {d}")
    # note this returns the relative position of the nearest protrusion, not its actual position
    return [nearest_x[0],nearest_y[0],d]

# model the growth of a protrusion point
def protrusion_growth(lattice,width,height,x,y,protrusion_id,d):
    # find the vector from the center of mass of the cell to the protrusion point
    com_pos = center_of_mass(lattice,width,height,protrusion_id-1) # protrusion_id - 1 is its respective body
    rel_com_x = x - com_pos[0]
    rel_com_y = y - com_pos[1]

    # find the relative position of the nearest protrusion point
    pro_pos = find_nearest_protrusion(lattice,width,height,x,y,protrusion_id)
    rel_pro_x = pro_pos[0]
    rel_pro_y = pro_pos[1]

    print(rel_com_x)
    print(rel_com_y)
    print(rel_pro_x)
    print(rel_pro_y)

    # if the nearest protrusion is within distance d of the protrusion point, attempt to grow towards it
    if pro_pos[2] <= d:
        # if it's further away on the x axis than y in either direction...
        if abs(rel_pro_x) >= abs(rel_pro_y):
            # return an x/y coordinate that is one closer to the protrusion
            return [round(rel_pro_x/abs(rel_pro_x)),0] # dividing by the absolute value gives a normalised value, i.e. 1 or -1 depending on direction
        else:
            return [0,round(rel_pro_y/abs(rel_pro_y))]

    # if the nearest protrusion is too far away, instead grow away from the center of mass
    else:
        current_distance = math.sqrt(rel_com_x**2 + rel_com_y**2)
        # find the distances of the four adjacent spots
        d1 = math.sqrt((rel_com_x+1)**2 + rel_com_y**2)
        d2 = math.sqrt((rel_com_x-1)**2 + rel_com_y**2)
        d3 = math.sqrt(rel_com_x**2 + (rel_com_y+1)**2)
        d4 = math.sqrt(rel_com_x**2 + (rel_com_y-1)**2)
        new_distance = np.array([d1,d2,d3,d4])

        best_distance = np.sort(new_distance)[-1] # get the two best directions
        second_best = np.sort(new_distance)[-2]

        # find the ratio of these; this ratio should always be >= 1
        ratio = best_distance / second_best

        # when ratio is equal to 1, we want a 50% chance of either, decreasing as ratio increases; use a negative exponential
        prob = np.exp(-((ratio-1)*10))/2 # this is equal to 1/2e^(-10x) which is 0.5 at x=0 (ratio=1) and ~0.1 at x=0.16(ratio=1.16)

        if random.random() > prob:
            idx = int(np.flatnonzero(new_distance == best_distance)[0])
            match idx: # returns the direction of movement that maximises the total distance from com
                case 0: return [1,0]
                case 1: return [-1,0]
                case 2: return [0,1]
                case 3: return [0,-1]

                case _: return [0,0]

        else:
            idx = int(np.flatnonzero(new_distance == second_best)[0])
            match idx: # returns the direction of movement that is second best if random probability allows
                case 0: return [1,0]
                case 1: return [-1,0]
                case 2: return [0,1]
                case 3: return [0,-1]

                case _: return [0,0]