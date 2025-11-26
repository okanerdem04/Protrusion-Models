import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import numba
import math

from calc_hamiltonian import neighbors

def center_of_mass(lattice,width,height,ind): # find the position of the center of mass of a given cell
    # inputs: 2d npArray lattice, int width, int height, int ind

    x_lattice = np.array([]) # empty arrays to store all x, y positions
    y_lattice = np.array([]) 
    
    # for each pixel in the lattice, append its indices to the x_ and y_lattices if they belong to the given cell ind
    for i in range(width):
        for j in range(height):
            if lattice[i,j] == ind:
                x_lattice = np.append(x_lattice,i)
                y_lattice = np.append(y_lattice,j)

    # calculate the sum of the x indices and y indices, divided by the total number of indices; this finds average value of both indices
    com_x = (np.sum(x_lattice) / np.size(x_lattice))
    com_y = (np.sum(y_lattice) / np.size(y_lattice))

    # return the center of mass indices as a list
    return [com_x, com_y]

def find_nearest_protrusion(lattice,width,height,x,y,protrusion_id): # find the indices of the closest protrusion pixel not belonging to a given id
    # inputs: 2d npArray lattice, int width, int height, int x, int y, int protrusion_id

    x_ind = 0
    y_ind = 0
    d = 9999999

    # apply a mask to the original lattice to create a new lattice where all non-zero values are the indices of protrusions not belonging to the original cell
    masked_lattice = np.zeros((width,height),dtype=int)
    masked_lattice[lattice%3 != 0] = 1
    masked_lattice[lattice//3 == protrusion_id // 3] = 0


    # vectorised alternative
    # if num_ones>0:
    #     _x = indices[0]
    #     _y = indices[1]
    #     _dist =  np.sqrt((_x-x)**2+(_y-y)**2)
    #     _which = _dist.argmin()
    #     assert which == _which,"vectorisation didnt work"

    # find the total number of non-zero indices and each of their x- and y-positions
    num_ones = np.count_nonzero(masked_lattice)
    indices = np.nonzero(masked_lattice)

    # go through each of the indices, find the distance between them and the given x,y point
    for i in range(num_ones):
        xn = indices[0][i]
        yn = indices[1][i]
        print(xn)

        dist = math.sqrt((xn-x)**2+(yn-y)**2)

        # store the closest point's distance and indices, overwriting them if a closer point is found
        if dist < d:
            d = dist
            x_ind = xn
            y_ind = yn
            # which = i

    print(f"Closest point to x {x} y {y} is at {x_ind} {y_ind}")

    # return the *relative* position of the nearest protrusion point and its distance
    nearest_x = x_ind - x
    nearest_y = y_ind - y
    return [nearest_x,nearest_y,d] # note that if there are no other protrusion points, this returns [0,0,9999999]; large distance should prevent any errors from this

def protrusion_growth(lattice,width,height,x,y,protrusion_id,d): # model the growth of a protrusion point
    # inputs: 2d npArray lattice, int width, int height, int x, int y, int protrusion_id, int d

    # find the vector from the center of mass of the cell to the protrusion point
    com_pos = center_of_mass(lattice,width,height,protrusion_id-1) # protrusion_id - 1 is its respective body
    rel_com_x = x - com_pos[0]
    rel_com_y = y - com_pos[1]

    # find the relative position of the nearest protrusion point
    pro_pos = find_nearest_protrusion(lattice,width,height,x,y,protrusion_id)
    rel_pro_x = pro_pos[0]
    rel_pro_y = pro_pos[1]

    '''print(rel_com_x)
    print(rel_com_y)
    print(rel_pro_x)
    print(rel_pro_y)'''

    # if the nearest protrusion is within distance d of the protrusion point, attempt to grow towards it
    if pro_pos[2] <= d:
        # TODO: make this growth far more organic
        if abs(rel_pro_x) >= abs(rel_pro_y):
            return [round(rel_pro_x/abs(rel_pro_x)),0]
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