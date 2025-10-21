# standard libraries
from hmac import new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import seaborn as sns
import colorcet as cc
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors, find_areas
from find_perimeter import find_perimeter

width = 200
height = 200
num_cells = 1
target_area = 100
alpha = 1
lambd = 1

lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.array(range(1,num_cells+2)) # cell index array
random.shuffle(cell_id) # this is literally just done to get nicer colours

# fill up the lattice with cells
#ind = 1
#for i in range(0,25):
#    x = i*8
#    for j in range(0,40):
#        y = j*5
#        lattice[x:x+8,y:y+5] = cell_id[ind]
#        ind += 1

lattice[100:107,100:107] = 1

new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

sweep = np.prod(new_lattice.shape)
nsweeps = 50
total_sweeps = nsweeps*sweep
n = 0

perimeters = np.zeros((25,total_sweeps),dtype=np.int64)
areas = np.zeros((25,total_sweeps),dtype=np.int64)

alphas = np.array([0.5, 1, 2, 4])
lambdas = np.array([0.5,1,2,4])

for a in range(0,4):
    for b in range(0,4):
        alpha = alphas[a]
        lambd = lambdas[b]
        # store perimeters and areas

        new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

        perimeters[n][0] = find_perimeter(new_lattice,width,height,1) # store the initial perimeter

        areas[n][0] = find_areas(new_lattice,cell_id,width,height,num_cells)[1] # store the initial area

        print("Alpha ",alpha," lambda ",lambd)


        for i in range(total_sweeps):
            x = random.randint(0,width-1)
            y = random.randint(0,height-1)

            original = new_lattice[x,y] 
            Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)] # selects a random adjacent position

            # check that invading site does not belong to same cell
            if new_lattice[Nx,Ny] != new_lattice[x,y]:
                # calculate hamiltonian prior to cell replacement
                old_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)
                # if not, the picked site is replaced by the cell value of the invading site
                new_lattice[x,y] = new_lattice[Nx,Ny]
                # calculate a new hamiltonian
                new_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)

                energy_change = new_ham - old_ham

                #print(f"energy_change {energy_change}")
        

                # if energy is increased, have a chance to replace the swapped cell
                if energy_change > 0:
                    prob = np.exp(-(energy_change))
                    if random.random() > prob:
                        new_lattice[x,y] = original # this reverts changes to the lattice
                        #print("change reverted")
                    else:
                        perimeters[n][i] = find_perimeter(new_lattice,width,height,1)
                        areas[n][i] = find_areas(new_lattice,cell_id,width,height,num_cells)[1] # store the initial area
         
                else:
                    perimeters[n][i] = find_perimeter(new_lattice,width,height,1)
                    areas[n][i] = find_areas(new_lattice,cell_id,width,height,num_cells)[1] # store the initial area

            #print(i,i//sweep, (new_lattice==1).sum())

        def front_fill(arr):
            # front-fills an array of zeros with some elements
            prev = np.arange(len(arr))
            prev[arr == 0] = 0
            prev = np.maximum.accumulate(prev)
            return arr[prev]

        perimeters[n] = front_fill(perimeters[n])

        areas[n] = front_fill(areas[n])

        n += 1


fig, ax = plt.subplots(4,4)

for i in range(16):
    title = f"Alpha {alphas[i//4]} lambda {lambdas[i%4]}"

    ax1 = ax[i%4,i//4]
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Perimeter", color="red")
    ax1.plot(np.array(range(0,total_sweeps)),perimeters[i], color="red")
    ax1.tick_params(axis='y', labelcolor="red")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Area", color="blue")
    ax2.plot(np.array(range(0,total_sweeps)),areas[i], color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

plt.tight_layout()
plt.show()