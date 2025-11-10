# standard libraries
from hmac import new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import seaborn as sns
import colorcet as cc
import numba
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors
from protrusion_growth import center_of_mass, find_nearest_protrusion


width = 100
height = 100
num_cells = 1
target_area = 100
alpha = 1
lambd = 3

# temperature coefficient for the likelihood of a protrusion point forming (1 is guaranteed, 0 is never)
protrusion_density = 0.2


lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.multiply(np.array(range(1,num_cells+1)),3) # cell index array
random.shuffle(cell_id) # this is literally just done to get nicer colours

lattice[45:55,45:55] = cell_id[0]

lattice[40,20] = 1

new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

sweep = np.prod(new_lattice.shape)
budding_sweeps = 10*sweep
protrusion_sweeps = 50*sweep

def run_mc(total_sweeps):
    for i in range(total_sweeps):
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)

        original = new_lattice[x,y] 
        Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)] # selects a random adjacent position

        # check that invading site does not belong to same cell or a protrusion, and that original site is not a protrusion
        if new_lattice[Nx,Ny]//3 != new_lattice[x,y]//3 and new_lattice[x,y]%3 == new_lattice[Nx,Ny]%3 == 0:
            # create a temporary lattice without any protrusion cells
            no_protrusion_lattice = np.copy(new_lattice)
            no_protrusion_lattice[no_protrusion_lattice % 3 != 0] = 0

            # calculate a pre-change hamiltonian
            old_ham = calc_hamiltonian(no_protrusion_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)

            new_lattice[x,y] = new_lattice[Nx,Ny]
            no_protrusion_lattice[x,y] = no_protrusion_lattice[Nx,Ny]

            # calculate a new hamiltonian
            new_ham = calc_hamiltonian(no_protrusion_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)

            energy_change = new_ham - old_ham

            print(f"energy_change {energy_change}")
        

            # if energy is increased, have a chance to replace the swapped cell
            if energy_change > 0:
                prob = np.exp(-(energy_change))
                if random.random() > prob:
                    new_lattice[x,y] = original # this reverts changes to the lattice
                    print("change reverted")

        # print(i,i//sweep, (new_lattice==1).sum()) # this one line of code is really slow (probably because it calculates a remainder every single step)


run_mc(budding_sweeps)

# introduce protrusion points to the cells
for i in range(width):
    for j in range(height):
        if new_lattice[i][j] != 0 and random.random() < protrusion_density and new_lattice[i][j] % 3 == 0:
            new_lattice[i][j] = new_lattice[i][j] + 1

find_nearest_protrusion(lattice,width,height,50,30,4)

fig, ax = plt.subplots()
sns.heatmap(new_lattice,cmap=sns.color_palette("hls", (num_cells+1)*3))
plt.show()