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


width = 50
height = 50
num_cells = 1
target_area = 40
target_protrusions = 10
alpha = 1.8
lambd = 1
lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.multiply(np.array(range(0,num_cells+1)),2) # cell index array - store cells as even numbers
protrusion_id = np.copy(cell_id) + 1 # protrusion index array - store protrusions as odd numbers

lattice[20:28,20:25] = 2
lattice[19,24] = 3


new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

sweep = np.prod(new_lattice.shape)
nsweeps = 50
total_sweeps = nsweeps*sweep


for i in range(total_sweeps):
    x = random.randint(0,width-1)
    y = random.randint(0,height-1)

    old_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y, target_protrusions)

    original = new_lattice[x,y] 
    Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)] # selects a random adjacent position

    # check that invading site does not belong to same cell
    if (new_lattice[Nx,Ny]//2) != (new_lattice[x,y]//2):
        # if not, the picked site is replaced by the cell value of the invading site
        new_lattice[x,y] = new_lattice[Nx,Ny]
        # calculate a new hamiltonian
        new_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y, target_protrusions)

        energy_change = new_ham - old_ham

        print(f"energy_change {energy_change}")
         

        # if energy is increased, have a chance to replace the swapped cell
        if energy_change > 0:
            prob = np.exp(-(energy_change))
            if random.random() > prob:
                new_lattice[x,y] = original # this reverts changes to the lattice
                print("change reverted")

    print(i,i//sweep, (new_lattice==2).sum(), (new_lattice==3).sum())

fig, ax = plt.subplots()
sns.heatmap(new_lattice,cmap=sns.color_palette("hls", 4))
plt.show()