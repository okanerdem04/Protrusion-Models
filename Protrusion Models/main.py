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
from calc_hamiltonian import calc_hamiltonian, neighbors


width = 200
height = 200
num_cells = 1000
target_area = 40
alpha = 2
lambd = 1

lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.array(range(1,num_cells+2)) # cell index array
random.shuffle(cell_id) # this is literally just done to get nicer colours

# fill up the lattice with cells
ind = 1
for i in range(0,25):
    x = i*8
    for j in range(0,40):
        y = j*5
        lattice[x:x+8,y:y+5] = cell_id[ind]
        ind += 1



new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

sweep = np.prod(new_lattice.shape)
nsweeps = 10
total_sweeps = nsweeps*sweep

for i in range(total_sweeps):
    x = random.randint(0,width-1)
    y = random.randint(0,height-1)

    old_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)

    original = new_lattice[x,y] 
    Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)] # selects a random adjacent position

    # check that invading site does not belong to same cell
    if new_lattice[Nx,Ny] != new_lattice[x,y]:
        # if not, the picked site is replaced by the cell value of the invading site
        new_lattice[x,y] = new_lattice[Nx,Ny]
        # calculate a new hamiltonian
        new_ham = calc_hamiltonian(new_lattice, cell_id, width, height, num_cells, target_area, alpha, lambd, x, y)

        energy_change = new_ham - old_ham

        print(f"energy_change {energy_change}")
        

        # if energy is increased, have a chance to replace the swapped cell
        if energy_change > 0:
            prob = np.exp(-(energy_change))
            if random.random() > prob:
                new_lattice[x,y] = original # this reverts changes to the lattice
                print("change reverted")

    print(i,i//sweep, (new_lattice==1).sum())

fig, ax = plt.subplots()
sns.heatmap(new_lattice,cmap=sns.color_palette("hls", num_cells))
plt.show()