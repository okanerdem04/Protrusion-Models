# standard libraries
from hmac import new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import seaborn as sns
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian

size = 100
num_cells = 1

lattice = np.zeros((size,size),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.array(range(1,num_cells)) # cell index array

# gen_lattice(lattice,cell_id,size,num_cells) # fill up the lattice array
width = 10
height = 8
a0 = 100
lattice[10:10+width,10:10+height] = 1

new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

all_lattices = np.zeros((10000,size,size),dtype=np.int64)

# loop 10000 times for now, should be adequate for now
sweep = np.prod(new_lattice.shape)
nsweeps = 20
total_sweeps = nsweeps*sweep
temp_lattice = new_lattice
for i in range(total_sweeps):
    # temp_lattice = np.copy(new_lattice)  
    old_ham = calc_hamiltonian(new_lattice, cell_id, size, num_cells, a0, 50, 1) # hamiltonian with preferred size 100

    rand_i = random.randint(0,size-1) # random i, j positions
    rand_j = random.randint(0,size-1)


    original = new_lattice[rand_i,rand_j] 
    inv_i = rand_i
    inv_j = rand_j

    # pick a random neighbouring site to be "invading"
    site_pick = random.randint(1,4)
    if site_pick == 1:
        # loop around the lattice if an edge is hit
        if inv_i + 1 == size: inv_i = 0
        else: inv_i += 1

    elif site_pick == 2:
        if inv_i - 1 == -1: inv_i = size-1
        else: inv_i -= 1

    elif site_pick == 3:
        if inv_j + 1 == size: inv_j = 0
        else: inv_j += 1

    if site_pick == 4:
        if inv_j - 1 == -1: inv_j = size-1
        else: inv_j -= 1

    # check that invading site does not belong to same cell
    if temp_lattice[rand_i][rand_j] != temp_lattice[inv_i][inv_j]:
        # if not, the picked site is replaced by the cell value of the invading site
        temp_lattice[rand_i][rand_j] = temp_lattice[inv_i][inv_j]
        # calculate a new hamiltonian
        new_ham = calc_hamiltonian(temp_lattice, cell_id, size, num_cells, a0, 5, 1) # hamiltonian with preferred size 10, surface tension coeff 0.2

        energy_change = new_ham - old_ham # energy change
        # print(f"energy change {energy_change}")

        # keep the change if energy is lowered or same
        if energy_change <= 0:
            new_lattice = temp_lattice
            # print("lowered energy")
        # else, calc probability to accept change
        else:
            prob = np.exp(-(energy_change)) # assume KbT = 1 (it's definitely not)
            if random.random() <= prob:
                new_lattice = temp_lattice
                # print("random energy increase")

            else:
                # print("no change")
                new_lattice[rand_i,rand_j] = original
    print(i,i//sweep, (new_lattice==1).sum())
    # all_lattices[i] = np.copy(new_lattice)


'''# animated heatmap shamelessly stolen from the internet
def init_heatmap():
    fig, ax = plt.subplots()
    sns.heatmap(all_lattices[0], ax=ax, cbar=False, square=True)
    return fig, ax

def update_heatmap(frame, ax):
    ax.clear()
    sns.heatmap(all_lattices[frame], ax=ax, cbar=False, square=True)
    print(f"anim frame {frame}")

fig, ax = init_heatmap()
anim = matplotlib.animation.FuncAnimation(fig, update_heatmap, frames=10000, fargs=(ax,), interval=30)

anim.save('animated_heatmap.gif', writer='Pillow')'''

sns.heatmap(new_lattice, cbar=False, square=True)
plt.show()

# csv = pd.DataFrame(all_lattices)
# csv.to_csv("testing_20_cells.csv")