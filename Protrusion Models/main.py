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
from protrusion_growth import center_of_mass, find_nearest_protrusion, protrusion_growth

# starting variables:
width = 100             # width of lattice
height = 100            # height of lattice
num_cells = 5           # number of unique cells
target_area = 100       # target area of the body of cells
target_prot = 200        # target area of the protrusions of cells
alpha = 2               # surface tension coefficient
lambd = 1               # area constraint coefficient
mu = 1                  # protrusion constraint coefficient
Jt = 1

# probability for the likelihood of a cell body being replaced by a protrusion (1 is guaranteed, 0 is never)
protrusion_density = 0.1

# create two 2d arrays to store all lattice info; one array stores each pixel's spin value, the other stores their compartment value
spins = np.zeros((width,height),dtype=np.int64)
compartments = np.zeros((width,height),dtype=np.int64)

cell_id = np.array(range(1,num_cells+1)) # cell index array to store all unique spins
# random.shuffle(cell_id) # this is literally just done to get nicer colours

# set up initial cells on the spin array
spins[10:20,10:20] = cell_id[0]
spins[40:50,40:50] = cell_id[1]
spins[80:90,80:90] = cell_id[2]
spins[10:20,80:90] = cell_id[3]
spins[80:90,10:20] = cell_id[4]


# assigns each of the x/y coords that exist on the spin array a compartment value of 1 (body)
for i in range(num_cells):
    compartments[spins == cell_id[i]] = 1

# calculate the value for the total number of sweeps, based on the size of the lattice
sweep = np.prod(spins.shape)
budding_sweeps = 0*sweep
protrusion_sweeps = 100*sweep

def run_mc(total_sweeps): # run the Monte Carlo simulation over the total number of sweeps
    for i in range(total_sweeps):
        # select a random position
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)

        # store that point on the lattice in a temporary variable, also find the index of a random neighbour
        og_spin = spins[x,y] 
        og_compartment = compartments[x,y] 
        Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)]

        # this section of the simulation handles the growth and energy of the cell body
        # check to see if both selected pixels are unique cell bodies (or the background)
        if spins[x,y] != spins[Nx,Ny] and compartments[x,y] <= 1 and compartments[Nx,Ny] <= 1:
            # calculate a pre-change hamiltonian
            old_ham = calc_hamiltonian(spins, compartments, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y)

            # replace the original selected position with its neighbour, re-calculate hamiltonian
            spins[x,y] = spins[Nx,Ny]
            compartments[x,y] = compartments[Nx,Ny]

            new_ham = calc_hamiltonian(spins, compartments, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y)

            # find the energy change
            energy_change = new_ham - old_ham
            print(f"energy_change {energy_change}")

            # if the energy is increased, revert the change with a probability that is proportional to the increase in energy
            if energy_change > 0:
                prob = np.exp(-(energy_change))
                if random.random() > prob:
                    spins[x,y] = og_spin
                    compartments[x,y] = og_compartment
                    print("change reverted")

        # this section of the simulation handles the growth and energy of protrusion tip
        # check to see if the selected lattice point is a protrusion tip
        if compartments[x,y] == 2 and spins[x,y] != 0:
            # calculate the position of the neighbouring cell the protrusion tip wants to grow into
            # # this is either going to be towards another protrusion or away from the center of mass of the cell
            new_protrusion_pos = protrusion_growth(spins,compartments,width,height,x,y,10)
            new_protrusion_x = (new_protrusion_pos[0]+x)%width
            new_protrusion_y = (new_protrusion_pos[1]+y)%height

            print(f"New protrusion at {new_protrusion_x} {new_protrusion_y}")

            # check to see that the selected neighbour is a different spin and not a protrusion
            if spins[new_protrusion_x,new_protrusion_y] != spins[x,y] and compartments[new_protrusion_x,new_protrusion_y] <= 1:
                # find the hamiltonian prior to any changes
                old_ham = calc_hamiltonian(spins, compartments, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y)

                # temporarily store the values of where the new protrusion will be
                temp_spin = spins[new_protrusion_x,new_protrusion_y]
                temp_compartment = compartments[new_protrusion_x,new_protrusion_y]

                # change the selected neighbour to a protrusion tip, replace the original pixel to an inactive protrusion cell
                spins[new_protrusion_x,new_protrusion_y] = spins[x,y]
                compartments[new_protrusion_x,new_protrusion_y] = 2
                compartments[x,y] = 3

                # find the hamiltonian after the swap, calculate energy change
                new_ham = calc_hamiltonian(spins, compartments, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y)
                energy_change = new_ham - old_ham

                # if energy is increased, have a chance to revert protrusion growth
                if energy_change > 0:
                    prob = np.exp(-(energy_change))
                    if random.random() > prob:
                        compartments[x,y] = 2
                        spins[new_protrusion_x,new_protrusion_y] = temp_spin
                        compartments[new_protrusion_x,new_protrusion_y] = temp_compartment
                        print("max protrusion length reached")

        #print(i,i//sweep, (new_lattice==1).sum()) # this one line of code is really slow (probably because it calculates a remainder every single step)
        if i == 300000:
            global snapshot_2
            snapshot_2 = np.copy(spins)
        if i == 600000:
            global snapshot_3
            snapshot_3 = np.copy(spins)

# run the Monte Carlo simulation without any protrusion cells for a certain amount of sweeps
run_mc(budding_sweeps)

# introduce protrusion points to the cells after running MC for a while
for i in range(width):
    for j in range(height):
        if spins[i][j] != 0 and random.random() < protrusion_density and compartments[i][j] == 1:
            buried = 0
            for nx, ny in neighbors(i,j,width,height):
                buried += (spins[nx][ny] != spins[i][j] and compartments[nx][ny] != compartments[i][j])
            if buried != 0:
                compartments[i][j] = 2

# take a snapshot of lattice after this
snapshot_1 = np.copy(spins)

# run the MC simulation again, this time with protrusion cells in the mix
run_mc(protrusion_sweeps)

palette = ["#ffffff", "#ffffff", "#ffffff", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555"]

# plot the final lattice
fig, ax = plt.subplots(2,2)
sns.heatmap(snapshot_1,square=True,cbar=False,ax=ax[0,0],xticklabels=False, yticklabels=False, vmin=0,vmax=18)
ax[0,0].annotate("a)",(10,20))
ax[0,0].axhline(y = 0, color = '000000', linewidth = 3)
ax[0,0].axhline(y = 200, color = '000000', linewidth = 3)
ax[0,0].axvline(x = 0, color = '000000', linewidth = 3)
ax[0,0].axvline(x = 200, color = '000000', linewidth = 3)
sns.heatmap(snapshot_2,square=True,cbar=False,ax=ax[0,1],xticklabels=False, yticklabels=False, vmin=0,vmax=18)
ax[0,1].annotate("b)",(10,20))
ax[0,1].axhline(y = 0, color = '000000', linewidth = 3)
ax[0,1].axhline(y = 200, color = '000000', linewidth = 3)
ax[0,1].axvline(x = 0, color = '000000', linewidth = 3)
ax[0,1].axvline(x = 200, color = '000000', linewidth = 3)
sns.heatmap(snapshot_3,square=True,cbar=False,ax=ax[1,0],xticklabels=False, yticklabels=False, vmin=0,vmax=18)
ax[1,0].annotate("c)",(10,20))
ax[1,0].axhline(y = 0, color = '000000', linewidth = 3)
ax[1,0].axhline(y = 200, color = '000000', linewidth = 3)
ax[1,0].axvline(x = 0, color = '000000', linewidth = 3)
ax[1,0].axvline(x = 200, color = '000000', linewidth = 3)
sns.heatmap(spins,square=True,cbar=False,ax=ax[1,1],xticklabels=False, yticklabels=False, vmin=0,vmax=18)
ax[1,1].annotate("d)",(10,20))
ax[1,1].axhline(y = 0, color = '000000', linewidth = 3)
ax[1,1].axhline(y = 200, color = '000000', linewidth = 3)
ax[1,1].axvline(x = 0, color = '000000', linewidth = 3)
ax[1,1].axvline(x = 200, color = '000000', linewidth = 3)

plt.tight_layout()
plt.show()