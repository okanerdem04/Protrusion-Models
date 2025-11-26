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
num_cells = 1           # number of unique cells
target_area = 100       # target area of the body of cells
target_prot = 10        # target area of the protrusions of cells
alpha = 1               # surface tension coefficient
lambd = 5               # area constraint coefficient
mu = 1                  # protrusion constraint coefficient

# probability for the likelihood of a cell body being replaced by a protrusion (1 is guaranteed, 0 is never)
protrusion_density = 0.2

lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.multiply(np.array(range(1,num_cells+1)),3) # cell index array; this stores the value of the body parts of the cells
# random.shuffle(cell_id) # this is literally just done to get nicer colours

# set up initial conditions
lattice[45:55,45:55] = cell_id[0]
lattice[10,10] = 1

'''lattice[45:55,145:155] = cell_id[1]
lattice[145:155,45:55] = cell_id[2]
lattice[145:155,145:155] = cell_id[3]
lattice[95:105,95:105] = cell_id[4]'''

new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later (TODO: get rid of this, it serves no purpose)

# calculate the value for the total number of sweeps, based on the size of the lattice
sweep = np.prod(new_lattice.shape)
budding_sweeps = 10*sweep
protrusion_sweeps = 50*sweep

def run_mc(total_sweeps): # run the Monte Carlo simulation over the total number of sweeps
    for i in range(total_sweeps):
        # select a random position
        x = random.randint(0,width-1)
        y = random.randint(0,height-1)

        # store that point on the lattice in a temporary variable, also find the index of a random neighbour
        original = new_lattice[x,y] 
        Nx, Ny = neighbors(x,y,width,height)[np.random.randint(0, 4)]

        # this section of the simulation handles the growth and energy of the cell body
        # check to see if both selected pixels are unique cell bodies
        if new_lattice[Nx,Ny]//3 != new_lattice[x,y]//3 and new_lattice[x,y]%3 == new_lattice[Nx,Ny]%3 == 0:
            # create a temporary lattice without any protrusion cells (TODO: make this step unnecessary)
            no_protrusion_lattice = np.copy(new_lattice)
            no_protrusion_lattice[no_protrusion_lattice % 3 != 0] = 0

            # calculate a pre-change hamiltonian
            old_ham = calc_hamiltonian(no_protrusion_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)

            # replace the original selected position with its neighbour, re-calculate hamiltonian
            new_lattice[x,y] = new_lattice[Nx,Ny]
            no_protrusion_lattice[x,y] = no_protrusion_lattice[Nx,Ny]
            new_ham = calc_hamiltonian(no_protrusion_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)

            # find the energy change
            energy_change = new_ham - old_ham
            print(f"energy_change {energy_change}")

            # if the energy is increased, revert the change with a probability that is proportional to the increase in energy
            if energy_change > 0:
                prob = np.exp(-(energy_change))
                if random.random() > prob:
                    new_lattice[x,y] = original
                    print("change reverted")

        # this section of the simulation handles the growth and energy of protrusion tip
        # check to see if the neighbouring lattice point is a protrusion tip
        if new_lattice[Nx,Ny]%3 == 1 and new_lattice[Nx,Ny]//3 != 0:
            # calculate the position of the neighbouring cell the protrusion tip wants to grow into
            # # this is either going to be towards another protrusion or away from the center of mass of the cell
            new_protrusion_pos = protrusion_growth(new_lattice,width,height,Nx,Ny,new_lattice[Nx,Ny],10)
            new_protrusion_x = (new_protrusion_pos[0]+Nx)%width
            new_protrusion_y = (new_protrusion_pos[1]+Ny)%height

            print(f"New protrusion at {new_protrusion_x} {new_protrusion_y}")

            # check to see that the selected neighbour does not already belong to its own cell or any other protrusions
            if new_lattice[new_protrusion_x,new_protrusion_y] // 3 != new_lattice[Nx,Ny]//3 and new_lattice[new_protrusion_x,new_protrusion_y] % 3 == 0:
                # find the hamiltonian prior to any changes
                old_ham = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)

                # temporarily store the value of where the new protrusion will be
                store = new_lattice[new_protrusion_x,new_protrusion_y]

                # change the selected neighbour to a protrusion tip, replace the original pixel to an inactive protrusion cell
                new_lattice[new_protrusion_x,new_protrusion_y] = new_lattice[Nx,Ny]
                new_lattice[Nx,Ny] = new_lattice[Nx,Ny]+1

                # find the hamiltonian after the swap, calculate energy change
                new_ham = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                energy_change = new_ham - old_ham

                # if energy is increased, have a chance to revert protrusion growth
                if energy_change > 0:
                    prob = np.exp(-(energy_change))
                    if random.random() > prob:
                        new_lattice[Nx,Ny] = new_lattice[Nx,Ny]-1
                        new_lattice[new_protrusion_x,new_protrusion_y] = store
                        print("max protrusion length reached")

        # print(i,i//sweep, (new_lattice==1).sum()) # this one line of code is really slow (probably because it calculates a remainder every single step)

# run the Monte Carlo simulation without any protrusion cells for a certain amount of sweeps
run_mc(budding_sweeps)

# introduce protrusion points to the cells after running MC for a while
for i in range(width):
    for j in range(height):
        if new_lattice[i][j] != 0 and random.random() < protrusion_density and new_lattice[i][j] % 3 == 0:
            new_lattice[i][j] = new_lattice[i][j] + 1

# run the MC simulation again, this time with protrusion cells in the mix
run_mc(protrusion_sweeps)

# plot the final lattice
fig, ax = plt.subplots()
sns.heatmap(new_lattice,cmap=sns.color_palette("hls", 6))
plt.show()