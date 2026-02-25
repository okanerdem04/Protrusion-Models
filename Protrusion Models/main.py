# standard libraries
from hmac import new
import numpy as np
import pandas as pd
import dask.dataframe as dd 
import matplotlib.pyplot as plt
import matplotlib.animation
import random
import seaborn as sns
import numba
import timeit
import tracemalloc
import skimage as ski
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors
from protrusion_growth import center_of_mass, find_nearest_protrusion, protrusion_growth, gaussian_2d, calculate_signals

t1 = timeit.default_timer()
tracemalloc.start()
init_memory = tracemalloc.get_traced_memory()


# starting variables:
width = 160             # width of lattice
height = 160            # height of lattice
num_cells = 16           # number of unique cells
target_area = 100       # target area of the body of cells
target_prot = 200       # target area of the protrusions of cells
alpha = 2               # surface tension coefficient
lambd = 1               # area constraint coefficient
mu = 0                  # protrusion constraint coefficient
Jt = -2                  # protrusion-body adhesion coefficient
signal_strength = 100   # strength of signal from protrusion tips
signal_range = 20       # range of signal from protrusion tips

# probability for the likelihood of a cell body being replaced by a protrusion (1 is guaranteed, 0 is never)
protrusion_density = 0.1

# create two 2d arrays to store all lattice info; one array stores each pixel's spin value, the other stores their compartment value
spins = np.zeros((width,height),dtype=np.int16)
compartments = np.zeros((width,height),dtype=np.int16)
# create another 2d array to store signal strengths
signals = np.zeros((width,height),dtype=float)

cell_id = np.array(range(1,num_cells+1)) # cell index array to store all unique spins
# random.shuffle(cell_id) # this is literally just done to get nicer colours

# set up initial cells on the spin array
spins[20:30,20:30]   = cell_id[0]
spins[60:70,20:30]   = cell_id[1]
spins[100:110,20:30] = cell_id[2]
spins[140:150,20:30] = cell_id[3]

spins[20:30,60:70]   = cell_id[4]
spins[60:70,60:70]   = cell_id[5]
spins[100:110,60:70] = cell_id[6]
spins[140:150,60:70] = cell_id[7]

spins[20:30,100:110]   = cell_id[8]
spins[60:70,100:110]   = cell_id[9]
spins[100:110,100:110] = cell_id[10]
spins[140:150,100:110] = cell_id[11]

spins[20:30,140:150]   = cell_id[12]
spins[60:70,140:150]   = cell_id[13]
spins[100:110,140:150] = cell_id[14]
spins[140:150,140:150] = cell_id[15]

# assigns each of the x/y coords that exist on the spin array a compartment value of 1 (body)
for i in range(num_cells):
    compartments[spins == cell_id[i]] = 1

# calculate the value for the total number of sweeps, based on the size of the lattice
sweep = np.prod(spins.shape)
budding_sweeps = 20*sweep
protrusion_sweeps = 100*sweep

# create arrays to store all timesteps for every sweep, this is to allow us to export to a file later
#exp_spins = np.zeros((sweep+1,width,height),dtype=np.int16)
#exp_comps = np.zeros((sweep+1,width,height),dtype=np.int16)

# create a h5py file with datasets for spins and compartments
'''f = h5py.File("test.hdf5", "a")
dsetspins = f.create_dataset("test spins", (budding_sweeps+protrusion_sweeps+1,width,height), dtype=np.int16, compression="gzip")
dsetcomps = f.create_dataset("test compartments", (budding_sweeps+protrusion_sweeps+1,width,height), dtype=np.int16, compression="gzip")'''
# also create two counter variables to keep track of the total timesteps and the timestep for the current sweep
stepsweep = 0
steptot = 0

def run_mc(total_sweeps): # run the Monte Carlo simulation over the total number of sweeps
    # define global variables to count total steps and steps per sweep
    global stepsweep
    global steptot
    global signals

    for i in range(total_sweeps+1):
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
            old_ham = calc_hamiltonian(spins, compartments, signals, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y, Nx, Ny)

            # replace the original selected position with its neighbour, re-calculate hamiltonian
            spins[x,y] = spins[Nx,Ny]
            compartments[x,y] = compartments[Nx,Ny]

            new_ham = calc_hamiltonian(spins, compartments, signals, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y, Nx, Ny)

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
        # check to see if the neighbouring lattice point is a protrusion tip and that the selected point is not a protrusion
        if compartments[Nx,Ny] == 2 and spins[Nx,Ny] != 0 and compartments[x,y] <= 1:
            # find all of the signals for other spins
            signals = calculate_signals(spins,compartments,width,height,spins[Nx,Ny],signal_range,signal_range/2,signal_strength)

            # if the protrusion tip is not in a field (its signal strength is zero) we do nothing
            if signals[Nx,Ny] == 0:
                print("MC step passed because of no field")
            
            # otherwise, do the usual logic of replacing a neighbour and comparing energies
            else:
                # calculate a pre-change hamiltonian
                old_ham = calc_hamiltonian(spins, compartments, signals, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y, Nx, Ny)

                # replace the original selected position with its neighbour, turn the neighbouring point into inactive protrusion, re-calculate hamiltonian
                spins[x,y] = spins[Nx,Ny]
                compartments[x,y] = compartments[Nx,Ny]
                compartments[Nx,Ny] += 1

                # since the tip has moved from [Nx,Ny] to [x,y], we also change Nx,Ny in the calc_hamiltonian function to x,y
                new_ham = calc_hamiltonian(spins, compartments, signals, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, Jt, x, y, x, y)

                # find the energy change
                energy_change = new_ham - old_ham
                print(f"energy_change {energy_change}")

                # if the energy is increased, revert the change with a probability that is proportional to the increase in energy
                if energy_change > 0:
                    prob = np.exp(-(energy_change))
                    if random.random() > prob:
                        spins[x,y] = og_spin
                        compartments[x,y] = og_compartment
                        compartments[Nx,Ny] -= 1
                        print("change reverted")

        '''# save current step to the big 3d array to later be exported
        exp_spins[stepsweep] = np.copy(spins)
        exp_comps[stepsweep] = np.copy(compartments)


        # if our current stepsweep value is exactly at the end of a sweep, we append it to a .npy file
        if stepsweep == width*height:
            t1 = timeit.default_timer()
            dsetspins[steptot-(width*height):steptot-1] = exp_spins[0:width*height-1]
            dsetspins.flush()
            dsetcomps[steptot-(width*height):steptot-1] = exp_comps[0:width*height-1]
            dsetcomps.flush()
            t2 = timeit.default_timer()
            print(f"Time taken to copy into dset is {t2-t1} sec")
            stepsweep = 0

        # increment the total step and sweep step counters
        stepsweep += 1
        steptot += 1'''



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

# run the MC simulation again, this time with protrusion cells in the mix
run_mc(protrusion_sweeps)

# read the snapshot of the other lattice stored in test.hdf5
'''f2 = h5py.File("test.hdf5", "r")
snapshot = f2['test spins'][200000-2]'''


print(f"Starting memory is ",init_memory)
print(tracemalloc.get_traced_memory())
palette = ["#ffffff", "#ffffff", "#ffffff", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555"]

signals = calculate_signals(spins,compartments,width,height,0,signal_range,signal_range/2,signal_strength)

# plot the final lattice
fig, ax = plt.subplots(2,2)
sns.heatmap(spins,square=True,cbar=False,ax=ax[0,0],xticklabels=False, yticklabels=False)
ax[0,0].annotate("Spins of the lattice",(0,0))
ax[0,0].axhline(y = 0, color = '000000', linewidth = 3)
ax[0,0].axhline(y = 200, color = '000000', linewidth = 3)
ax[0,0].axvline(x = 0, color = '000000', linewidth = 3)
ax[0,0].axvline(x = 200, color = '000000', linewidth = 3)
sns.heatmap(compartments,square=True,cbar=False,ax=ax[0,1],xticklabels=False, yticklabels=False)
ax[0,1].annotate("Compartments of the lattice",(0,0))
ax[0,1].axhline(y = 0, color = '000000', linewidth = 3)
ax[0,1].axhline(y = 200, color = '000000', linewidth = 3)
ax[0,1].axvline(x = 0, color = '000000', linewidth = 3)
ax[0,1].axvline(x = 200, color = '000000', linewidth = 3)
sns.heatmap(signals,square=True,cbar=False,ax=ax[1,0],xticklabels=False, yticklabels=False)
ax[1,0].annotate("Signal strength of the lattice",(0,0))
ax[1,0].axhline(y = 0, color = '000000', linewidth = 3)
ax[1,0].axhline(y = 200, color = '000000', linewidth = 3)
ax[1,0].axvline(x = 0, color = '000000', linewidth = 3)
ax[1,0].axvline(x = 200, color = '000000', linewidth = 3)
'''sns.heatmap(spins,square=True,cbar=False,ax=ax[1,1],xticklabels=False, yticklabels=False, vmin=0,vmax=18)
ax[1,1].annotate("d)",(10,20))
ax[1,1].axhline(y = 0, color = '000000', linewidth = 3)
ax[1,1].axhline(y = 200, color = '000000', linewidth = 3)
ax[1,1].axvline(x = 0, color = '000000', linewidth = 3)
ax[1,1].axvline(x = 200, color = '000000', linewidth = 3)'''

plt.tight_layout()
plt.show()