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
import h5py
import sys
import time
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors
from protrusion_growth import center_of_mass, find_nearest_protrusion, protrusion_growth, gaussian_2d, calculate_signals

t1 = timeit.default_timer()
tracemalloc.start()
init_memory = tracemalloc.get_traced_memory()



# starting variables:
width = 1000             # width of lattice
height = 1000            # height of lattice
num_cells = 400           # number of unique cells
target_area = 100       # target area of the body of cells
target_prot = 200       # target area of the protrusions of cells
alpha = 2               # surface tension coefficient
lambd = 1               # area constraint coefficient
mu = 0                  # protrusion constraint coefficient
Jt = -3                 # protrusion-body adhesion coefficient
signal_strength = 50   # strength of signal from protrusion tips
signal_range = 20       # range of signal from protrusion tips

if(len(sys.argv) > 1): # handle input from command line
    target_area = int(sys.argv[1])
    alpha = int(sys.argv[2])
    lambd = int(sys.argv[3])
    signal_strength = int(sys.argv[4])
    signal_range = int(sys.argv[5])

# create two 2d arrays to store all lattice info; one array stores each pixel's spin value, the other stores their compartment value
spins = np.zeros((width,height),dtype=np.int16)
compartments = np.zeros((width,height),dtype=np.int16)
# create another 2d array to store signal strengths
signals = np.zeros((width,height),dtype=float)

cell_id = np.array(range(1,num_cells+1)) # cell index array to store all unique spins
# random.shuffle(cell_id) # this is literally just done to get nicer colours

# set up initial cells on the spin array
for i in range(20):
    for j in range(20):
        spins[(i*50+20):(i*50+30),(j*50+20):(j*50+30)] = cell_id[i*20+j]

# assigns each of the x/y coords that exist on the spin array a compartment value of 1 (body)
for i in range(num_cells):
    compartments[spins == cell_id[i]] = 1

# calculate the value for the total number of sweeps, based on the size of the lattice
sweep = np.prod(spins.shape)
budding_sweeps = 20*sweep
protrusion_sweeps = 100*sweep
if(len(sys.argv) > 6):
    protrusion_sweeps = int(sys.argv[6])*sweep

# create arrays to store all timesteps for every sweep, this is to allow us to export to a file later
#exp_spins = np.zeros((sweep+1,width,height),dtype=np.int16)
#exp_comps = np.zeros((sweep+1,width,height),dtype=np.int16)

# create a h5py file with datasets for spins and compartments
if(len(sys.argv) > 7):
    f = h5py.File(sys.argv[7], "a")
else:
    f = h5py.File(f"",time.time(),".out", "a")

dsetspins = f.create_dataset("test spins", ((budding_sweeps/sweep)+(protrusion_sweeps/sweep)+1,width,height), dtype=np.int16, compression="gzip")
dsetcomps = f.create_dataset("test compartments", ((budding_sweeps/sweep)+(protrusion_sweeps/sweep)+1,width,height), dtype=np.int16, compression="gzip")
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

            # if the energy is increased, revert the change with a probability that is proportional to the increase in energy
            if energy_change > 0:
                prob = np.exp(-(energy_change))
                if random.random() > prob:
                    spins[x,y] = og_spin
                    compartments[x,y] = og_compartment

        # this section of the simulation handles the growth and energy of protrusion tip
        # check to see if the neighbouring lattice point is a protrusion tip and that the selected point is not a protrusion
        if compartments[Nx,Ny] == 2 and spins[Nx,Ny] != 0 and compartments[x,y] <= 1:
            # find all of the signals for other spins
            signals = calculate_signals(spins,compartments,width,height,spins[Nx,Ny],signal_range,signal_range/2,signal_strength)

            # if the protrusion tip is not in a field (its signal strength is zero) we do nothing
            if signals[Nx,Ny] == 0:
                pass
            
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

                # if the energy is increased, revert the change with a probability that is proportional to the increase in energy
                if energy_change > 0:
                    prob = np.exp(-(energy_change))
                    if random.random() > prob:
                        spins[x,y] = og_spin
                        compartments[x,y] = og_compartment
                        compartments[Nx,Ny] -= 1

        # if our current steptot value is exactly at the end of a sweep, we append the current spins, compartments to the file
        if steptot == sweep:
            dsetspins[stepsweep] = spins
            dsetspins.flush()
            dsetcomps[stepsweep] = compartments
            dsetcomps.flush()
            print(f"Reached the final step of sweep ",stepsweep)
            stepsweep += 1
            steptot = 0

        # increment the total step and sweep step counters
        steptot += 1



# run the Monte Carlo simulation without any protrusion cells for a certain amount of sweeps
run_mc(budding_sweeps)

# introduce protrusion tips based on the contours of cells
contours = ski.measure.find_contours(compartments)
for contour in contours:
    # set a number of tips with slight randomness
    num_tips = 5
    num_tips = round(random.gauss(num_tips,1))

    # get evenly spaced indices for the number of protrusion tips
    length = len(contour[:,0])
    ix = np.linspace(0,length-1,num=num_tips+1,dtype=int)
    ix = np.delete(ix,len(ix)-1) # we remove the final element of this array since it gives an equal position to the first

    for i in ix:
        i = round(random.gauss(i,2.5)) # introduce a slight amount of noise to the position
        x = round(contour[i,0])
        y = round(contour[i,1])
        compartments[x,y] = 2
        # if the selected position currently doesn't belong to a cell, set its spin to the highest value of its neighbours
        if spins[x,y] == 0:
            for nx, ny in neighbors(x,y,width,height):
                if spins[nx,ny] > spins[x,y]:
                    spins[x,y] = spins[nx,ny]

# run the MC simulation again, this time with protrusion cells in the mix
run_mc(protrusion_sweeps)

# read the snapshot of the other lattice stored in test.hdf5
'''f2 = h5py.File("test.hdf5", "r")
snapshot = f2['test spins'][200000-2]'''

palette = ["#ffffff", "#ffffff", "#ffffff", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555", "#000000", "#999999", "#555555"]

signals = calculate_signals(spins,compartments,width,height,0,signal_range,signal_range/2,signal_strength)

# plot the final lattice
# fig, ax = plt.subplots(2,2)
# sns.heatmap(spins,square=True,cbar=False,ax=ax[0,0])
# ax[0,0].annotate("Spins of the lattice",(0,0))
# ax[0,0].axhline(y = 0, color = '000000', linewidth = 3)
# ax[0,0].axhline(y = 200, color = '000000', linewidth = 3)
# ax[0,0].axvline(x = 0, color = '000000', linewidth = 3)
# ax[0,0].axvline(x = 200, color = '000000', linewidth = 3)
# sns.heatmap(compartments,square=True,cbar=False,ax=ax[1,1],xticklabels=False, yticklabels=False)
# ax[1,1].annotate("Compartments of the lattice",(0,0))
# ax[1,1].axhline(y = 0, color = '000000', linewidth = 3)
# ax[1,1].axhline(y = 200, color = '000000', linewidth = 3)
# ax[1,1].axvline(x = 0, color = '000000', linewidth = 3)
# ax[1,1].axvline(x = 200, color = '000000', linewidth = 3)
# sns.heatmap(signals,square=True,cbar=False,ax=ax[1,0],xticklabels=False, yticklabels=False)
# ax[1,0].annotate("Signal strength of the lattice",(0,0))
# ax[1,0].axhline(y = 0, color = '000000', linewidth = 3)
# ax[1,0].axhline(y = 200, color = '000000', linewidth = 3)
# ax[1,0].axvline(x = 0, color = '000000', linewidth = 3)
# ax[1,0].axvline(x = 200, color = '000000', linewidth = 3)
# sns.heatmap(points,square=True,cbar=True,ax=ax[0,1],xticklabels=False, yticklabels=False)
# for contour in contours:
#     ax[0,1].plot(contour[:,1],contour[:,0])
# ax[0,1].set_xlim([0,200])
# ax[0,1].set_ylim([200,0])
# ax[0,1].annotate("Filtered image",(0,0))
# ax[0,1].set_box_aspect(1)
# ax[0,1].axhline(y = 0, color = '000000', linewidth = 3)
# ax[0,1].axhline(y = 200, color = '000000', linewidth = 3)
# ax[0,1].axvline(x = 0, color = '000000', linewidth = 3)
# ax[0,1].axvline(x = 200, color = '000000', linewidth = 3)

# plt.tight_layout()
# plt.show()