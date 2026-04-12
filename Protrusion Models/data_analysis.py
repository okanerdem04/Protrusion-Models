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
from calc_hamiltonian import adhesion_energy, calc_hamiltonian, find_areas, neighbors
from protrusion_growth import center_of_mass, find_nearest_protrusion, protrusion_growth, gaussian_2d, calculate_signals


path = "./bash files/report bash/reportdatasigma/"
nums = [0,2,3,4,5,6,7,8,9,10]

#fig, ax = plt.subplots(2,2)

'''for i in range(0,9):
    fname = f'{path}sc_a2_l2_j1_{nums[i]}.out'
    print(fname)
    file = h5py.File(fname,'r')
    lattice = file["test compartments"][119]
    sns.heatmap(lattice,square=True,cbar=False,ax=ax[i//3,i%3],xticklabels=False, yticklabels=False)'''



palette = ["#000000","#ffffff","#bbbbbb","#999999"]

seps = [100,95,90,85,80,75,70,65,60,55,50,45,40,35,30]
sigs = [10,9,8,7,6,5,4,3]
percentage_connections = [0,0,0,0,0,0,0,0]
percentage_densities = [0,0,0,0,0,0,0,0]
percentage_soma = [0,0,0,0,0,0,0,0]
percentage_protrusion = [0,0,0,0,0,0,0,0]

persim_seps = [[0,0,0,0,0]]*15
for i in range(0,15):
    persim_seps[i] = [seps[i]]*5

persim_sigs = [[0,0,0,0,0]]*8
for i in range(0,8):
    persim_sigs[i] = [sigs[i]]*5

persim_connections = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
persim_densities = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
persim_soma = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
persim_protrusion = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]



total_energies = []
'''energy_file = h5py.File(f"energy_odsaavaser_time.out", "a")
dset70 = energy_file.create_dataset("70",dtype=float)
dset65 = energy_file.create_dataset("65",dtype=float)
dset60 = energy_file.create_dataset("60",dtype=float)
dset55 = energy_file.create_dataset("55",dtype=float)
dset50 = energy_file.create_dataset("50",dtype=float)
dset45 = energy_file.create_dataset("45",dtype=float)
dset40 = energy_file.create_dataset("40",dtype=float)
dset35 = energy_file.create_dataset("35",dtype=float)
dset30 = energy_file.create_dataset("30",dtype=float)

energy_file.flush()'''

for a in range(0,8):
    sig = 10-a

    f1 = f'{path}sigma{sig}_1.out'
    f2 = f'{path}sigma{sig}_2.out'
    f3 = f'{path}sigma{sig}_3.out'
    f4 = f'{path}sigma{sig}_4.out'
    f5 = f'{path}sigma{sig}_5.out'
    files = [f1,f2,f3,f4,f5]

    total_tips = [0,0,0,0,0]
    total_connections = [0,0,0,0,0]
    total_background = [0,0,0,0,0]
    total_soma = [0,0,0,0,0]
    total_protrusion = [0,0,0,0,0]

    sim_energies = []
    for f in range(0,5):
        area_energy = 0
        adhesion_energy_c = 0
        signal_energy = 0
            
        file = h5py.File(files[f],'r')
        compartments = file["test compartments"][119]
        spins = file["test spins"][119]
        for i in range(0,220):
            for j in range(0,220):
                if compartments[i,j] == 2:
                    #signal_lattice = calculate_signals(spins, compartments, sep*4, sep*4, spins[i,j], 1, 24, 50)
                    #signal_energy -= signal_lattice[i,j]
                    total_protrusion[f] += 1

                    temp = total_connections[f]
                    total_tips[f] += 1;
                    for nx, ny in neighbors(i,j,220,220):
                        if compartments[nx,ny] == 2 and spins[nx,ny] != spins[i,j]:
                            total_connections[f] = temp + 1

                elif compartments[i,j] == 3:
                    total_protrusion[f] += 1
                
                elif compartments[i,j] == 1:
                    total_soma[f] += 1

                elif spins[i,j] == 0:
                    total_background[f] += 1

                #adhesion_energy_c += 2*(adhesion_energy(spins,compartments,sep*4,sep*4,i,j,-0.5))

        '''areas, dummy = find_areas(spins, compartments, sep*4, sep*4, 16)
        for i in range(1,16):
            area_energy += ((areas[i]-175) ** 2)'''

        #sim_energies.append(area_energy+signal_energy+adhesion_energy_c)

        #print(f"Energy for sep {sep} is {area_energy+signal_energy+adhesion_energy_c}")

        persim_connections[a][f] = (total_connections[f]/total_tips[f])*100
        persim_densities[a][f] = (1-(total_background[f]/(220*220)))*100
        persim_soma[a][f] = ((total_soma[f]/(220*220)))*100
        persim_protrusion[a][f] = ((total_protrusion[f]/(220*220)))*100


    #total_energies.append(sim_energies)

    percent_connections = sum(total_connections)/sum(total_tips)
    print(f"Percentage total connections in sig {sig} is {percent_connections}")
    percent_density = sum(total_background)/(220*220*5)
    print(f"Percentage background in sig {sig} is {percent_density}")
    percent_soma = sum(total_soma)/(220*220*5)
    percent_protrusion = sum(total_protrusion)/(220*220*5)

    percentage_connections[a] = percent_connections*100
    percentage_densities[a] = (1-percent_density)*100
    percentage_soma[a] = (percent_soma)*100
    percentage_protrusion[a] = (percent_protrusion)*100

fig, ax = plt.subplots(2,2)
plt.style.use('seaborn-v0_8-whitegrid')

''' ax[0].plot(sigs,percentage_connections,c="red")
ax[1].stackplot(sigs,percentage_soma,percentage_protrusion,colors=["red","green"],alpha=0.7)
for i in range(0,8):
    ax[0].scatter(persim_sigs[i],persim_connections[i],marker="x",c="blue")
    ax[1].scatter(persim_sigs[i],persim_soma[i],marker="x",c="#990000")
    new_prots = [persim_soma[i][j] + persim_protrusion[i][j] for j in range(len(persim_protrusion[i]))]
    ax[1].scatter(persim_sigs[i],new_prots,marker="x",c="#006600")

ax[0].grid()
ax[1].grid()

ax[0].set_xlabel("Protrusion tip field standard deviation")
ax[1].set_xlabel("Protrusion tip field standard deviation")

ax[0].set_ylabel("% Connected Protrusions", rotation=0, labelpad=50)
ax[1].set_ylabel("% Cell Density", rotation=0, labelpad=20)

ax[0].set_title("a) Percentage of protrusion tips that form connections against signal field standard deviation",loc='center')
ax[1].set_title("b) Percentage of lattice taken up by osteocyte cells against signal field standard deviation",loc='center')'''


sigma3 = h5py.File(f'{path}sigma3_1.out','r')
spins3 = sigma3["test spins"][119]
comps3 = sigma3["test compartments"][119]

sigma6 = h5py.File(f'{path}sigma6_1.out','r')
spins6 = sigma6["test spins"][119]
comps6 = sigma6["test compartments"][119]

sigma8 = h5py.File(f'{path}sigma8_2.out','r')
spins8 = sigma8["test spins"][119]
comps8 = sigma8["test compartments"][119]

sigma10 = h5py.File(f'{path}sigma10_1.out','r')
spins10 = sigma10["test spins"][119]
comps10 = sigma10["test compartments"][119]


sns.heatmap(comps3,square=True,cbar=False,xticklabels=False,yticklabels=False,cmap=palette,ax=ax[0,0])
ax[0,0].set_title("a) Final state for the simulation of σ = 3",loc='center')

sns.heatmap(comps6,square=True,cbar=False,xticklabels=False,yticklabels=False,cmap=palette,ax=ax[0,1])
ax[0,1].set_title("b) Final state for the simulation of σ = 6",loc='center')

sns.heatmap(comps8,square=True,cbar=False,xticklabels=False,yticklabels=False,cmap=palette,ax=ax[1,0])
ax[1,0].set_title("c) Final state for the simulation of σ = 8",loc='center')

sns.heatmap(comps10,square=True,cbar=False,xticklabels=False,yticklabels=False,cmap=palette,ax=ax[1,1])
ax[1,1].set_title("d) Final state for the simulation of σ = 10",loc='center')

'''print(total_energies)

timesteps = [30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220]
for i in range(0,9):
    ax.plot(timesteps,total_energies[i], label=f"{70-(i*5)} pixel separation")

ax.grid()
ax.set_xlabel("Timestep (sweeps)")
ax.set_ylabel("Hamiltonian", rotation=0, labelpad=15)
#sns.heatmap(compartments,square=True,cbar=False,xticklabels=False,yticklabels=False,cmap=palette)
ax.legend(loc='upper right')'''

'''ax[0,0].annotate("a)",(3,350),annotation_clip=False )
ax[0,1].annotate("b)",(3,350),annotation_clip=False )
ax[1,0].annotate("c)",(3,350),annotation_clip=False )
ax[1,1].annotate("d)",(3,350),annotation_clip=False )'''

plt.tight_layout()
plt.show()