# standard libraries
from hmac import new
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors, find_areas
from find_perimeter import find_perimeter
from center_of_mass import center_of_mass

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

lattice[100:110,100:110] = 1

new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later

sweep = np.prod(new_lattice.shape)
nsweeps = 50
total_sweeps = nsweeps*sweep

# store perimeters, areas and centers of mass
perimeters = np.zeros(total_sweeps,dtype=np.int64)
perimeters[0] = find_perimeter(new_lattice,width,height,1) # store the initial perimeter

areas = np.zeros(total_sweeps,dtype=np.int64)
areas[0] = find_areas(new_lattice,cell_id,width,height,num_cells)[1] # store the initial area

centers_of_mass = np.zeros((total_sweeps,2)) # x and y for center of mass means storing as a 2d array
centers_of_mass[0] = center_of_mass(new_lattice,width,height,1) # store the initial perimeter

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
                perimeters[i] = find_perimeter(new_lattice,width,height,1)
                areas[i] = find_areas(new_lattice,cell_id,width,height,num_cells)[1]
                centers_of_mass[i] = center_of_mass(new_lattice,width,height,1)
                print(i)
         
        else:
            perimeters[i] = find_perimeter(new_lattice,width,height,1)
            areas[i] = find_areas(new_lattice,cell_id,width,height,num_cells)[1]
            centers_of_mass[i] = center_of_mass(new_lattice,width,height,1)

    #print(i,i//sweep, (new_lattice==1).sum())

def front_fill(arr):
    # front-fills an array of zeros with some elements
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

full_perimeters = front_fill(perimeters)

full_areas = front_fill(areas)

# we cannot front-fill a 2d array with our function; split the center of mass array into two 1d arrays
com_x = front_fill(centers_of_mass[:,0])
com_y = front_fill(centers_of_mass[:,1])

# calculate the distance from the first center of mass
def calc_distance(com_x, com_y):
    # store the initial positions
    start_x = com_x[0]
    start_y = com_y[0]

    # array for the distances per timestep
    distance = np.zeros(total_sweeps)
    
    for i in range(0, total_sweeps):
        delta_x = com_x[i] - start_x # difference in x, y
        delta_y = com_y[i] - start_y

        delta = math.sqrt((delta_x ** 2) + (delta_y ** 2)) # pythagoras innit
        distance[i] = delta

    return distance

distances = calc_distance(com_x, com_y)

fig, (ax1,ax3) = plt.subplots(1,2)

ax1.set_title("x/y position over time")
ax1.set_xlabel("Timestep")
ax1.set_ylabel("Distance from start", color="red")
ax1.plot(np.array(range(0,total_sweeps)),distances, color="red")
ax1.tick_params(axis='y', labelcolor="red")

#ax2 = ax1.twiny()
#ax2.plot(np.array(range(0,total_sweeps)),com_y, color="blue")
#ax2.tick_params(axis='y', labelcolor="blue")

sns.heatmap(new_lattice,cmap=sns.color_palette("hls", 2), square=True, ax=ax3)

plt.show()