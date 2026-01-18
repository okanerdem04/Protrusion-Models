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
import scienceplots
from scipy.ndimage import label
from skimage.measure import find_contours
from matplotlib.patches import Polygon
# homemade functions
from gen_lattice import gen_lattice
from calc_hamiltonian import calc_hamiltonian, neighbors, find_protrusions, find_areas
from protrusion_growth import center_of_mass, find_nearest_protrusion, protrusion_growth

# starting variables:
width = 50             # width of lattice
height = 50            # height of lattice
num_cells = 1         # number of unique cells
target_area = 200        # target area of the body of cells
target_prot = 400       # target area of the protrusions of cells
alpha = 1               # surface tension coefficient
lambd = 1               # area constraint coefficient
mu = 1                  # protrusion constraint coefficient

# probability for the likelihood of a cell body being replaced by a protrusion (1 is guaranteed, 0 is never)
protrusion_density = 0.1

lattice = np.zeros((width,height),dtype=np.int64) # lattice is stored as a 2d numpy array
cell_id = np.multiply(np.array(range(1,num_cells+1)),3) # cell index array; this stores the value of the body parts of the cells
random.shuffle(cell_id) # this is literally just done to get nicer colours

# set up initial conditions
lattice[20:30,20:30] = cell_id[0]

'''lattice[45:55,145:155] = cell_id[1]
lattice[145:155,45:55] = cell_id[2]
lattice[145:155,145:155] = cell_id[3]
lattice[95:105,95:105] = cell_id[4]'''

# store areas, hams
area_200 = np.zeros(1250,dtype=np.int64)
area_300 = np.zeros(1250,dtype=np.int64)
area_400 = np.zeros(1250,dtype=np.int64)
area_500 = np.zeros(1250,dtype=np.int64)
area_600 = np.zeros(1250,dtype=np.int64)
area_700 = np.zeros(1250,dtype=np.int64)
area_800 = np.zeros(1250,dtype=np.int64)
area_900 = np.zeros(1250,dtype=np.int64)
area_1000 = np.zeros(1250,dtype=np.int64)

ham_200 = np.zeros(1250,dtype=float)
ham_300 = np.zeros(1250,dtype=float)
ham_400 = np.zeros(1250,dtype=float)
ham_500 = np.zeros(1250,dtype=float)
ham_600 = np.zeros(1250,dtype=float)
ham_700 = np.zeros(1250,dtype=float)
ham_800 = np.zeros(1250,dtype=float)
ham_900 = np.zeros(1250,dtype=float)
ham_1000 = np.zeros(1250,dtype=float)


timesteps = np.zeros(1250,dtype=float)

for k in range(9):
    new_lattice = np.copy(lattice) # duplicate the old lattice, this allows a comparison later (TODO: get rid of this, it serves no purpose)

    if k == 1:
        target_area = 300
    if k == 2:
        target_area = 400
    if k == 3:
        target_area = 500
    if k == 4:
        target_area = 600
    if k == 5:
        target_area = 700
    if k == 6:
        target_area = 800
    if k == 7:
        target_area = 900
    if k == 8:
        target_area = 1000


    # calculate the value for the total number of sweeps, based on the size of the lattice
    sweep = np.prod(new_lattice.shape)
    budding_sweeps = 50*sweep
    protrusion_sweeps = 100*sweep

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
            # check to see if the selected lattice point is a protrusion tip
            if new_lattice[x,y]%3 == 1 and new_lattice[x,y]//3 != 0:
                # calculate the position of the neighbouring cell the protrusion tip wants to grow into
                # # this is either going to be towards another protrusion or away from the center of mass of the cell
                new_protrusion_pos = protrusion_growth(new_lattice,width,height,x,y,new_lattice[x,y],10)
                new_protrusion_x = (new_protrusion_pos[0]+x)%width
                new_protrusion_y = (new_protrusion_pos[1]+y)%height

                print(f"New protrusion at {new_protrusion_x} {new_protrusion_y}")

                # check to see that the selected neighbour does not already belong to its own cell or any other protrusions
                if new_lattice[new_protrusion_x,new_protrusion_y] // 3 != new_lattice[x,y]//3 and new_lattice[new_protrusion_x,new_protrusion_y] % 3 == 0:
                    # find the hamiltonian prior to any changes
                    old_ham = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)

                    # temporarily store the value of where the new protrusion will be
                    store = new_lattice[new_protrusion_x,new_protrusion_y]

                    # change the selected neighbour to a protrusion tip, replace the original pixel to an inactive protrusion cell
                    new_lattice[new_protrusion_x,new_protrusion_y] = new_lattice[x,y]
                    new_lattice[x,y] = new_lattice[x,y]+1

                    # find the hamiltonian after the swap, calculate energy change
                    new_ham = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                    energy_change = new_ham - old_ham

                    # if energy is increased, have a chance to revert protrusion growth
                    if energy_change > 0:
                        prob = np.exp(-(energy_change))
                        if random.random() > prob:
                            new_lattice[x,y] = new_lattice[x,y]-1
                            new_lattice[new_protrusion_x,new_protrusion_y] = store
                        
            # stop the Monte Carlo simulation as soon as we reach the limit of protrusions
            num_protrusions = find_protrusions(new_lattice,width,height,num_cells)[1]
            if num_protrusions >= target_prot:
                print(f"Total number of inactive protrusions reached in {i/sweep} sweeps")
                break

            if i % 100 == 0:
                if k == 0:
                    area_200[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_200[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 1:
                    area_300[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_300[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 2:
                    area_400[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_400[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 3:
                    area_500[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_500[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 4:
                    area_600[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_600[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 5:
                    area_700[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_700[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 6:
                    area_800[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_800[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 7:
                    area_900[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_900[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)
                elif k == 8:
                    area_1000[i//100] = find_areas(new_lattice,width,height,num_cells)[1]
                    ham_1000[i//100] = calc_hamiltonian(new_lattice, width, height, num_cells, lambd, target_area, alpha, mu, target_prot, x, y)

                timesteps[i//100] = i/2500

            # print(i,i//sweep, (new_lattice==1).sum()) # this one line of code is really slow (probably because it calculates a remainder every single step)

    # run the Monte Carlo simulation without any protrusion cells for a certain amount of sweeps
    run_mc(budding_sweeps)

    # introduce protrusion points to the cells after running MC for a while
    '''for i in range(width):
        for j in range(height):
            if new_lattice[i][j] != 0 and new_lattice[i][j] % 3 == 0:
                # we only want to apply protrusion points to the border of the cell, i.e. where its neighbours
                outside = 0
                for nx, ny in neighbors(i,j,width,height):
                    outside += (new_lattice[nx, ny] != new_lattice[i,j])
                if random.random() < protrusion_density and outside >= 1:
                    new_lattice[i][j] = new_lattice[i][j] + 1'''

    # run the MC simulation again, this time with protrusion cells in the mix
    #run_mc(protrusion_sweeps)

# plot the final lattice
#fig, ax0 = plt.subplots(2,2)
#sns.heatmap(new_lattice,cmap=sns.color_palette(palette='Greys'),cbar=False,square=True)

#plt.style.use('science')
#plt.rcParams['text.usetex'] = True

'''ax0[0,0].plot(timesteps,area_200,color='blue')
ax0[0,0].set_title(r"$A_o$ = 200")
ax0[0,0].set_box_aspect(1)
ax0[0,0].set_xlabel("Time (sweeps)")
ax0[0,0].set_ylabel(r"a", color='blue')
ax0[0,0].tick_params(axis='y', labelcolor='blue')
ax2 = ax0[0,0].twinx() 
ax2.set_box_aspect(1)
ax2.plot(timesteps,ham_200,color='red')
ax2.set_ylabel(r'H', color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax0[0,1].plot(timesteps,area_300,color='blue')
ax0[0,1].set_title(r"$A_o$ = 300")
ax0[0,1].set_box_aspect(1)
ax0[0,1].set_xlabel("Time (sweeps)")
ax0[0,1].set_ylabel(r"a", color='blue')
ax0[0,1].tick_params(axis='y', labelcolor='blue')
ax3 = ax0[0,1].twinx() 
ax3.set_box_aspect(1)
ax3.plot(timesteps,ham_300,color='red')
ax3.set_ylabel(r'H', color='red')
ax3.tick_params(axis='y', labelcolor='red')

ax0[1,0].plot(timesteps,area_400,color='blue')
ax0[1,0].set_title(r"$A_o$ = 400")
ax0[1,0].set_box_aspect(1)
ax0[1,0].set_xlabel("Time (sweeps)")
ax0[1,0].set_ylabel(r"a", color='blue')
ax0[1,0].tick_params(axis='y', labelcolor='blue')
ax4 = ax0[1,0].twinx() 
ax4.set_box_aspect(1)
ax4.plot(timesteps,ham_400,color='red')
ax4.set_ylabel(r'H', color='red')
ax4.tick_params(axis='y', labelcolor='red')

ax0[1,1].plot(timesteps,area_500,color='blue')
ax0[1,1].set_title(r"$A_o$ = 500")
ax0[1,1].set_box_aspect(1)
ax0[1,1].set_xlabel("Time (sweeps)")
ax0[1,1].set_ylabel(r"a", color='blue')
ax0[1,1].tick_params(axis='y', labelcolor='blue')
ax5 = ax0[1,1].twinx() 
ax5.set_box_aspect(1)
ax5.plot(timesteps,ham_500,color='red')
ax5.set_ylabel(r'H', color='red')
ax5.tick_params(axis='y', labelcolor='red')'''


fig, axn = plt.subplots()

plt.tight_layout()

lowest = np.array([0,0,0,0,0,0,0,0,0])
lowest[0] = np.argwhere(area_200 == 200)[0]/25
lowest[1] = np.argwhere(area_300 == 300)[0]/25
lowest[2] = np.argwhere(area_400 == 400)[0]/25
lowest[3] = np.argwhere(area_500 == 500)[0]/25
lowest[4] = np.argwhere(area_600 == 600)[0]/25
lowest[5] = np.argwhere(area_700 == 700)[0]/25
lowest[6] = np.argwhere(area_800 == 800)[0]/25
lowest[7] = np.argwhere(area_900 == 900)[0]/25
lowest[8] = np.argwhere(area_1000 == 1000)[0]/25

model1 = np.poly1d(np.polyfit([200,300,400,500,600,700,800,900,1000],lowest, 1))
polyline = np.linspace(200, 1000, 50)
axn.plot(polyline,model1(polyline),color='red',linestyle='--')
axn.scatter([200,300,400,500,600,700,800,900,1000],lowest,marker='x')
axn.set_xlabel("Target area")
axn.set_ylabel("Stationarity time (sweeps)")


# This entire block of code from here onwards is AI generated, it is solely for data visualisation
# (creating a heatmap where each region of data is given an outline)
def blocky_polygons(mask):
    """Return polygons that follow the grid edges (no diagonals).

    This builds explicit grid-edge segments for each True cell, removes
    internal edges (shared by two cells) and then chains the remaining
    outer edges into closed polygon rings.
    """
    from collections import defaultdict

    polygons = []
    labeled, num = label(mask)

    for region in range(1, num + 1):
        region_mask = (labeled == region)
        ys, xs = np.where(region_mask)

        # collect all edges as directed undirected tuples of corner points
        edges = []
        for y, x in zip(ys, xs):
            # corners: (x, y), (x+1, y), (x+1, y+1), (x, y+1)
            p0 = (x, y)
            p1 = (x+1, y)
            p2 = (x+1, y+1)
            p3 = (x, y+1)
            edges.extend([
                (p0, p1),
                (p1, p2),
                (p2, p3),
                (p3, p0),
            ])

        # count edges and keep only outer edges (appear once)
        edge_count = {}
        for a, b in edges:
            key = (a, b) if a <= b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

        outer_edges = [e for e, cnt in edge_count.items() if cnt == 1]
        if not outer_edges:
            continue

        # build adjacency map from outer edges
        adj = defaultdict(list)
        for a, b in outer_edges:
            adj[a].append(b)
            adj[b].append(a)

        # traverse adjacency to build closed chains (polygons)
        while adj:
            start = next(iter(adj))
            chain = [start]
            current = start
            prev = None

            while True:
                nbrs = adj.get(current, [])
                # pick next neighbor that is not the previous vertex
                if not nbrs:
                    break
                if prev is None:
                    nxt = nbrs[0]
                else:
                    nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else nbrs[0])

                # remove the used edge from adjacency
                try:
                    adj[current].remove(nxt)
                except ValueError:
                    pass
                try:
                    adj[nxt].remove(current)
                except ValueError:
                    pass
                if not adj[current]:
                    del adj[current]
                if not adj.get(nxt):
                    adj.pop(nxt, None)

                # append next to chain
                chain.append(nxt)
                prev, current = current, nxt

                # closed loop?
                if current == start:
                    break

            # convert chain of corner points to polygon points (x,y)
            if len(chain) >= 4:
                # ensure closed
                if chain[0] != chain[-1]:
                    chain.append(chain[0])
                points = [(float(px), float(py)) for px, py in chain]
                polygons.append(points)

    return polygons

# 
'''for val in np.unique(new_lattice):
    # skip background
    if val == 0:
        continue
    mask = (new_lattice == val)
    polys = blocky_polygons(mask)
    for points in polys:
        if len(points) < 3:
            continue
        poly = Polygon(points, closed=True, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(poly)'''

plt.show()