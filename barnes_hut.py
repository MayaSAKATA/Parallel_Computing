import numpy as np
import time
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys
import numba

G = 1.560339e-13  # Gravitational constant

def initialize_grid(positions):
    """
    Initialize square_size, radius and min coordinates from the current positions.
    """
    global square_size, radius, min_x, min_y

    min_x  = np.min(positions[:, 0])
    min_y  = np.min(positions[:, 1])
    size_x = np.max(positions[:, 0]) - min_x
    size_y = np.max(positions[:, 1]) - min_y
    size_z = np.max(positions[:, 2]) - np.min(positions[:, 2])

    square_size = np.array([size_x, size_y, size_z], dtype=np.float64) * 1.05 / 20
    radius = np.sqrt(size_x**2 + size_y**2 + size_z**2)

    return square_size, radius, min_x, min_y


@numba.njit
def grid_matrice_crs(positions, square_size, min_x, min_y):
    """
    The goal is to generate two lists: 
    1. One that stores the cumulative star counts (offsets) for each grid cell.
    2. One that stores the star IDs sorted by their grid cell order.
    
    'position' is a list of star coordinates, where position[id][0] is the x-coordinate of star 'id'.

    """
    n = len(positions)
    aux = np.zeros(400, dtype=numba.int64) # Counts the number of stars in each cell (20x20 = 400 cells) 
    beg_cases = np.zeros(401, dtype=numba.int64) # Stores the starting index of each cell in the 'tab' array

    for i in range(n):
        # min() prevents index errors if a star is exactly on the grid boundary
        col   = min(int((positions[i][0] - min_x) / square_size[0]), 19)
        ligne = min(int((positions[i][1] - min_y) / square_size[1]), 19)
        aux[ligne * 20 + col] += 1

    
    beg_cases[1:] = np.cumsum(aux) # beg_cases[0] = 0, beg_cases[1] = number of stars in cell 0, beg_cases[2] = total stars in cells 0 and 1...

    # Stores star IDs sorted by grid cell order
    aux2 = np.zeros(400, dtype=numba.int64)
    tab  = np.zeros(n,   dtype=numba.int64)

    for i in range(n):
        col = min(int((positions[i][0] - min_x) / square_size[0]), 19)
        ligne = min(int((positions[i][1] - min_y) / square_size[1]), 19)
        place = ligne * 20 + col
        tab[beg_cases[place] + aux2[place]] = i
        aux2[place] += 1

    return beg_cases, tab


@numba.njit
def cell_center_of_mass(positions, mass, beg_cases, tab, cell):
    """
    Compute the center of mass and total mass of all stars in a given cell.
    Formula : Center_of_mass = (\sum_j m_j * x_j)  (\sum m_j)
    """
    cx, cy, cz = 0.0, 0.0, 0.0
    total_mass = 0.0
    for k in range(beg_cases[cell], beg_cases[cell + 1]):
        j = tab[k]
        m = mass[j]
        cx += positions[j][0] * m
        cy += positions[j][1] * m
        cz += positions[j][2] * m
        total_mass += m
    if total_mass > 0.0:
        cx /= total_mass
        cy /= total_mass
        cz /= total_mass
    return cx, cy, cz, total_mass

@numba.njit(parallel=True)
def calculate_acceleration(positions, mass, square_size, radius, min_x, min_y):
    """
    Gravitational accelerations with a Barnes-Hut-like approximation:
      - if a cell is far (0.5 * dist_to_cell_center > radius) -> use its center of mass
      - otherwise -> sum over individual stars in the cell
    """
    beg_cases, tab = grid_matrice_crs(positions, square_size, min_x, min_y)

    n = positions.shape[0]
    accelerations = np.zeros((n, 3), dtype=np.float64)

    for i in numba.prange(n):
        acc = np.zeros(3)

        for cell in range(400): # 20 * 20
    
            if beg_cases[cell] == beg_cases[cell + 1]: # Skip empty cells
                continue

            cx, cy, cz, total_mass = cell_center_of_mass(positions, mass, beg_cases, tab, cell)

            # Distance from star i to the cell's center of mass
            dx = cx - positions[i][0]
            dy = cy - positions[i][1]
            dz = cz - positions[i][2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)

            if dist < 1e-10:
                continue

            if 0.5 * dist > radius: # Far cell : treat as a single body at its center of mass
                acc[0] += G * total_mass * dx / (dist**3)
                acc[1] += G * total_mass * dy / (dist**3)
                acc[2] += G * total_mass * dz / (dist**3)

            else: # Near cell : sum over individual stars
                for k in range(beg_cases[cell], beg_cases[cell + 1]):
                    j = tab[k]
                    if i == j:
                        continue
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dz = positions[j][2] - positions[i][2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    if dist > 1e-10:
                        acc[0] += G * mass[j] * dx / (dist**3)
                        acc[1] += G * mass[j] * dy / (dist**3)
                        acc[2] += G * mass[j] * dz / (dist**3)

        accelerations[i] = acc

    return accelerations


def step(dt):
    """
    Updates the all the positions in the system after a time step dt using the Verlet integration method.
    """
    global positions, velocity, mass, square_size, radius, min_x, min_y

    acc = calculate_acceleration(positions, mass, square_size, radius, min_x, min_y)

    new_pos = positions + velocity * dt + 0.5 * acc * dt**2
    new_acc = calculate_acceleration(new_pos, mass, square_size, radius, min_x, min_y)
    new_vel = velocity + 0.5 * (acc + new_acc) * dt

    positions = new_pos
    velocity  = new_vel
    return positions

def load_galaxy(filename):
    """
    Load a system of bodies from a file like (mass, positionx, positiony, positionz, velocityx, velocityy, velocityz)
    """
    positions, velocity, color, mass = [], [], [], []
    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            mass.append(data[0])
            positions.append(data[1:4])
            velocity.append(data[4:7])
            color.append(generate_star_color(data[0]))
    return (np.array(positions), np.array(velocity), np.array(mass), np.array(color))


if __name__ == "__main__":
    global positions, velocity, mass, color, square_size, radius

    galaxy_file = "data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100")
    positions, velocity, mass, color = load_galaxy(galaxy_file)

    square_size, radius, min_x, min_y = initialize_grid(positions)

    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3

    # Time the execution of 10 steps
    start_time = time.time()
    for _ in range(10):
        step(dt)
    end_time = time.time()
    print(f"Time for 10 steps ({len(mass)} bodies): {end_time - start_time:.4f} seconds\n")

    # Visualization
    luminosities = np.ones(len(positions), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(positions, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)