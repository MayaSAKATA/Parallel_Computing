from operator import pos
import numpy as np
from scipy.spatial import distance_matrix, distance
import time
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys
import numba

G = 1.560339e-13 # Gravitationnal constant

def initialize_grid(position):
    """
    Initialize the global variable square_size and radius of the square
    """
    global square_size, radius
    size_x = max(position[:, 0]) - min(position[:, 0])
    size_y = max(position[:, 1]) - min(position[:, 1])
    size_z = max(position[:, 2]) - min(position[:, 2])
    square_size = [size_x, size_y, size_z]*1.05 / 20 # Ajust size with a margin
    radius = np.sqrt(size_x**2 + size_y**2 + size_z**2)

def assign_to_grid(position):
    """
    Assign each star to a grid square : grid = {(grid_x, grid_y): [index1, index2, ...}
    For each square coordinates (grid_x, grid_y) we have a list of idx of stars
    """
    global square_size
    grid = {}
    for i, pos in enumerate(position):
        grid_x = int(pos[0] // square_size[0])
        grid_y = int(pos[1] // square_size[1])
        key = (grid_x, grid_y)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid

def center_gravity(position, mass):
    """
    Calculate the center of gravity of a set of stars.
    """
    total_mass = np.sum(mass)
    center_of_mass_x = np.sum(position[0]*mass) / total_mass
    center_of_mass_y = np.sum(position[1]*mass) / total_mass
    center_of_mass_z = np.sum(position[2]*mass) / total_mass

    return [center_of_mass_x, center_of_mass_y, center_of_mass_z]

@numba.njit(parallel=True)
def calculate_acceleration(position, mass):
    """
    Calculate the gravitational accelerations on each body due to all other bodies.
    Based on this formula : accel[i] = f[i] / m[i] 
    """
    global square_size, radius
    grid = assign_to_grid(position) # Update grid assignment at each step

    for key, indices in grid.items():
        s = grid[key]
        dist = distance.euclidean(center_gravity(position[indices], mass[indices]), s)



    n = position.shape[0]
    accelerations = np.zeros((n, 3), dtype=np.float64) 

    for i in numba.prange(n):
        acc = np.zeros(3)
        for j in range(n):
            if i == j:
                continue
            diff = position[j] - position[i]
            dist = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            if dist > 1e-10:
                acc += G * mass[j] * diff / (dist**3)
        accelerations[i] = acc
    return accelerations

def step(dt):
    """
    Updates the all the positions in the system after a time step dt.
    """
    global position, velocity, mass
    acc = calculate_acceleration(position, mass)

    new_position = position + velocity * dt + 0.5 * acc * dt**2

    new_acc = calculate_acceleration(new_position, mass)

    new_velocity = velocity + 0.5 * (acc + new_acc) * dt

    # upater doesn't edit variables, returns new values
    position = new_position
    velocity = new_velocity
    return position


def load_galaxy(filename):
    """
    Load a system of bodies from a file like (mass, positionx, positiony, positionz, velocityx, velocityy, velocityz)
    """
    position = [] # [[0, 0, 0], [0, 0, 0], ...]
    velocity = [] # [[0, 0, 0], [0, 0, 0], ...]
    color = []  # [(255, 255, 255), (255, 255, 255), ...]
    mass = [] # [1.0, 1.0, ...]

    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            position.append(data[1:4])
            velocity.append(data[4:7])
            color.append(generate_star_color(data[0]))
            mass.append(data[0])
    return np.array(position), np.array(velocity), np.array(mass), np.array(color)

if __name__ == "__main__":

    global position, velocity, mass, color, square_size, radius
    position, velocity, mass, color  = load_galaxy("data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100"))
    
    square_size, radius = initialize_grid(position)


    print(f"position shape: {position.shape}")

    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2

    # Time the execution of 10 steps
    # start_time = time.time()
    # for _ in range(10):
    #     step(dt)
    # end_time = time.time()
    # print(f"Time for 10 steps ({len(mass)} bodies): {end_time - start_time:.4f} seconds\n")
    
    # Visualization
    luminosities = np.ones(len(position), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(position, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)