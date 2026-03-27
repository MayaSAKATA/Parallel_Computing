import numpy as np
from scipy.spatial import distance
import time
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys

G = 1.560339e-13  # Gravitationnal constant


def initialize_grid(positions):
    """
    Initialize the global variable square_size and radius of the square
    """
    global square_size, radius
    size_x = positions[:, 0].max() - positions[:, 0].min()
    size_y = positions[:, 1].max() - positions[:, 1].min()
    size_z = positions[:, 2].max() - positions[:, 2].min()

    dx = size_x / 20
    dy = size_y / 20
    dz = size_z / 20

    square_size = np.array([dx, dy, dz]) * 1.05 # Add a 5% margin to ensure stars on the edge stay within bounds
    radius = np.sqrt(dx**2 + dy**2 + dz**2)
    return square_size, radius


def assign_to_grid(positions):
    """
    Assign each star to a grid square : grid = {(grid_x, grid_y): [index1, index2, ...}
    For each square coordinates (grid_x, grid_y) we have a list of idx of stars
    """
    global square_size
    grid = {}

    for i, pos in enumerate(positions):
        gx = int(pos[0] // square_size[0])
        gy = int(pos[1] // square_size[1])
        gz = int(pos[2] // square_size[2])
        key = (gx, gy, gz)

        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    return grid


def center_gravity(positions, mass):
    """
    Calculate the center of gravity of a set of stars.
    """
    total_mass = mass.sum()
    cx = np.sum(positions[:, 0] * mass) / total_mass
    cy = np.sum(positions[:, 1] * mass) / total_mass
    cz = np.sum(positions[:, 2] * mass) / total_mass
    return np.array([cx, cy, cz]), total_mass


def calculate_acceleration(positions, mass):
    """
    Calculate the gravitational accelerations on each body due to all other bodies.
    """
    global radius

    n = len(positions)
    accelerations = np.zeros((n, 3))

    # Compute once
    grid = assign_to_grid(positions)
    cg, total_mass = center_gravity(positions, mass)

    for i in range(n):
        acc = np.zeros(3)

        for key, indices in grid.items():
            # Distance betwen center of mass and cell
            dist = distance.euclidean(cg, positions[i])

            # Barnes-Hut approximation 
            if 0.5 * dist > radius :
                diff = cg - positions[i]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    acc += G * total_mass * diff / (d**3)

            else:
                for j in indices:
                    if i == j:
                        continue
                    diff = positions[j] - positions[i]
                    d = np.linalg.norm(diff)
                    if d > 1e-10:
                        acc += G * mass[j] * diff / (d**3)

        accelerations[i] = acc

    return accelerations


def step(dt):
    """
    Update the positions and velocities of all stars using the Verlet integration method.
    """
    global positions, velocity, mass
    acc = calculate_acceleration(positions, mass)

    new_positions = positions + velocity * dt + 0.5 * acc * dt**2
    new_acc = calculate_acceleration(new_positions, mass)
    new_velocity = velocity + 0.5 * (acc + new_acc) * dt

    positions = new_positions
    velocity = new_velocity
    return positions


def load_galaxy(filename):
    """
    Load a system of stars from a file like (mass, positionx, positiony, positionz, velocityx, velocityy, velocityz)
    """
    positions, velocity, color, mass = [], [], [], []

    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            positions.append(data[1:4])
            velocity.append(data[4:7])
            color.append(generate_star_color(data[0]))
            mass.append(data[0])

    return np.array(positions), np.array(velocity), np.array(mass), np.array(color)


if __name__ == "__main__":
    global positions, velocity, mass, color, square_size, radius

    positions, velocity, mass, color = load_galaxy(f"data/galaxy_{sys.argv[2] if len(sys.argv) > 2 else '100'}")

    square_size, radius = initialize_grid(positions)
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2

    start = time.time()
    for _ in range(10):
        step(dt)
    end = time.time()

    print(f"Time for 10 steps ({len(mass)} bodies): {end - start:.4f} seconds")

    luminosities = np.ones(len(positions), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(positions, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)