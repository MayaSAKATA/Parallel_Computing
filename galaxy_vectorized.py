import numpy as np
from scipy.spatial import distance_matrix
import time
from galaxy_body import load_galaxy
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys

G = 1.560339e-13 # Gravitationnal constant

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

def calculate_acceleration(position, mass):
    """
    Calculate the gravitational accelerations on each body due to all other bodies.
    Based on this formula : accel[i] = f[i] / m[i] 
    """
    diff = position[np.newaxis, :, :] - position[:, np.newaxis, :] # diff[i,j] = position[j] - position[i]
    dist = np.linalg.norm(diff, axis=2) # dist[i,j] = ||position[j] - position[i]||

    grav_factor = np.where(dist > 1e-10, G * mass / (dist**3), 0.0) # Calcul du facteur gravitationnel: G * m_j / r_ij^3
    
    np.fill_diagonal(grav_factor, 0.0) # avoid i == j case
    total_acc = np.sum(grav_factor[:, :, np.newaxis] * diff, axis=1)

    return total_acc

def update(acceleration, velocity, dt):
    """
    Updates position and velocity using the provided formulas:
    p(t+dt) = p(t) + dt*v(t) + 0.5 * dt^2 * a(t)
    v(t+dt) = v(t) + dt * a(t)
    """
    new_position = position + velocity * dt + 0.5 * acceleration * dt**2
    new_velocity = velocity + acceleration * dt

    return new_position, new_velocity


def step(dt):
    """
    Updates the all the positions in the system after a time step dt.
    """
    global position, velocity, mass
    accel = calculate_acceleration(position, mass)
    new_position, new_velocity = update(accel, velocity, dt)
    # upater doesn't edit variables, returns new values
    position = new_position
    velocity = new_velocity
    return position
    
if __name__ == "__main__":

    global position, velocity, mass, color
    position, velocity, mass, color  = load_galaxy("data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100"))
    
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2

    # Time the execution of 10 steps
    start_time = time.time()
    for _ in range(10):
        step(dt)
    end_time = time.time()
    print(f"Time for 10 steps ({len(mass)} bodies): {end_time - start_time:.4f} seconds\n")
    
    # Visualization
    luminosities = np.ones(len(position), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(position, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)