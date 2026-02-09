import numpy as np
from scipy.spatial import distance_matrix
import time
from Body import load_galaxy
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys

G = 1.560339e-13 # Gravitationnal constant

def load_galaxy(filename):
    """
    Load a system of bodies from a file like (mass, positionx, positiony, positionz, speedx, speedy, speedz)
    """
    position = [] # [[0, 0, 0], [0, 0, 0], ...]
    speed = [] # [[0, 0, 0], [0, 0, 0], ...]
    color = []  # [(255, 255, 255), (255, 255, 255), ...]
    mass = [] # [1.0, 1.0, ...]

    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.split()))
            position.append(data[1:4])
            speed.append(data[4:7])
            color.append(generate_star_color(data[0]))
            mass.append(data[0])
    return np.array(position), np.array(speed), np.array(mass), np.array(color)

def acceleration(position, mass):
    """
    Calculate the gravitational accelerations on each body due to all other bodies.
    """
    total_acc = np.array([])

    M = np.transpose(mass).dot(mass)
    P = distance_matrix(position) # marche pas

    total_acc  = G * M / (P**3 + 1e-10) * (position - position.T)

    return total_acc

def update(acceleration, speed, dt):
    """
    Updates position and velocity using the provided formulas:
    p(t+dt) = p(t) + dt*v(t) + 0.5 * dt^2 * a(t)
    v(t+dt) = v(t) + dt * a(t)
    """
    position += speed * dt + 0.5 * acceleration * dt**2
    speed += acceleration * dt

    return position, speed


def step(dt):
    """
    Updates the all the positions in the system after a time step dt.
    """
    global position, speed, mass
    accel = acceleration(position, mass)
    position, speed = update(accel, speed, dt)

    return position
    
if __name__ == "__main__":

    global position, speed, mass, color
    position, speed, mass, color  = load_galaxy("data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100"))

    if len(sys.argv) > 1:
        dt = float(sys.argv[1])
    else :
        dt = 1e-2  # Time step in years
            
    
    # Visualization
    luminosities = np.ones(len(position), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(position, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)