import numpy as np
import time
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys

G = 1.560339e-13 # Gravitationnal constant
class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.color = generate_star_color(mass)

    def __str__(self):
        return f"Mass: {self.mass}, Position: {self.position}, Velocity: {self.velocity}, Color: {self.color}"
    
    def distance(self, other):
        diff = self.position - other.position
        return np.linalg.norm(diff)
    
    def update(self, acceleration, dt):
        """
        Updates position and velocity using the provided formulas:
        p(t+dt) = p(t) + dt*v(t) + 0.5 * dt^2 * a(t)
        v(t+dt) = v(t) + dt * a(t)
        """
        self.position += self.velocity * dt + 0.5 * acceleration * dt**2
        self.velocity += acceleration * dt
    
class NBodies:

    def __init__(self, bodies_list):
        """
        Initialize a system of bodies from a file containing their properties (mass, positionx, positiony, positionz, speedx, speedy, speedz).
        """
        self.collection = bodies_list
        self.positions = np.array([body.position for body in bodies_list], dtype=np.float64)
        self.velocities = np.array([body.velocity for body in bodies_list], dtype=np.float64)
        self.masses = np.array([body.mass for body in bodies_list], dtype=np.float64)

    def calculate_acceleration(self, positions, masses):
        """
        Calculate the gravitational accelerations on each body due to all other bodies.
        """
        num_bodies = len(masses)
        accelerations = np.zeros((num_bodies, 3))

        for i in range(num_bodies):
            diff = positions - positions[i]

            diff[i] = 0
            dist = np.linalg.norm(diff, axis=1)
            mask = dist > 1e-10 # Avoid division by zero for very close bodies
            accel_components = G * (masses[mask]/(dist[mask]**3))[:, np.newaxis] * diff[mask]
            accelerations[i] = np.sum(accel_components, axis=0)
                
        return accelerations
    
    def update_position(self, dt):
        # step 1
        v1 = self.velocities
        p1 = self.positions
        a1 = self.calculate_acceleration(p1, self.masses)

        # step 2
        v2 = v1 + 0.5 * a1 * dt
        p2 = p1 + 0.5 * v1 * dt
        a2 = self.calculate_acceleration(p2, self.masses)

        # step 3
        v3 = v1 + 0.5 * a2 * dt
        p3 = p1 + 0.5 * v2 * dt
        a3 = self.calculate_acceleration(p3, self.masses)

        # step 4
        v4 = v1 + a3 * dt
        p4 = p1 + v3 * dt
        a4 = self.calculate_acceleration(p4, self.masses)

        # update position and velocity
        self.positions += (dt / 6) * (v1 + 2*v2 + 2*v3 + v4)
        self.velocities += (dt / 6) * (a1 + 2*a2 + 2*a3 + a4)

    def step(self, dt):
        """
        Performs one complete simulation step.
        """
        global system
        system.update_position(dt)
        return system.positions

def load_galaxy(filename):
        """
        Load a system of bodies from a file like (mass, positionx, positiony, positionz, speedx, speedy, speedz)
        """
        bodies = []
        with open(filename, 'r') as file:
            for line in file:
                data = list(map(float, line.split()))
                mass = data[0]
                position = data[1:4]
                speed = data[4:7]
                body = Body(mass, position, speed)
                bodies.append(body)
        return bodies

if __name__ == "__main__":

    galaxy = load_galaxy("data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100"))
    global system
    system = NBodies(galaxy)

    if system:
        if len(sys.argv) > 1:
            dt = float(sys.argv[1])
        else :
            dt = 1e-2  # Time step in years
        
        start_time = time.time()
        for _ in range(10):
            system.step(dt)
        end_time = time.time()
        
        print(f"Time for 10 steps ({len(system.collection)} bodies): {end_time - start_time:.4f} seconds")
    
    # Visualization
    points = np.array([body.position for body in system.collection])
    colors = np.array([body.color for body in system.collection])
    luminosities = np.ones(len(system.collection), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(points, colors, luminosities, bounds)
    visualizer.run(updater=system.step, dt=dt)