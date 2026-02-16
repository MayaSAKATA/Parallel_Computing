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

    def calculate_accelerations(self, body_i):
        """
        Calculate the gravitational accelerations on each body due to all other bodies.
        """
        total_acc = np.zeros(3)
        
        for body_j in self.collection:
            if body_j is not body_i: # i != j
                dist = body_j.distance(body_i)
                if dist > 1e-10:
                    diff = body_j.position - body_i.position
                    total_acc += G * body_j.mass * diff / (dist**3)
                
        return total_acc

    def step(self, dt):
        """
        Performs one complete simulation step.
        """
        for b in self.collection:
            accel = self.calculate_accelerations(b)
            b.update(accel, dt)
        return [body.position for body in self.collection]

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