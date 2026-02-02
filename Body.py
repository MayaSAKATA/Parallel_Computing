import numpy as np
from galaxy_generator import generate_star_color

class Body:
    def __init__(self, mass, position, speed):
        self.mass = mass
        self.position = position
        self.speed = speed
        self.color = generate_star_color(mass)

    def __str__(self):
        return f"Mass: {self.mass}, Position: {self.position}, Speed: {self.speed}, Color: {self.color}"

    def update(self, dt):
        acceleration = NBodies.get_attraction(self)/self.mass
        self.speed = (self.speed[0] + acceleration[0] * dt, self.speed[1] + acceleration[1] * dt, self.speed[2] + acceleration[2] * dt)
        self.position = (self.position[0] + self.speed[0] * dt + 1/2*dt**2*acceleration[0], self.position[1] + self.speed[1] * dt + 1/2*dt**2*acceleration[1], self.position[2] + self.speed[2] * dt + 1/2*dt**2*acceleration[2])
    
    def distance(self, other):
        return np.linalg.norm(np.array(self.position) - np.array(other.position))
    
class NBodies:

    def __init__(self, filename):
        """
        Initialize a system of bodies from a file containing their properties (mass, positionx, positiony, positionz, speedx, speedy, speedz).
        """
        list_bodies = []
        with open(filename, 'r') as file:
            for line in file:
                data = line.split()
                mass = float(data[0])
                position = (float(data[1]), float(data[2]), float(data[3]))
                speed = (float(data[4]), float(data[5]), float(data[6]))
                body = Body(mass, position, speed)
                list_bodies.append(body)

        global G # Gravitationnal constant
        G = 1.560339e-13
        self.collection = list_bodies
        self.attraction = 0
        for body in self.collection:
            if body != self:
                self.attraction += G*self.mass*body.mass/np.linalg.norm(body.position - self.position )**3*(body.position - self.position)
    
    def get_attraction(self):
        """
        Returns the total gravitational attraction on this body.
        """
        return self.attraction
    
    system = None
    def update_positions(dt):
        """
        Updates the positions of all bodies in the system after a time step dt.
        """
        global system
        system.update_positions(dt)
        return [body.position for body in system.collection]

def main():
    """
    Fonction principale pour tester le générateur de galaxie.
    """
    import sys
    
    # Default parameters
    n_stars = 100
    output_file = "data/galaxy_100"
    
    # Read command line arguments
    if len(sys.argv) > 1:
        n_stars = int(sys.argv[1])
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Galaxy generation
    masses, positions, velocities, colors = generate_galaxy(
        n_stars=n_stars,
        output_file=output_file
    )
    
    print(f"\nStatistiques de la galaxie:")
    print(f"  - Nombre total d'objets: {len(masses)}")
    print(f"  - Nombre d'étoiles: {n_stars}")
    print(f"  - Masse totale: {sum(masses):.2e} masses solaires")
    print(f"  - Masse moyenne des étoiles: {np.mean(masses[1:]):.2f} masses solaires")
    print(f"  - Distance min/max: {min(np.linalg.norm(p) for p in positions[1:]):.4f} / {max(np.linalg.norm(p) for p in positions[1:]):.4f} années-lumière")


if __name__ == "__main__":
    main()
