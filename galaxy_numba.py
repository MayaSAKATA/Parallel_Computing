import numpy as np
import time
from galaxy_generator import generate_star_color
from visualizer3d_vbo import Visualizer3D
import sys
import numba

G = 1.560339e-13 # Gravitationnal constant

#---------------------------
#---version non-parallèle---
#---------------------------
@numba.njit

def calculate_acceleration(position, velocity, mass, dt):
    n = position.shape[0]
    new_pos = np.empty_like(position)
    new_vel = np.empty_like(velocity)

    for i in range(n):  # normal Python loop
        acc = np.zeros(3)
        for j in range(n):
            if i == j:
                continue
            diff = position[j] - position[i]
            dist = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
            if dist > 1e-10:
                acc += G * mass[j] * diff / (dist**3)

        new_pos[i] = position[i] + velocity[i] * dt + 0.5 * acc * dt**2
        new_vel[i] = velocity[i] + acc * dt

    return new_pos, new_vel


def step(dt):
    """
    Updates the all the positions in the system after a time step dt.
    """
    global position, velocity, mass
<<<<<<< HEAD
    new_position, new_velocity = calculate_acceleration(position, velocity, mass, dt)
    # upater doesn't edit variables, returns new values
=======
    new_position, new_velocity = calculate_acceleration(position, velocity, mass)
    # updater doesn't edit variables, returns new values
>>>>>>> bbe5a6b003b98aaada5eb5c37c66b696ee9768d2
    position = new_position
    velocity = new_velocity
    return position






#------------------------
#---version parrallèle---
#------------------------
# @numba.njit(parallel=True)
# def calculate_acceleration(position, velocity, mass):
#     """
#     Calculate the gravitational accelerations on each body due to all other bodies.
#     Based on this formula : accel[i] = f[i] / m[i] 
#     """
#     n = position.shape[0]
#     new_pos = np.empty_like(position)
#     new_vel = np.empty_like(velocity)

#     for i in numba.prange(n):
#         acc = np.zeros(3)
#         for j in range(n):
#             if i == j:
#                 continue
#             diff = position[j] - position[i]
#             dist = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
#             if dist > 1e-10:
#                 acc += G * mass[j] * diff / (dist**3)
#         new_pos[i] = position[i] + velocity[i] * dt + 0.5 * acc * dt**2
#         new_vel[i] = velocity[i] + acc * dt

#     return new_pos, new_vel

# def step(dt):
#     """
#     Updates the all the positions in the system after a time step dt.
#     """
#     global position, velocity, mass
#     new_position, new_velocity = calculate_acceleration(position, velocity, mass)
#     # upater doesn't edit variables, returns new values
#     position = new_position
#     velocity = new_velocity
#     return position



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


def measure_time(func, *args, **kwargs):
    """
    Measure the execution time of any function.
    Returns (result, elapsed_time).
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


# --- main non parrallèle ---
if __name__ == "__main__":
    
    # print("Numba threads:", numba.get_num_threads())

    # --- Measure load_galaxy ---
    (position, velocity, mass, color), t_load = measure_time(
        load_galaxy,
        "data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100")
    )
    print(f"load_galaxy time: {t_load:.6f} s")

    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2

    # --- Measure calculate_acceleration ---
    # Warm-up for Numba JIT
    calculate_acceleration(position, velocity, mass, dt)

    (_, _), t_acc = measure_time(calculate_acceleration, position, velocity, mass, dt)
    print(f"calculate_acceleration time: {t_acc:.6f} s")

    # --- Measure step ---
    # Warm-up
    step(dt)

    (_, t_step) = measure_time(step, dt)
    print(f"step time: {t_step:.6f} s")

    # --- Visualization ---
    luminosities = np.ones(len(position), dtype=np.float32)
    bounds = ((-3, 3), (-3, 3), (-3, 3))

    visualizer = Visualizer3D(position, color, luminosities, bounds)
    visualizer.run(updater=step, dt=dt)

    
# # --- main parallèle ---
# if __name__ == "__main__":
    
#     print("Numba threads:", numba.get_num_threads())

#     # --- Measure load_galaxy ---
#     (position, velocity, mass, color), t_load = measure_time(
#         load_galaxy,
#         "data/galaxy_{}".format(sys.argv[2] if len(sys.argv) > 2 else "100")
#     )
#     print(f"load_galaxy time: {t_load:.6f} s")

#     dt = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2

#     # --- Measure calculate_acceleration ---
#     # Warm-up for Numba JIT
#     calculate_acceleration(position, velocity, mass)

#     (_, _), t_acc = measure_time(calculate_acceleration,position, velocity, mass)
#     print(f"calculate_acceleration time: {t_acc:.6f} s")

#     # --- Measure step ---
#     # Warm-up
#     step(dt)

#     (_, t_step) = measure_time(step, dt)
#     print(f"step time: {t_step:.6f} s")

#     # --- Visualization ---
#     luminosities = np.ones(len(position), dtype=np.float32)
#     bounds = ((-3, 3), (-3, 3), (-3, 3))

#     visualizer = Visualizer3D(position, color, luminosities, bounds)
#     visualizer.run(updater=step, dt=dt)

    
    
