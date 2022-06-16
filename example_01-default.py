from kangaroo import *
import numpy as np

particles, goals = load(open('example_01.json').read())

solve(goals)

expected = np.array([
    [0.204663774296581, 0.709309307605647, 0],
    [-0.999574237567416, 0.999897261645, 0],
    [0.999523084414622, 0.00042533038873443, 0],
    [5.11531555739133E-05, 1.99967740797591, 0],
])

for i, particle in enumerate(particles):
    x, y, z = particle.position
    x_exp, y_exp, z_exp = expected[i]

    print(f'Particle {i}')
    print(f'  x = {x:<24} (error = {np.abs(x - x_exp)})')
    print(f'  y = {y:<24} (error = {np.abs(y - y_exp)})')
    print(f'  z = {z:<24} (error = {np.abs(z - z_exp)})')
