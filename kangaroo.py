"""
kangaroo.py
by Thomas Oberbichler <thomas.oberbichler@gmail.com> (c) 2022

This is a Python implementation of Nesterov-Momentum for form-finding.
With the current settings it will produce the same results as the solver
in Kangaroo3D for Grasshopper (http://kangaroo3d.com/).

It contains following goals:
- Spring
- Anchor

Additional goals can be implemented in the same way.

The results are compared to Kangaroo. They match on machine accuracy.
"""

import json
import numpy as np
from numpy.linalg import norm


class Particle:
    def __init__(self, x, y, z):
        self.position = np.asarray([x, y, z], float)
        self.velocity = np.zeros(3)
        self.force = np.zeros(3)
        self.mass = 0.0


class Spring:
    def __init__(self, particle_a, particle_b, rest_length, stiffness):
        self.particles = [particle_a, particle_b]
        self.rest_length = rest_length
        self.stiffness = stiffness

        self.force = np.zeros((2, 3))
        self.lumped_mass = np.array([2.0 * stiffness, 2.0 * stiffness])  # "2.0 *" is just required to reproduce the same results as Kangaroo3d.

    def compute(self):
        particle_a, particle_b = self.particles

        v = particle_b.position - particle_a.position

        rest_length = self.rest_length
        actual_length = norm(v)

        delta = actual_length - rest_length

        direction = v / actual_length
        
        self.force[0] = delta * self.stiffness * direction
        self.force[1] = -self.force[0]


class Anchor:
    def __init__(self, particle, stiffness):
        self.particles = [particle]
        self.target = particle.position.copy()
        self.stiffness = stiffness

        self.force = np.zeros((1, 3))
        self.lumped_mass = np.array([stiffness], float)

    def compute(self):
        delta = self.target - self.particles[0].position

        self.force[0] = delta * self.stiffness


class ResidualCriterion:
    """
    Checks whether the residual forces are zero.
    """
    def __init__(self, tolerance=1e-8):
        self.tolerance = tolerance

    def __call__(self, i, particles):
        residual_norm = 0.0

        for particle in particles:
            residual_norm += particle.force @ particle.force

        residual_norm = np.sqrt(residual_norm)

        print(f'i = {i:>4}: rnorm = {residual_norm:<12.6g}')

        return residual_norm < self.tolerance


class KangarooCriterion:
    """
    Kangaroo checks at every 10th iteration if the average speed is zero.
    """
    def __init__(self, tolerance=1e-15):
        self.tolerance = tolerance

    def __call__(self, i, particles):
        if i % 10 != 9:
            return False
        
        average_squared_velocity = 0.0
        residual_norm = 0.0

        for particle in particles:
            average_squared_velocity += norm(particle.velocity)**2
            residual_norm += particle.force @ particle.force

        average_squared_velocity /= len(particles)
        residual_norm = np.sqrt(residual_norm)

        print(f'i = {i:>4}: average_squared_velocity = {average_squared_velocity:<12.6g} rnorm = {residual_norm:<12.6g}')

        return average_squared_velocity < self.tolerance


def solve(goals, damping=0.9, maxiter=int(1e6), breaking_criterion=None):
    """Solves a system of goals using Nesterov-Momentum.


    Keyword arguments:
    damping -- the damping factor (default: 0.9)
    maxiter -- the maximum number of iterations (default: 1e6)
    breaking_criterion -- the criterion for breaking the solving process (default: KangarooCriterion)

    The iteration for Nesterov-Momentum:

      p_{t+1} = \beta p_{t} - \alpha ∇f(x_{t} + \beta p_{t})
      
      x_{t+1} = x_{t} + p_{t+1}
    
      where t      ... time/iteration
            x      ... position
            p      ... momentum (p_{0} = 0)
            \alpha ... step-size/learing-rate
            \beta  ... momentum-decay

    The scalar step-size \alpha is replaced with the inverse mass-matrix:

      v_{t+1} = \beta_{t} v_{t} - M^{-1} * r(x_{t} + \beta_{t} v_{t})
                                \-----------------------------------/
                                               a_{t}

      x_{t+1} = x_{t} + v_{t+1}
              = x_{t} + \beta_{t} v_{t} + a_{t}

      where v ... velocity (v_{0} = 0)
            M ... (artificial) lumped mass-matrix
            r ... residual force (= ∇f)
            a ... acceleration

    \beta acts as a damping factor and is adapted in each iteration:

      \beta_{t+1} = 1.0   if a_t @ v_{t+1} >= 0
                  < 1.0   otherwise
    
      where \beta_0 = 0
    """
   
    # use Kangaroos breaking criterion by default
    if breaking_criterion is None:
        breaking_criterion = KangarooCriterion()

    # collect particles
    particles = set()
    for goal in goals:
        particles = particles.union(set(goal.particles))

    print(f'Number of particles: {len(particles)}')
    print(f'Number of goals:     {len(goals)}')

    for i in range(maxiter):
        for particle in particles:
            # move to x_{t} + \beta_{t} v_{t}
            particle.position += particle.velocity

            # reset force and mass
            particle.force = np.zeros(3)
            particle.mass = 0.0

        for goal in goals:
            # compute force and mass contribution of each goal
            goal.compute()

            # accumulate force and mass at each particle -> r_{t} and M
            for j, particle in enumerate(goal.particles):
                particle.force += goal.force[j]
                particle.mass += goal.lumped_mass[j]

        for particle in particles:
            if norm(particle.force) == 0.0:
                particle.velocity = np.zeros(3)
            else:
                # a_{t} = -M^{-1} * r(x_{t} + \beta_{t} v_{t})
                a_t = particle.force / particle.mass
                
                # v_{t+1} = v_{t} + a_{t}
                particle.velocity += a_t

                # x_{t+1} = x_{t} + \beta_{t} v_{t} + a_{t}
                #           \---------------------/ \-----/
                #               applied in L175     missing
                particle.position += a_t

                # Adapt momentum-decay:
                # \beta_{t+1} < 1.0    if a_{t} @ v_{t+1} < 0
                #               1.0    otherwise 
                if a_t @ particle.velocity < 0.0:
                    particle.velocity *= damping

                # => particle.velocity includes damping (= \beta_{t} * v_{t})

        if breaking_criterion(i, particles):
            break

    if i + 1 == maxiter:
        print("WARNING: Maximum number of iterations has been exceeded")
    else:
        print("Success")


def load(json_str):
    data = json.loads(json_str)

    particles = []
    goals = []

    for particle_data in data["particles"]:
        x = particle_data["x"]
        y = particle_data["y"]
        z = particle_data["z"]
        stiffness = particle_data["stiffness"]

        particle = Particle(x, y, z)

        particles.append(particle)

        if stiffness == 0:
            continue

        anchor = Anchor(particle, stiffness)

        goals.append(anchor)

    for spring_data in data["springs"]:
        particle_a = particles[spring_data["particle_a"]]
        particle_b = particles[spring_data["particle_b"]]
        rest_length = spring_data["rest_length"]
        stiffness = spring_data["stiffness"]

        spring = Spring(particle_a, particle_b, rest_length, stiffness)

        goals.append(spring)
    
    return particles, goals
