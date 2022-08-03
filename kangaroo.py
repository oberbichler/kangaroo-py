"""
kangaroo.py
by Thomas Oberbichler <thomas.oberbichler@gmail.com> (c) 2022

This is a Python implementation of Dynamic-Relaxation as used in Kangaroo3D
for Grasshopper (http://kangaroo3d.com/).

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
        self.move = np.zeros(3)
        self.mass = 0.0


class Spring:
    def __init__(self, particle_a, particle_b, rest_length, stiffness):
        self.particles = [particle_a, particle_b]
        self.rest_length = rest_length
        self.stiffness = stiffness

        self.move = np.zeros((2, 3))
        self.weighting = np.array([2.0 * stiffness, 2.0 * stiffness])  # "2.0 *" is just required to reproduce the same results as Kangaroo3d.

    def compute(self):
        particle_a, particle_b = self.particles

        v = particle_b.position - particle_a.position

        rest_length = self.rest_length
        actual_length = norm(v)

        delta = actual_length - rest_length

        direction = v / actual_length
        
        self.move[0] = delta * self.stiffness * direction / (2.0 * stiffness) # == 0.5 * delta * direction
        self.move[1] = -self.move[0]


class Anchor:
    def __init__(self, particle, stiffness):
        self.particles = [particle]
        self.target = particle.position.copy()
        self.stiffness = stiffness

        self.move = np.zeros((1, 3))
        self.weighting = np.array([stiffness], float)

    def compute(self):
        delta = self.target - self.particles[0].position

        self.move[0] = delta * self.stiffness / self.stiffness # == delta


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
    """Solves a system of goals using Dynamic-Relaxation.


    Keyword arguments:
    damping -- the damping factor (default: 0.9)
    maxiter -- the maximum number of iterations (default: 1e6)
    breaking_criterion -- the criterion for breaking the solving process (default: KangarooCriterion)

    For each particle we compute the residual force:
   
      r_{t} = m a_{t}                                     (1)
   
      where:  r ... residual force at the particle
              m ... mass of the particle
              a ... accelleration of the particle
              t ... current time
   
    The mass is assumed to be constant over time.
   
    The accelleration depends on time. It is given by the
    derivative of the velocity w.r.t. time. This derivative
    is approximated by central finite differences:
   
      a_{t} = dv/dt ≈ (v_{t+Δt/2} - v_{t-Δt/2}) / Δt      (2)
   
      where:  v ... velocity of the particle
              Δt... change of time
   
    From the combination of (1) and (2) we obtain:
   
      v_{t+Δt/2} = v_{t-Δt/2} + (r_{t} / m) Δt            (3)
   
    We compute the new position of each particle after Δt:
   
      x_{t+Δt} = x_t + v_{t+Δt/2} Δt
   
      where:  x ... location of the particle
              t ... time of last iteration
   
    The change of the location of the particle is given by:
   
      v_{t+Δt/2} Δt = v_{t-Δt/2} Δt + (r_{t} / m) Δt^2   (4)
                    = Δx_{t-Δt/2}   + Δx_{t}
   
    It consists of two parts which result from
    - the velocity and  -> Δx_{t-Δt/2} = v_{t-Δt/2} Δt    (5)
    - the acceleration  -> Δx_{t} = (r_{t} / m) Δt^2      (6)
    of the particle.
   
    We use a pseudo-timestep Δt = Δt^2 = 1.
   
    Starting at t=0 we move each node by
    - Δx_{t-Δt/2} at half time steps and
    - Δx_{t}      at full time steps:
   
   
               Δx_{t}  
                  ↓
      t  =  0     1     2     3 ...     n
              0.5   1.5   2.5       ...
               ↑
            Δx_{t-Δt/2}
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
        # time:
        # t = i + 0.5

        for particle in particles:
            # apply Δx_{t-Δt/2} -> (5)
            particle.position += particle.velocity

            # reset force and mass
            particle.force = np.zeros(3)
            particle.mass = 0.0

        # time:
        # t = i + 1.0

        for goal in goals:
            # compute force and mass contribution of each goal
            goal.compute()

            # accumulate force and mass at each particle
            for j, particle in enumerate(goal.particles):
                particle.force += goal.move[j] * goal.weighting[j]
                particle.mass += goal.weighting[j]

        for particle in particles:
            if norm(particle.force) == 0.0:
                particle.velocity = np.zeros(3)
            else:
                dx_a = particle.force / particle.mass
                
                # apply Δx_{t} -> (6)
                particle.position += dx_a
                
                # compute v_{t+Δt/2} -> (4)
                particle.velocity += dx_a

                # apply damping if Δx_{t} and Δx_{t+Δt/2} are in
                # opposite direction
                if dx_a @ particle.velocity < 0.0:
                    particle.velocity *= damping

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
