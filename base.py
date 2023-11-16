##
#
# General utilities for performing MPPI.
#
##

import pygame
import numpy as np
from typing import List
from dataclasses import dataclass


class Obstacle:
    """
    A rectangular obstacle.
    """

    def __init__(self, x: np.array, y: np.array, width: float, height: float):
        """
        Create an obstacle with the given parameters. 
        """
        self.left = x - width / 2
        self.right = x + width / 2
        self.top = y - height / 2
        self.bottom = y + height / 2
        self.width = width
        self.height = height

    def signed_distance(self, pos: np.array) -> float:
        """
        Compute the signed distance from the given point to the obstacle.
        This is negative if the point is inside the obstacle.
        """
        dx = max(self.left - pos[0], pos[0] - self.right)
        dy = max(self.top - pos[1], pos[1] - self.bottom)
        return max(dx, dy)

    def contains(self, pos: np.array) -> bool:
        """
        Check if the given point is inside the obstacle.
        """
        return self.left <= pos[0] <= self.right and self.top <= pos[1] <= self.bottom

    def draw(self, screen: pygame.Surface):
        """
        Draw the obstacle for pygame.
        """
        buffer = 10   # a little extra space to account for the robot's radius
        pygame.draw.rect(screen, (0, 0, 0), (self.left+buffer,
                         self.top+buffer, self.width-2*buffer, self.height-2*buffer))


@dataclass
class ProblemData:
    """
    A little struct for storing problem data and solver parameters.
    """
    # Target state
    x_nom: np.array

    # Obstacles
    obstacles: List[Obstacle]

    # Robot dynamics xdot = f(x, u)
    robot_dynamics: callable = None

    # Solver parameters
    temperature: float = 1.0
    sampling_variance: float = 100
    num_samples: int = 100
    horizon: int = 20

    # Cost function parameters
    control_cost: float = 0.01
    obstacle_cost: float = 1e2
    obstacle_smoothing_factor: float = 10

    # Time step for the dynamics
    time_step: float = 0.01


def sample_control_tape(u_nom: np.array, data: ProblemData) -> np.array:
    """
    Given the the nominal control u_nom, return a perturbed control tape that is
    sampled from a Gaussian distribution centered at u_nom.
    """
    du = np.random.normal(0, data.sampling_variance, u_nom.shape)
    return u_nom + du


def rollout(x0: np.array,
            u_tape: np.array,
            data: ProblemData) -> np.array:
    """
    Given the initial state x0 and the control tape u_tape, return the
    resulting state trajectory.

    Args:
        x0: The initial state.
        u_tape: The control tape.
        data: Problem data, including parameters, obstacles, target state, etc.

    """
    x = x0
    x_traj = [x]
    for u in u_tape:
        xdot = data.robot_dynamics(x, u)
        x = x + xdot * data.time_step
        x_traj.append(x)
    return np.array(x_traj)
