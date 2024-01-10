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

        Args:
            x: The x-coordinate of the center of the obstacle.
            y: The y-coordinate of the center of the obstacle.
            width: The width of the obstacle.
            height: The height of the obstacle.
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

        Args:
            pos: The point (a.k.a. robot position) to compute the distance to.

        Returns:
            dist: the shortest distance from the point to the obstacle.
        """
        dx = max(self.left - pos[0], pos[0] - self.right)
        dy = max(self.top - pos[1], pos[1] - self.bottom)
        return max(dx, dy)

    def contains(self, pos: np.array) -> bool:
        """
        Check if the given point is inside the obstacle.

        Args:
            pos: The point (a.k.a. robot position) to check.

        Returns:
            True if the point is inside the obstacle, False otherwise.
        """
        return self.left <= pos[0] <= self.right and self.top <= pos[1] <= self.bottom

    def draw(self, screen: pygame.Surface, buffer: float = 0.1, pixels_per_meter: int = 100):
        """
        Draw the obstacle for pygame.

        Args:
            screen: The pygame surface to draw on.
            buffer: The amount to shrink the obstacle by before drawing, in meters, to account for the robot's radius.
            pixels_per_meter: The number of pygame pixels that represent one meter.
        """
        left_px = int((self.left + buffer) * pixels_per_meter)
        top_px = int((self.top + buffer) * pixels_per_meter)
        width_px = int((self.width - 2*buffer) * pixels_per_meter)
        height_px = int((self.height - 2*buffer) * pixels_per_meter)
        pygame.draw.rect(screen, (0, 0, 0), (left_px, top_px, width_px, height_px))


@dataclass
class ProblemData:
    """
    A little struct for storing problem data and solver parameters.
    """
    # Target state
    x_nom: np.array

    # Obstacles
    obstacles: List[Obstacle]

    # Solver parameters
    mppi_lambda: float = 1.0
    mppi_sample_variance: np.array = np.array([0.2, 1.0])
    mppi_num_samples: int = 100
    mppi_horizon: int = 10
    mppi_dt: float = 0.05
    mppi_u_max: np.array = np.array([np.inf, np.inf])

    # Experimental settings
    sample_mppi: bool = True
    reject_samples: bool = False
    
    motion_primitives: bool = False
    reject_primitives: bool = False

    # Cost function parameters
    state_cost: np.array = np.array([20, 20, 0])
    control_cost: np.array = np.array([1.0, 0.1])
    obstacle_cost: float = 1e4
    obstacle_smoothing_factor: float = 0.1

    # Time step for the simulated dynamics
    sim_time_step: float = 0.01


def unicycle_dynamics(x, u, dt):
    """
    Forward dynamics of the unicycle robot.

    Args:
        x: state vector [px, py, theta]
        u: control vector [v, omega]
        dt: time step

    Returns:
        x_next: next state vector
    """
    x_next = np.zeros(3)
    x_next[0] = x[0] + u[0] * np.cos(x[2]) * dt
    x_next[1] = x[1] + u[0] * np.sin(x[2]) * dt
    x_next[2] = x[2] + u[1] * dt
    return x_next
