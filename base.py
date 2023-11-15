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

    def signed_distance_to(self, pos: np.array) -> float:
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
        pygame.draw.rect(screen, (0, 0, 0), (self.left, self.top, self.width, self.height))

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
    temperature: float = 1.0
    sampling_variance: float = 100
    num_samples: int = 100
    horizon: int = 20

    # Cost function parameters
    control_cost: float = 0.01
    obstacle_cost: float = 1e6

    # Time step for the dynamics
    time_step: float = 0.01
