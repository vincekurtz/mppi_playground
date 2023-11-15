#!/usr/bin/env python

import pygame
import numpy as np
import time
from typing import List

from base import ProblemData, Obstacle

from vanilla_mppi import vanilla_mppi
from rejection_sample_mppi import rejection_sample_mppi
from just_stop_mppi import just_stop_mppi

def integrator_dynamics(x: np.array, u: np.array) -> np.array:
    """
    Given the state x and control u, return xdot for a simple
    robot with integrator dynamics. 
    """
    return u


def simulate(mppi: callable = vanilla_mppi):
    """
    Run a quick little simulation with pygame, using the given MPPI strategy.
    """
    # Set up pygame
    pygame.init()
    screen = pygame.display.set_mode([500, 500])

    # Set the initial state
    x = np.array([200, 200])

    # Create problem data
    obstacles = [
        Obstacle(400, 100, 100, 100),
        Obstacle(200, 300, 300, 50),]
    data = ProblemData(x_nom=np.array([400, 250]),
                       obstacles=obstacles,
                       robot_dynamics=integrator_dynamics)

    # Initialize the nominal control tape
    u_nom = np.array([[0.0, 0.0] for _ in range(data.horizon)])

    # Run until the user asks to quit
    running = True
    dragging_target = False
    while running:
        # Draw stuff
        screen.fill((255, 255, 255))  # White background
        for obstacle in data.obstacles:
            obstacle.draw(screen)
        pygame.draw.circle(screen, (0, 0, 255), x, 10)  # Robot's position
        pygame.draw.circle(screen, (0, 255, 0), data.x_nom,
                           10)  # Target position

        # Perform an MPPI step
        Us, Xs = mppi(x, u_nom, data)

        # Visualize a few of the MPPI samples
        for i in range(min(len(Xs), 20)):
            x_traj = Xs[i]
            for t in range(len(x_traj)-1):
                pygame.draw.line(screen, (255, 0, 0),
                                 x_traj[t, :], x_traj[t+1, :], width=1)

        # Visualize the best trajectory with a thicker line
        x_star = Xs[-1]
        for t in range(len(x_star)-1):
            pygame.draw.line(screen, (0, 0, 255),
                             x_star[t, :], x_star[t+1, :], width=3)

        # Update the nominal control tape
        u_nom = Us[-1]

        # Update the state
        xdot = data.robot_dynamics(x, u_nom[0])
        x = x + xdot * data.time_step

        pygame.display.flip()
        for event in pygame.event.get():
            # Update the target position if the user clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                dragging_target = True
                data.x_nom = np.array(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONUP:
                dragging_target = False
            if event.type == pygame.MOUSEMOTION and dragging_target:
                data.x_nom = np.array(pygame.mouse.get_pos())

            # Close the window if the user presses the close button
            if event.type == pygame.QUIT:
                running = False

        # run in roughly real time
        time.sleep(data.time_step)


if __name__ == "__main__":
    simulate(mppi=just_stop_mppi)
