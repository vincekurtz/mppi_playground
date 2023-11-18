#!/usr/bin/env python

import pygame
import numpy as np
import time
import sys

from base import ProblemData, Obstacle, unicycle_dynamics
from mppi import do_mppi_iteration


def simulate():
    """
    Run a quick little simulation with pygame.
    """
    # Set up pygame
    pygame.init()
    ppm = 100   # pixels per meter
    screen = pygame.display.set_mode([5*ppm, 5*ppm])

    # Set the initial state
    x = np.array([2, 2, 0])

    # Create problem data
    obstacles = [
        Obstacle(4, 1, 1, 1),
        Obstacle(2, 3, 3, 0.5),]
    data = ProblemData(x_nom=np.array([4, 2.5, 0]),
                       obstacles=obstacles)

    # Initialize the nominal control tape
    u_nom = np.array([[0.0, 0.0] for _ in range(data.mppi_horizon-1)])

    # Run until the user asks to quit
    running = True
    dragging_target = False
    while running:
        # Fill the background
        screen.fill((255, 255, 255))

        # Draw the obstacles
        for obstacle in data.obstacles:
            obstacle.draw(screen, pixels_per_meter=ppm)

        # Perform an MPPI step
        Us, Xs = do_mppi_iteration(x, u_nom, data)

        # Visualize a few of the MPPI samples
        for i in range(min(len(Xs), data.mppi_num_samples)):
            x_traj = Xs[i]
            for t in range(len(x_traj)-1):
                pygame.draw.line(screen, (255, 0, 0),
                                 ppm*x_traj[t, :2], ppm*x_traj[t+1, :2], width=1)
        
        # Draw the robot and the target
        pygame.draw.circle(screen, (0, 0, 255), (ppm*x[0], ppm*x[1]), 10)
        pygame.draw.circle(screen, (0, 255, 0), (ppm*data.x_nom[0], ppm*data.x_nom[1]), 10)

        # Visualize the best trajectory with a thicker line
        x_star = Xs[-1]
        for t in range(len(x_star)-1):
            pygame.draw.line(screen, (0, 0, 255),
                             ppm*x_star[t, :2], ppm*x_star[t+1, :2], width=3)

        # Update the nominal control tape
        u_nom = Us[-1]

        # Update the state
        u = u_nom[0]
        x = unicycle_dynamics(x, u, data.sim_time_step)

        pygame.display.flip()
        for event in pygame.event.get():
            # Update the target position if the user clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                dragging_target = True
                data.x_nom[:2] = np.array(pygame.mouse.get_pos()) / ppm
            if event.type == pygame.MOUSEBUTTONUP:
                dragging_target = False
            if event.type == pygame.MOUSEMOTION and dragging_target:
                data.x_nom[:2] = np.array(pygame.mouse.get_pos()) / ppm

            # Close the window if the user presses the close button
            if event.type == pygame.QUIT:
                running = False

        # run in roughly real time
        time.sleep(data.sim_time_step)


if __name__ == "__main__":
    simulate()
