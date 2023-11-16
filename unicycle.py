#!/usr/bin/env python

import pygame
import numpy as np
import time
import sys

from base import ProblemData, Obstacle
from vanilla_mppi import vanilla_mppi
from rejection_sample_mppi import rejection_sample_mppi
from just_stop_mppi import just_stop_mppi
from log_barrier_mppi import log_barrier_mppi
from motion_primitives import motion_primitive_planner


def unicycle_dynamics(x: np.array, u: np.array) -> np.array:
    """
    Given the state x and control u, return xdot for a simple
    robot with unicycle dynamics.
    """
    theta = x[2]
    v = u[0]
    omega = u[1]

    xdot = np.array([v * np.cos(theta),
                     v * np.sin(theta),
                     omega])
    return xdot


def simulate(mppi: callable = vanilla_mppi):
    """
    Run a quick little simulation with pygame, using the given MPPI strategy.
    """
    # Set up pygame
    pygame.init()
    screen = pygame.display.set_mode([500, 500])

    # Set the initial state
    x = np.array([200, 200, 0])

    # Create problem data
    obstacles = [
        Obstacle(400, 100, 100, 100),
        Obstacle(200, 300, 300, 50),]
    data = ProblemData(x_nom=np.array([400, 250, 0]),
                       obstacles=obstacles,
                       robot_dynamics=unicycle_dynamics)
    data.sampling_variance = np.array([50, 5])
    data.state_cost = np.array([1, 1, 0.0])
    data.control_cost = np.array([0.01, 1.0])

    # Initialize the nominal control tape
    u_nom = np.array([[0.0, 0.0] for _ in range(data.horizon)])

    # Run until the user asks to quit
    running = True
    dragging_target = False
    while running:
        # Fill the background
        screen.fill((255, 255, 255))

        # Draw the obstacles
        for obstacle in data.obstacles:
            obstacle.draw(screen)

        # Perform an MPPI step
        Us, Xs = mppi(x, u_nom, data)

        # Visualize a few of the MPPI samples
        for i in range(min(len(Xs), data.num_samples)):
            x_traj = Xs[i]
            for t in range(len(x_traj)-1):
                pygame.draw.line(screen, (255, 0, 0),
                                 x_traj[t, :2], x_traj[t+1, :2], width=1)
        
        # Draw the robot and the target
        pygame.draw.circle(screen, (0, 0, 255), (x[0], x[1]), 10)
        pygame.draw.circle(screen, (0, 255, 0), (data.x_nom[0], data.x_nom[1]), 10)

        # Visualize the best trajectory with a thicker line
        x_star = Xs[-1]
        for t in range(len(x_star)-1):
            pygame.draw.line(screen, (0, 0, 255),
                             x_star[t, :2], x_star[t+1, :2], width=3)

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
                data.x_nom[:2] = np.array(pygame.mouse.get_pos())
            if event.type == pygame.MOUSEBUTTONUP:
                dragging_target = False
            if event.type == pygame.MOUSEMOTION and dragging_target:
                data.x_nom[:2] = np.array(pygame.mouse.get_pos())

            # Close the window if the user presses the close button
            if event.type == pygame.QUIT:
                running = False

        # run in roughly real time
        time.sleep(data.time_step)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python simulate.py [vanilla|juststop|rejection|barrier]")
        sys.exit(0)
    
    if sys.argv[1] == "vanilla":
        simulate(mppi=vanilla_mppi)
    elif sys.argv[1] == "juststop":
        simulate(mppi=just_stop_mppi)
    elif sys.argv[1] == "rejection":
        simulate(mppi=rejection_sample_mppi)
    elif sys.argv[1] == "barrier":
        simulate(mppi=log_barrier_mppi)
    elif sys.argv[1] == "primitive":
        simulate(mppi=motion_primitive_planner)
    else:
        print(f"Unknown MPPI method: {sys.argv[1]}")
