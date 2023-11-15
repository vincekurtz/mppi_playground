#!/usr/bin/env python

import pygame
import numpy as np
import time

TIME_STEP = 0.001

def robot_dynamics(x: np.array, u: np.array) -> np.array:
    """
    Given the state x and control u, return the next state for a simple
    robot with integrator dynamics. 
    """
    return x + u * TIME_STEP



# Initialize pygame
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Set the initial state
x = np.array([200, 200])
u = np.array([10, 10])

# Initialize the target position
x_nom = np.array([400, 250])

# Run until the user asks to quit
running = True
while running:

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw the robot
    pygame.draw.circle(screen, (0, 0, 255), x, 10)

    # Draw the target position
    pygame.draw.circle(screen, (0, 255, 0), x_nom, 10)

    # Compute the control input
    u = -2.0 * (x - x_nom)

    # Update the state
    x = robot_dynamics(x, u)

    # Flip the display
    pygame.display.flip()

    for event in pygame.event.get():
        # Update the target position if the user clicks
        if event.type == pygame.MOUSEBUTTONDOWN:
            x_nom = np.array(pygame.mouse.get_pos())

        # Close the window if the user presses the close button
        if event.type == pygame.QUIT:
            running = False


    time.sleep(TIME_STEP)

