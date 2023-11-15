#!/usr/bin/env python

import pygame
import numpy as np
import time

TIME_STEP = 0.01
TEMPERATURE = 1.0
SAMPLING_VARIANCE = 100.0
HORIZON = 20
NUM_SAMPLES = 10

def robot_dynamics(x: np.array, u: np.array) -> np.array:
    """
    Given the state x and control u, return the next state for a simple
    robot with integrator dynamics. 
    """
    return x + u * TIME_STEP

def proportional_controller(x: np.array, x_nom: np.array) -> np.array:
    """
    Given the state x and the target state x_nom, return a control input
    u that will drive the robot towards the target state.
    """
    k = 2.0
    return -k * (x - x_nom)

def sample_control_tape(x0: np.array, u_nom: np.array) -> np.array:
    """
    Given the initial state x0 and the nominal control u_nom, return a
    perturbed control tape that is sampled from a Gaussian distribution
    centered at u_nom.
    """
    du = np.random.normal(0, SAMPLING_VARIANCE, u_nom.shape)
    return u_nom + du

def rollout(x0: np.array, u_tape: np.array) -> np.array:
    """
    Given the initial state x0 and the control tape u_tape, return the
    resulting state trajectory.
    """
    x = x0
    x_traj = [x]
    for u in u_tape:
        x = robot_dynamics(x, u)
        x_traj.append(x)
    return np.array(x_traj)



# Initialize pygame
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Set the initial state
x = np.array([200, 200])
u = np.array([10, 10])

# Initialize the target position
x_nom = np.array([400, 250])

# Initialize the nominal control tape
u_nom = np.array([[500, 0.0] for _ in range(HORIZON)])
x_traj = rollout(x, u_nom)
print(x_traj)

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
    u = proportional_controller(x, x_nom)

    # Sample some control tapes
    for _ in range(NUM_SAMPLES):
        u_tape = sample_control_tape(x, u_nom)
        x_traj = rollout(x, u_tape)

        for t in range(HORIZON-1):
            pygame.draw.line(screen, (255, 0, 0), x_traj[t], x_traj[t+1])

    # Update the nominal control tape
    u_nom = np.array([u for _ in range(HORIZON)])

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

