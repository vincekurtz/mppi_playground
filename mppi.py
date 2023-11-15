#!/usr/bin/env python

import pygame
import numpy as np
import time
from typing import List

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

def compute_cost(x: np.array, x_nom: np.array, u: np.array) -> float:
    """
    Given the state x, the nominal state x_nom, and the control u, return
    the running cost. 
    """
    return np.linalg.norm(x - x_nom)**2 + 0.01 * np.linalg.norm(u)**2

def compute_trajectory_cost(x_traj: np.array, x_nom: np.array, u_tape: np.array) -> float:
    """
    Given the state trajectory x_traj, the nominal state x_nom, and the
    control tape u_tape, return the total cost.
    """
    cost = 0.0
    for x, u in zip(x_traj, u_tape):
        cost += compute_cost(x, x_nom, u)
    return cost

def vanilla_mppi(x0: np.array, x_nom: np.array, u_guess: np.array) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0, the nominal state x_nom, and an initial guess
    for the control tape, perform MPPI to get a new control tape.

    Returns a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = []
    Xs = []
    costs = []
    for _ in range(NUM_SAMPLES):
        u_tape = sample_control_tape(x0, u_guess)
        x_tape = rollout(x0, u_tape)
        Us.append(u_tape)
        Xs.append(x_tape)
        costs.append(compute_trajectory_cost(x_tape, x_nom, u_tape))

    # Compute the weights
    costs = np.array(costs)
    min_cost = np.min(costs)
    weights = np.exp(-(costs-min_cost) / TEMPERATURE)
    weights /= np.sum(weights)

    # Compute the new control tape
    u_nom = np.zeros(u_guess.shape)
    for u_tape, weight in zip(Us, weights):
        u_nom += weight * u_tape

    # Compute the new state trajectory
    x_nom = rollout(x0, u_nom)

    # Append the new control tape and state trajectory
    Us.append(u_nom)
    Xs.append(x_nom)

    return Us, Xs


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
    u = u_nom[0]

    # Perform an MPPI step
    Us, Xs = vanilla_mppi(x, x_nom, u_nom)

    # Visualize the MPPI samples
    for x_traj in Xs:
        for t in range(len(x_traj)-1):
            pygame.draw.line(screen, (255, 0, 0), x_traj[t, :], x_traj[t+1,:], width=1)

    # Visualize the best trajectory with a thicker line
    x_star = Xs[-1]
    for t in range(len(x_star)-1):
        pygame.draw.line(screen, (15, 20, 20), x_star[t, :], x_star[t+1,:], width=3)

    # Update the nominal control tape
    u_nom = Us[-1]

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

