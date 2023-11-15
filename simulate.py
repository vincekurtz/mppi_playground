#!/usr/bin/env python

import pygame
import numpy as np
import time
from typing import List

from base import ProblemData, Obstacle

def robot_dynamics(x: np.array, u: np.array, data: ProblemData) -> np.array:
    """
    Given the state x and control u, return the next state for a simple
    robot with integrator dynamics. 
    """
    return x + u * data.time_step

def sample_control_tape(u_nom: np.array, data: ProblemData) -> np.array:
    """
    Given the the nominal control u_nom, return a perturbed control tape that is
    sampled from a Gaussian distribution centered at u_nom.
    """
    du = np.random.normal(0, data.sampling_variance, u_nom.shape)
    return u_nom + du

def rollout(x0: np.array, u_tape: np.array, data: ProblemData) -> np.array:
    """
    Given the initial state x0 and the control tape u_tape, return the
    resulting state trajectory.
    """
    x = x0
    x_traj = [x]
    for u in u_tape:
        x = robot_dynamics(x, u, data)
        x_traj.append(x)
    return np.array(x_traj)

def compute_cost(x: np.array, u: np.array, data: ProblemData) -> float:
    """
    Given the state x and control, compute the running cost.
    """
    state_cost = np.linalg.norm(x - data.x_nom)**2
    control_cost = data.control_cost * np.linalg.norm(u)**2

    obstacle_cost = 0
    for obstacle in data.obstacles:
        if obstacle.contains(x):
            obstacle_cost = data.obstacle_cost

    return state_cost + control_cost + obstacle_cost

def compute_trajectory_cost(x_traj: np.array, u_tape: np.array, data: ProblemData) -> float:
    """
    Given the state trajectory x_traj, the nominal state x_nom, and the
    control tape u_tape, return the total cost.
    """
    cost = 0.0
    for x, u in zip(x_traj, u_tape):
        cost += compute_cost(x, u, data)
    return cost

def vanilla_mppi(x0: np.array, u_guess: np.array, data: ProblemData) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform MPPI to get a new control tape.

    Returns a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = []
    Xs = []
    costs = []
    for _ in range(data.num_samples):
        u_tape = sample_control_tape(u_guess, data)
        x_tape = rollout(x0, u_tape, data)
        Us.append(u_tape)
        Xs.append(x_tape)
        costs.append(compute_trajectory_cost(x_tape, u_tape, data))

    # Compute the weights
    costs = np.array(costs)
    min_cost = np.min(costs)
    weights = np.exp(-(costs-min_cost) / data.temperature)
    weights /= np.sum(weights)

    # Compute the new control tape
    u_nom = np.zeros(u_guess.shape)
    for u_tape, weight in zip(Us, weights):
        u_nom += weight * u_tape

    # Compute the new state trajectory
    x_nom = rollout(x0, u_nom, data)

    # Append the new control tape and state trajectory
    Us.append(u_nom)
    Xs.append(x_nom)

    return Us, Xs

def contains_collisions(x_traj: np.array, data: ProblemData) -> bool:
    """
    Given the state trajectory x_traj, check if it contains any collisions.
    """
    for x in x_traj:
        for obstacle in data.obstacles:
            if obstacle.contains(x):
                return True
    return False

def rejection_sample_mppi(x0: np.array, u_guess: np.array, data: ProblemData) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform MPPI to get a new control tape.

    Do a rejection sampling variation, where we only keep samples that are
    collision-free.

    Returns a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = []
    Xs = []
    costs = []
    rs_iters = 0
    while len(Us) < NUM_SAMPLES and rs_iters < 500:
        u_tape = sample_control_tape(x0, u_guess)
        x_tape = rollout(x0, u_tape)

        # Check if the trajectory is collision-free. If so, keep it.
        if not contains_collisions(x_tape, data):
            Us.append(u_tape)
            Xs.append(x_tape)
            costs.append(compute_trajectory_cost(x_tape, u_tape, data))
        rs_iters += 1

    if len(Us) == 0:
        print("Warning: no collision-free samples found. Returning a zero control tape.")
        u = np.zeros(u_guess.shape)
        return [u], [rollout(x0, u)]

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

def just_stop_mppi(x0: np.array, u_guess: np.array, data: ProblemData) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform MPPI to get a new control tape.

    This is the same as vanilla_mppi, but we override the trajectory if it's 
    going to collide with the wall and just stop instead.

    Returns a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = []
    Xs = []
    costs = []
    for _ in range(NUM_SAMPLES):
        u_tape = sample_control_tape(u_guess, data)
        x_tape = rollout(x0, u_tape, data)
        Us.append(u_tape)
        Xs.append(x_tape)
        costs.append(compute_trajectory_cost(x_tape, u_tape, data))

    # Compute the weights
    costs = np.array(costs)
    min_cost = np.min(costs)
    weights = np.exp(-(costs-min_cost) / data.temperature)
    weights /= np.sum(weights)

    # Compute the new control tape
    u_nom = np.zeros(u_guess.shape)
    for u_tape, weight in zip(Us, weights):
        u_nom += weight * u_tape

    # Compute the new state trajectory
    x_nom = rollout(x0, u_nom)

    # Check if that new trajectory contains any collisions. If so, just stop.
    if contains_collisions(x_nom, data):
        print("Warning: new trajectory contains collisions. Stopping.")
        u_nom = np.zeros(u_guess.shape)
        x_nom = rollout(x0, u_nom)

    # Append the new control tape and state trajectory
    Us.append(u_nom)
    Xs.append(x_nom)

    return Us, Xs

def simulate(mppi=vanilla_mppi):
    """
    Run a quick little simulation with pygame. 
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
    data = ProblemData(x_nom=np.array([400, 250]), obstacles=obstacles)

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
        pygame.draw.circle(screen, (0, 255, 0), data.x_nom, 10)  # Target position

        # Perform an MPPI step
        Us, Xs = mppi(x, u_nom, data)

        # Visualize a few of the MPPI samples
        for i in range(min(len(Xs), 20)):
            x_traj = Xs[i]
            for t in range(len(x_traj)-1):
                pygame.draw.line(screen, (255, 0, 0), x_traj[t, :], x_traj[t+1,:], width=1)

        # Visualize the best trajectory with a thicker line
        x_star = Xs[-1]
        for t in range(len(x_star)-1):
            pygame.draw.line(screen, (0, 0, 255), x_star[t, :], x_star[t+1,:], width=3)

        # Update the nominal control tape
        u_nom = Us[-1]

        # Update the state
        x = robot_dynamics(x, u_nom[0], data)

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
    # mppi should be one of 
    #  - vanilla_mppi
    #  - just_stop_mppi
    #  - rejection_sample_mppi
    simulate(mppi=vanilla_mppi)
