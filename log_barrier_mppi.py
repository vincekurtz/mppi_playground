##
#
# The most basic MPPI algorithm.
#
##

import numpy as np
from typing import List

from base import ProblemData, rollout, sample_control_tape


def compute_cost(x: np.array, u: np.array, data: ProblemData) -> float:
    """
    Given the state x and control, compute the running cost.
    """
    state_cost = np.linalg.norm(x - data.x_nom)**2
    control_cost = data.control_cost * np.linalg.norm(u)**2

    obstacle_cost = 0
    for obstacle in data.obstacles:
        phi = obstacle.signed_distance(x)
        rho = 10
        obstacle_cost += -rho * np.log(max(0.0,phi))

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


def log_barrier_mppi(x0: np.array,
                     u_guess: np.array,
                     data: ProblemData,
                     ) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform MPPI to get a new control tape.

    Use the log barrier method to handle the obstacle constraints.

    Args:
        x0: The initial state.
        u_guess: The initial guess for the control tape.
        data: Problem data, including parameters, obstacles, target, dynamics, etc.

    Return:
        A list of control tapes and a list of state trajectories, where the
        last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = []
    Xs = []
    costs = []
    for _ in range(data.num_samples-1):
        u_tape = sample_control_tape(u_guess, data)
        x_tape = rollout(x0, u_tape, data)
        Us.append(u_tape)
        Xs.append(x_tape)
        costs.append(compute_trajectory_cost(x_tape, u_tape, data))
    
    # Staying stationary is always a possibility: add it to the list of samples
    u_tape_stationary = np.zeros(u_guess.shape)
    x_tape_stationary = rollout(x0, u_tape_stationary, data)
    Us.append(u_tape_stationary)
    Xs.append(x_tape_stationary)
    costs.append(compute_trajectory_cost(x_tape_stationary, u_tape_stationary, data))

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
