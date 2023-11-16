##
#
# The most basic MPPI algorithm.
#
##

import numpy as np
from typing import List

from base import ProblemData, rollout, sample_control_tape

def smoothmin(a: float, b: float, smoothing_factor: float) -> float:
    """
    Given two values a and b, return a smooth approximation of min(a, b).
    """
    exp_a = np.exp(-a / smoothing_factor)
    exp_b = np.exp(-b / smoothing_factor)
    return -smoothing_factor * np.log(exp_a + exp_b)


def compute_cost(x: np.array, u: np.array, data: ProblemData) -> float:
    """
    Given the state x and control, compute the running cost.
    """
    x_err = x - data.x_nom
    state_cost = x_err.T @ np.diag(data.state_cost) @ x_err
    control_cost = u.T @ np.diag(data.control_cost) @ u

    obstacle_cost = 0
    for obstacle in data.obstacles:
        phi = obstacle.signed_distance(x)
        c = -smoothmin(0, phi, data.obstacle_smoothing_factor)
        obstacle_cost += data.obstacle_cost * c**2

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


def vanilla_mppi(x0: np.array,
                 u_guess: np.array,
                 data: ProblemData,
                 ) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform MPPI to get a new control tape.

    Args:
        x0: The initial state.
        u_guess: The initial guess for the control tape.
        data: Problem data, including parameters, obstacles, target, dynamics, etc.

    Return a list of control tapes and a list of state trajectories, where the
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
