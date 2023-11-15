##
#
# A simple MPPI variant that discards samples that collide with obstacles.
#
##

import numpy as np
from typing import List

from .base import ProblemData, rollout, sample_control_tape
from .vanilla_mppi import compute_trajectory_cost


def contains_collisions(x_traj: np.array, data: ProblemData) -> bool:
    """
    Given the state trajectory x_traj, check if it contains any collisions.
    """
    for x in x_traj:
        for obstacle in data.obstacles:
            if obstacle.contains(x):
                return True
    return False


def rejection_sample_mppi(x0: np.array,
                          u_guess: np.array,
                          data: ProblemData
                          ) -> (List[np.array], List[np.array]):
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
    while len(Us) < data.num_samples and rs_iters < 500:
        u_tape = sample_control_tape(u_guess, data)
        x_tape = rollout(x0, u_tape, data)

        # Check if the trajectory is collision-free. If so, keep it.
        if not contains_collisions(x_tape, data):
            Us.append(u_tape)
            Xs.append(x_tape)
            costs.append(compute_trajectory_cost(x_tape, u_tape, data))
        rs_iters += 1

    if len(Us) == 0:
        print("Warning: no collision-free samples found. Returning a zero control tape.")
        u = np.zeros(u_guess.shape)
        return [u], [rollout(x0, u, data)]

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
