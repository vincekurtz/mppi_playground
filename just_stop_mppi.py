##
#
# The simplest collision avoidance method: if MPPI indicates that the current
# path will collide with an obstacle, just stop.
#
##

import numpy as np
from typing import List

from .base import ProblemData, rollout, sample_control_tape
from .vanilla_mppi import compute_trajectory_cost
from .rejection_sample_mppi import contains_collisions


def just_stop_mppi(x0: np.array,
                   u_guess: np.array,
                   data: ProblemData
                   ) -> (List[np.array], List[np.array]):
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

    # Check if that new trajectory contains any collisions. If so, just stop.
    if contains_collisions(x_nom, data):
        print("Warning: new trajectory contains collisions. Stopping.")
        u_nom = np.zeros(u_guess.shape)
        x_nom = rollout(x0, u_nom, data)

    # Append the new control tape and state trajectory
    Us.append(u_nom)
    Xs.append(x_nom)

    return Us, Xs
