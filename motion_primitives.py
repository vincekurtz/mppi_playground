##
#
# A very simple planning algorithm based on a pre-defined set of motion
# primitives.
#
##

import numpy as np
from typing import List

from base import ProblemData, rollout
from vanilla_mppi import compute_trajectory_cost

def get_control_primitives(u_min: np.array, 
                           u_max: np.array,
                           num_samples: int, 
                           num_steps: int) -> np.array:
    """
    Generate a set of candidate control tapes by uniformly sampling zero order
    holds between u_min and u_max.
    """
    Us = []
    for _ in range(num_samples):
        u = np.zeros((num_steps, u_min.shape[0]))
        this_input = np.random.uniform(u_min, u_max)
        for t in range(num_steps):
            u[t,:] = this_input
        Us.append(u)
    return Us

PRIMITIVE_LIBRARY = get_control_primitives(np.array([-500, -50]),
                                             np.array([500, 50]),
                                             100,
                                             20)

def motion_primitive_planner(x0: np.array,
                 u_guess: np.array,
                 data: ProblemData,
                 ) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    choose the best motion primitive to get to the goal. 

    Args:
        x0: The initial state.
        u_guess: The initial guess for the control tape.
        data: Problem data, including parameters, obstacles, target, dynamics, etc.

    Return a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    # Sample some trajectories
    Us = PRIMITIVE_LIBRARY
    Xs = []
    costs = []
    for i in range(data.num_samples):
        u_tape = Us[i]
        x_tape = rollout(x0, u_tape, data)
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
