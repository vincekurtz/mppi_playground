##
#
# A basic MPPI implementation
#
##

import numpy as np
from typing import List
import pickle

from base import ProblemData, unicycle_dynamics

def sample_control_tape(u_nom: np.array, data: ProblemData) -> np.array:
    """
    Given the the nominal control u_nom, return a perturbed control tape that is
    sampled from a Gaussian distribution centered at u_nom.

    Args:
        u_nom: The nominal control tape.
        data: Problem data, including parameters, obstacles, target state, etc.

    Returns:
        u: The perturbed control tape.
    """
    u = np.zeros(u_nom.shape)

    for t in range(data.mppi_horizon - 1):
        u[t,:] = np.random.normal(u_nom[t,:], data.mppi_sample_variance)
        u[t,:] = np.clip(u[t,:], -data.mppi_u_max, data.mppi_u_max)
    return u


def rollout(x0: np.array,
            u_traj: np.array,
            data: ProblemData) -> np.array:
    """
    Given the initial state x0 and the control tape u_tape, return the
    resulting state trajectory.

    Args:
        x0: The initial state.
        u_traj: The control tape.
        data: Problem data, including parameters, obstacles, target state, etc.

    Returns:
        x_traj: The state trajectory.
    """
    x = x0
    x_traj = [x]
    for u in u_traj:
        x = unicycle_dynamics(x, u, data.mppi_dt)
        x_traj.append(x)
    return np.array(x_traj)

def smoothmin(a: float, b: float, smoothing_factor: float) -> float:
    """
    Given two values a and b, return a smooth approximation of min(a, b).
    """
    exp_a = np.exp(-a / smoothing_factor)
    exp_b = np.exp(-b / smoothing_factor)
    return -smoothing_factor * np.log(exp_a + exp_b)


def running_cost(x: np.array, u: np.array, data: ProblemData) -> float:
    """
    Compute the quadratic running cost of a state-control pair.

    Args:
        x: The state.
        u: The control.
        data: Problem data, including parameters, obstacles, target state, etc.

    Returns:
        cost: The running cost, including obstacle avoidance penalty
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


def compute_trajectory_cost(x_traj: np.array, u_traj: np.array, data: ProblemData) -> float:
    """
    Compute the total cost of a trajectory.

    Args:
        x_traj: The state trajectory.
        u_traj: The control tape.
        data: Problem data, including parameters, obstacles, target state, etc.

    Returns:
        cost: The total cost of the trajectory.
    """
    cost = 0.0
    for t in range(data.mppi_horizon - 1):
        cost += running_cost(x_traj[t,:], u_traj[t,:], data)
    # TODO: consider adding a terminal cost
    return cost

def contains_collisions(x_traj: np.array, data: ProblemData) -> bool:
    """
    Check if a given state trajectory contains any collisions.

    Args:
        x_traj: The state trajectory.
        data: Problem data, including parameters, obstacles, target state, etc.

    Returns:
        True if the trajectory contains a collision, False otherwise.
    """
    for x in x_traj:
        for obstacle in data.obstacles:
            if obstacle.contains(x):
                return True
    return False


def do_mppi_iteration(x0: np.array,
                      u_guess: np.array,
                      data: ProblemData,
                      ) -> (List[np.array], List[np.array]):
    """
    Given the initial state x0 and an initial guess for the control tape,
    perform an MPPI iteration to get a new control tape.

    Args:
        x0: The initial state.
        u_guess: The initial guess for the control tape.
        data: Problem data, including parameters, obstacles, target, dynamics, etc.

    Return a list of control tapes and a list of state trajectories, where the
    last element of each list is the best control tape and state trajectory.
    """
    assert data.sample_mppi or data.motion_primitives, "We need some sort of samples!"

    # Sample some trajectories
    Us = []
    Xs = []
    costs = []

    if data.sample_mppi:
        # Collect samples from MPPI
        for _ in range(data.mppi_num_samples):
            u_traj = sample_control_tape(u_guess, data)
            x_traj = rollout(x0, u_traj, data)
            if not data.reject_samples or not contains_collisions(x_traj, data):
                # Only keep the trajectory if it is collision-free, or if
                # rejection sampling is turned off
                Us.append(u_traj)
                Xs.append(x_traj)
                costs.append(compute_trajectory_cost(x_traj, u_traj, data))
    if data.motion_primitives:
        # Collect samples from the motion primitives
        with open("motion_primitives.pkl", 'rb') as f:
            # TODO: avoid loading from disk every iteration
            u_prim = pickle.load(f)

        for u_traj in u_prim:
            x_traj = rollout(x0, u_traj, data)
            if not data.reject_primitives or not contains_collisions(x_traj, data):
                Us.append(u_traj)
                Xs.append(x_traj)
                costs.append(compute_trajectory_cost(x_traj, u_traj, data))

    # Compute the weights
    costs = np.array(costs)
    min_cost = np.min(costs)
    weights = np.exp(-(costs-min_cost) / data.mppi_lambda)
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
