##
#
# An MPPI algorithm that attempts to enforce collision avoidance constraints
# more strictly via an augmented Lagrangian.
#
##

import numpy as np
from typing import List
from .base import ProblemData, rollout, sample_control_tape
from .vanilla_mppi import smoothmin


def compute_cost(x: np.array, u: np.array, t: int, data: ProblemData) -> (float, np.array):
    """
    Given the state x and control, compute the running cost.

    Args:
        x: The state.
        u: The control.
        t: The time step
        data: Problem data, including parameters, obstacles, etc.

    Return:
        l: The running cost.
        c: The constraint function c(x) [ which we hope is <= 0 ]
    """
    state_cost = np.linalg.norm(x - data.x_nom)**2
    control_cost = data.control_cost * np.linalg.norm(u)**2

    obstacle_cost = 0
    c = np.zeros(len(data.obstacles))
    for i in range(len(data.obstacles)):
        obstacle = data.obstacles[i]
        phi = obstacle.signed_distance(x)
        c[i] = -phi

        # Penalty cost is quadratic in the constraint violation
        penalty_cost = data.obstacle_cost * smoothmin(0, phi, data.obstacle_smoothing_factor)**2

        # Lagrange multiplier term grows and shrinks with the constraint violation
        lmbda = data.lagrange_multipliers[i, t]
        lagrange_cost = lmbda * c[i]

        obstacle_cost += penalty_cost + lagrange_cost
    
    running_cost = state_cost + control_cost + obstacle_cost

    return running_cost, c


def compute_trajectory_cost(x_traj: np.array, 
                            u_tape: np.array, 
                            data: ProblemData) -> (float, np.array):
    """
    Given the state trajectory x_traj, the nominal state x_nom, and the
    control tape u_tape, return the total cost.

    Args:
        x_traj: The state trajectory.
        u_tape: The control tape.
        data: Problem data, including parameters, obstacles, etc.

    Return:
        cost: The total cost.
        c: The constraint function c(x) for each step and each obstacle
    """
    cost = 0.0
    c = np.zeros((len(data.obstacles), data.horizon))
    for t in range(len(u_tape)):
        x = x_traj[t]
        u = u_tape[t]
        running_cost, ct = compute_cost(x, u, t, data)
        cost += running_cost
        c[:, t] = ct
    return cost, c


def augmented_lagrangian_mppi(x0: np.array,
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
    cs = []
    for _ in range(data.num_samples):
        u_tape = sample_control_tape(u_guess, data)
        x_tape = rollout(x0, u_tape, data)
        Us.append(u_tape)
        Xs.append(x_tape)
        cost, c = compute_trajectory_cost(x_tape, u_tape, data)
        costs.append(cost)
        cs.append(c)

    # Compute the weights
    costs = np.array(costs)
    min_cost = np.min(costs)
    weights = np.exp(-(costs-min_cost) / data.temperature)
    weights /= np.sum(weights)

    # Compute the new control tape
    u_nom = np.zeros(u_guess.shape)
    for u_tape, weight in zip(Us, weights):
        u_nom += weight * u_tape

    # Update the Lagrange multiplier estimates
    lagrange_multipliers = []
    for i in range(data.num_samples):
        c = cs[i]
        lmbda = np.zeros(data.lagrange_multipliers.shape)
        for t in range(data.horizon):
            for j in range(len(data.obstacles)):
                mu = data.obstacle_cost
                if c[j, t] > data.lagrange_multipliers[j, t]/mu:
                    lmbda[j, t] = data.lagrange_multipliers[j,t] + mu * c[j, t]
        lagrange_multipliers.append(lmbda)

    data.lagrange_multipliers *= 0.0
    for lmbda, weight in zip(lagrange_multipliers, weights):
        data.lagrange_multipliers += weight * lmbda

    # Compute the new state trajectory
    x_nom = rollout(x0, u_nom, data)

    # Append the new control tape and state trajectory
    Us.append(u_nom)
    Xs.append(x_nom)

    return Us, Xs
