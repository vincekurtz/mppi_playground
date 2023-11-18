#!/usr/bin/env python

##
#
# Tools for constructing a set of motion primitives for the unicycle robot.
#
##

import numpy as np
import matplotlib.pyplot as plt
import pickle

from dynamics import unicycle_dynamics

def get_trajectory(x_nom, num_steps, dt):
    """
    Compute a trajectory that takes the robot from the origin to x_nom
    (approximately) in num_steps steps. Uses the POSQ algorithm described
    in http://www.spencer.eu/papers/palmieriIROS14.pdf.

    Args:
        x_nom: final state vector  [px, py, theta]
        num_steps: number of steps to take
        dt: time step
    
    Returns:
        x_traj: the state trajectory
        u_traj: the corresponding control tape
    """
    x_traj = np.zeros((3, num_steps))
    u_traj = np.zeros((2, num_steps-1))
    x_traj[:, 0] = np.zeros(3)

    # Feedback gains
    K_v = 1
    K_rho = 2
    K_alpha = 3
    K_beta = -1

    assert K_alpha + 5/3 * K_beta - 2/np.pi * K_rho > 0, "Unstable gains"

    for t in range(num_steps-1):
        # Compute a control input that will drive the robot toward x_nom
        rho = np.linalg.norm(x_nom[:2] - x_traj[:2, t])
        alpha = normalize_angle(np.arctan2(x_nom[1] - x_traj[1, t], x_nom[0] - x_traj[0, t]) - x_traj[2, t])
        beta = normalize_angle(x_nom[2] - x_traj[2, t] - alpha)
        u_traj[0, t] = K_rho * np.tanh(rho * K_v)
        u_traj[1, t] = K_alpha * alpha + K_beta * beta

        # Simulate forward
        x_traj[:, t+1] = unicycle_dynamics(x_traj[:, t], u_traj[:, t], dt)

    return x_traj, u_traj

def normalize_angle(theta):
    """
    Normalize an angle to be in [-pi, pi].

    Args:
        theta: angle in radians
    
    Returns:
        theta: angle in radians, normalized to [-pi, pi]
    """
    while theta < -np.pi:
        theta += 2*np.pi
    while theta > np.pi:
        theta -= 2*np.pi
    return theta

def plot_trajectory(x_traj):
    """
    Plot the trajectory using matplotlib arrows.

    Args:
        x_traj: the state trajectory, size (3, num_steps)
    """
    N = x_traj.shape[1]
    for t in range(N):
        px = x_traj[0, t]
        py = x_traj[1, t]
        theta = x_traj[2, t]
        plt.arrow(px, py, 0.1*np.cos(theta), 0.1*np.sin(theta), head_width=0.05, head_length=0.1, fc='k', ec='k')

def generate_trajectories(radius, num_samples, num_steps, dt):
    """
    Generate a set of trajectories from the origin to uniformly spaced points
    on a circle of the given radius

    Args:
        radius: radius of the circle
        num_samples: number of trajectories to generate
        num_steps: number of steps in each trajectory
        dt: time step
    
    Returns:
        x_trajs: a list of state trajectories
        u_trajs: a list of control tapes
    """
    x_trajs = []
    u_trajs = []
    thetas = np.linspace(-np.pi, np.pi, num_samples)
    for i in range(num_samples):
        # Sample a target state
        theta = thetas[i]
        x_nom = np.array([radius*np.cos(theta), radius*np.sin(theta), theta])

        # Compute a trajectory to the target state 
        x_traj, u_traj = get_trajectory(x_nom, num_steps, dt)
        x_trajs.append(x_traj)
        u_trajs.append(u_traj)
    return x_trajs, u_trajs


if __name__=="__main__":
    # Set some parameters
    radius = 1.0
    num_samples = 20
    num_steps = 20
    dt = 0.1

    # Generate trajectories
    x_trajs, u_trajs = generate_trajectories(radius, num_samples, num_steps, dt)

    # Add a zero input option to the set of motion primitives
    u_trajs.append(np.zeros((2, num_steps-1)))

    # Save the primitives to a file
    fname = "motion_primitves.pkl"
    print(f"Saving motion primitives to {fname}")
    with open(fname, 'wb') as f:
        pickle.dump(u_trajs, f)  # MPPI only needs the control tapes

    # Plot the trajectories
    plt.figure()
    plt.axis('equal')

    for x_traj in x_trajs:
        plot_trajectory(x_traj)
    
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.show()
