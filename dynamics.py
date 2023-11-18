##
#
# Robot dynamics definitions
#
##

import numpy as np

def unicycle_dynamics(x, u, dt):
    """
    Forward dynamics of the unicycle robot.

    Args:
        x: state vector [px, py, theta]
        u: control vector [v, omega]
        dt: time step

    Returns:
        x_next: next state vector
    """
    x_next = np.zeros(3)
    x_next[0] = x[0] + u[0] * np.cos(x[2]) * dt
    x_next[1] = x[1] + u[0] * np.sin(x[2]) * dt
    x_next[2] = x[2] + u[1] * dt
    return x_next
