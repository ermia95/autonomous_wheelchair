"""Forward kinematics simulation for a differential drive mobile robot.

Author: Amir
Date: 2026-07-13
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Robot physical parameters
WHEEL_RADIUS = 15  # Wheel radius 
WHEEL_BASE = 4*WHEEL_RADIUS  # Distance between wheel centers

# Simulation time configurations
SIMULATION_TIME = 100.0
NUM_SAMPLES = 10000
time_steps = np.linspace(0.0, SIMULATION_TIME, NUM_SAMPLES)

# Input wheel angular velocity profiles (rad/s)
left_wheel_speeds = 2.0 * np.ones(time_steps.shape)
right_wheel_speeds = 1.4 * np.ones(time_steps.shape)


# state[0]=x_dot, state[1]=y_dot, state[2]=theta_dot
def robot_kinematics_model(
    state,
    t,
    time_vector,
    wheel_base,
    wheel_radius,
    left_speed_profile,
    right_speed_profile,
):
    """Calculates time derivatives of the robot pose (x, y, theta)."""

    # Extract current state variables
    x, y, theta = state

    # Interpolate wheel velocities at the current time 
    omega_left  = np.interp(t,time_vector,left_speed_profile)
    omega_right = np.interp(t,time_vector,right_speed_profile)

    # Calculate wheel linear velocities
    v_left = wheel_radius * omega_left
    v_right = wheel_radius * omega_right

    # Compute overall robot linear and angular velocities
    linear_velocity = (v_right + v_left) / 2.0
    angular_velocity = (v_right - v_left) / wheel_base

    # Compute state derivatives
    dx_dt = linear_velocity * np.cos(theta)
    dy_dt = linear_velocity * np.sin(theta)
    dtheta_dt = angular_velocity
    
    state_derivative = [dx_dt,dy_dt,dtheta_dt]

    return state_derivative


# Initial robot pose [x, y, theta]
initial_pose = [500.0, 500.0, 0.0]

# Solve the ordinary differential equations
pose_trajectory = odeint(
    robot_kinematics_model,
    initial_pose,
    time_steps,
    args=(
        time_steps,
        WHEEL_BASE,
        WHEEL_RADIUS,
        left_wheel_speeds,
        right_wheel_speeds,
    ),
)

# save the simulation data
np.save('simulationData.npy', pose_trajectory)

# Extract trajectory coordinates
x_coords = pose_trajectory[:, 0]
y_coords = pose_trajectory[:, 1]
theta_coords = pose_trajectory[:, 2]

# Plot the 2D trajectory of the robot
plt.plot(time_steps, x_coords,'b',label='x')
plt.plot(time_steps, y_coords,'r',label='y')
plt.plot(time_steps, theta_coords,'m',label='θ')

# Reference line for initial theta
plt.axhline(y=initial_pose[2], color='k', linestyle='--', linewidth=1.5, label='θ_reference')

plt.xlabel('time')
plt.ylabel('x,y,θ')
plt.legend()
plt.savefig('simulationResult.png',dpi=600)
plt.show()

