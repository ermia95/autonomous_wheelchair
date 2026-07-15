"""
Forward kinematics simulation for a differential drive mobile robot.

Author: Amir
Date: 2026-07-13
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from scipy.integrate import odeint

# Robot physical parameters
WHEEL_RADIUS = 15
WHEEL_BASE = 4 * WHEEL_RADIUS

# Simulation time configurations
SIMULATION_TIME = 100.0
NUM_SAMPLES = 10000
time_steps = np.linspace(0.0, SIMULATION_TIME, NUM_SAMPLES)

# Input wheel angular velocity profiles (rad/s)
left_wheel_speeds = 2.0 * np.ones(time_steps.shape)
right_wheel_speeds = 1.4 * np.ones(time_steps.shape)


def robot_kinematics_model(
    state,
    t,
    time_vector,
    wheel_base,
    wheel_radius,
    left_speed_profile,
    right_speed_profile,
):
    """Calculate time derivatives of the robot pose (x, y, theta)."""

    # Extract current state variables
    x, y, theta = state

    # Interpolate wheel velocities at the current time
    omega_left = np.interp(t, time_vector, left_speed_profile)
    omega_right = np.interp(t, time_vector, right_speed_profile)

    # Calculate wheel linear velocities
    v_left = wheel_radius * omega_left
    v_right = wheel_radius * omega_right

    # Compute robot linear and angular velocities
    linear_velocity = (v_right + v_left) / 2.0
    angular_velocity = (v_right - v_left) / wheel_base

    # Compute state derivatives
    dx_dt = linear_velocity * np.cos(theta)
    dy_dt = linear_velocity * np.sin(theta)
    dtheta_dt = angular_velocity

    state_derivative = [dx_dt, dy_dt, dtheta_dt]
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

# Save the simulation data
np.save("simulationData.npy", pose_trajectory)

# Extract trajectory coordinates
x_coords = pose_trajectory[:, 0]
y_coords = pose_trajectory[:, 1]
theta_coords = pose_trajectory[:, 2]

# Plot x, y, and theta versus time
plt.figure(figsize=(10, 6))
plt.plot(time_steps, x_coords, "b", label="x")
plt.plot(time_steps, y_coords, "r", label="y")
plt.plot(time_steps, theta_coords, "m", label="theta")
plt.axhline(
    y=initial_pose[2],
    color="k",
    linestyle="--",
    linewidth=1.5,
    label="theta_reference",
)
plt.xlabel("time")
plt.ylabel("x, y, theta")
plt.legend()
plt.tight_layout()
plt.savefig("simulationResult.png", dpi=600)
plt.show()


def rotate_translate_shape(local_points, x_center, y_center, theta):
    """Rotate a local polygon and translate it to the global frame."""
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ]
    )
    global_points = local_points @ rotation_matrix.T
    global_points[:, 0] += x_center
    global_points[:, 1] += y_center
    return global_points


# Animation figure
fig, ax = plt.subplots(figsize=(8, 8))

margin = 100
ax.set_xlim(np.min(x_coords) - margin, np.max(x_coords) + margin)
ax.set_ylim(np.min(y_coords) - margin, np.max(y_coords) + margin)
ax.set_aspect("equal", adjustable="box")
ax.set_title("Differential Drive Robot Motion")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)

# Trajectory line
trajectory_line, = ax.plot([], [], "y-", linewidth=2, label="trajectory")

# Robot geometry in local coordinates
body_length = 60
body_width = 40
wheel_length = 16
wheel_width = 10
wheel_offset = body_width / 2 + wheel_width / 2

# Main body rectangle
body_local = np.array(
    [
        [-body_length / 2, -body_width / 2],
        [ body_length / 2, -body_width / 2],
        [ body_length / 2,  body_width / 2],
        [-body_length / 2,  body_width / 2],
    ]
)

# Left wheel rectangle
left_wheel_local = np.array(
    [
        [-wheel_length / 2, -wheel_width / 2 + wheel_offset],
        [ wheel_length / 2, -wheel_width / 2 + wheel_offset],
        [ wheel_length / 2,  wheel_width / 2 + wheel_offset],
        [-wheel_length / 2,  wheel_width / 2 + wheel_offset],
    ]
)

# Right wheel rectangle
right_wheel_local = np.array(
    [
        [-wheel_length / 2, -wheel_width / 2 - wheel_offset],
        [ wheel_length / 2, -wheel_width / 2 - wheel_offset],
        [ wheel_length / 2,  wheel_width / 2 - wheel_offset],
        [-wheel_length / 2,  wheel_width / 2 - wheel_offset],
    ]
)

# Front marker triangle
front_marker_local = np.array(
    [
        [body_length / 2 + 8, 0],
        [body_length / 2 - 8, 8],
        [body_length / 2 - 8, -8],
    ]
)

# Initial patches
body_patch = Polygon(
    rotate_translate_shape(body_local.copy(), x_coords[0], y_coords[0], theta_coords[0]),
    closed=True,
    facecolor="deepskyblue",
    edgecolor="navy",
    linewidth=2,
)

left_wheel_patch = Polygon(
    rotate_translate_shape(left_wheel_local.copy(), x_coords[0], y_coords[0], theta_coords[0]),
    closed=True,
    facecolor="dimgray",
    edgecolor="black",
    linewidth=1.5,
)

right_wheel_patch = Polygon(
    rotate_translate_shape(right_wheel_local.copy(), x_coords[0], y_coords[0], theta_coords[0]),
    closed=True,
    facecolor="dimgray",
    edgecolor="black",
    linewidth=1.5,
)

front_marker_patch = Polygon(
    rotate_translate_shape(front_marker_local.copy(), x_coords[0], y_coords[0], theta_coords[0]),
    closed=True,
    facecolor="crimson",
    edgecolor="darkred",
    linewidth=1.5,
)

ax.add_patch(body_patch)
ax.add_patch(left_wheel_patch)
ax.add_patch(right_wheel_patch)
ax.add_patch(front_marker_patch)
ax.legend()


def init():
    """Initialize animation objects."""
    trajectory_line.set_data([], [])
    return (
        trajectory_line,
        body_patch,
        left_wheel_patch,
        right_wheel_patch,
        front_marker_patch,
    )


def update(frame_index):
    """Update animation frame."""
    x = x_coords[frame_index]
    y = y_coords[frame_index]
    theta = theta_coords[frame_index]

    trajectory_line.set_data(x_coords[: frame_index + 1], y_coords[: frame_index + 1])

    body_patch.set_xy(rotate_translate_shape(body_local.copy(), x, y, theta))
    left_wheel_patch.set_xy(rotate_translate_shape(left_wheel_local.copy(), x, y, theta))
    right_wheel_patch.set_xy(rotate_translate_shape(right_wheel_local.copy(), x, y, theta))
    front_marker_patch.set_xy(rotate_translate_shape(front_marker_local.copy(), x, y, theta))

    return (
        trajectory_line,
        body_patch,
        left_wheel_patch,
        right_wheel_patch,
        front_marker_patch,
    )


animation = FuncAnimation(
    fig,
    update,
    frames=len(time_steps),
    init_func=init,
    interval=1,
    blit=True,
)

plt.show()

