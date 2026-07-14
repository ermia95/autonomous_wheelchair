import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Robot Parameters
# -----------------------------
WHEEL_RADIUS = 15.0
WHEEL_BASE = 60.0

# -----------------------------
# Simulation Parameters
# -----------------------------
DT = 0.05
TOTAL_TIME = 100.0
MAX_STEPS = int(TOTAL_TIME / DT)

# -----------------------------
# Initial Pose and Goal
# -----------------------------
state = np.array([500.0, 500.0, 0.0])  # [x, y, theta]
goal = np.array([400.0, 800.0])

# -----------------------------
# Controller Gains
# -----------------------------
KV = 0.8
KW = 4.0

# -----------------------------
# Saturation Limits
# -----------------------------
V_MAX = 50.0
W_MAX = 2.0

# -----------------------------
# Rate Limits
# -----------------------------
MAX_ACCEL = 30.0
MAX_ALPHA = 4.0

# -----------------------------
# Stop Condition
# -----------------------------
DIST_TOLERANCE = 5.0

# -----------------------------
# Previous Commands
# -----------------------------
v_prev = 0.0
w_prev = 0.0

# -----------------------------
# Data Storage
# -----------------------------
x_history = []
y_history = []
theta_history = []

robot_reached_goal = False


def normalize_angle(angle):
    """
    Normalize angle to the range [-pi, pi].
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def point_controller(state, goal, v_prev, w_prev):
    """
    Point tracking controller for a differential-drive robot.
    Input:
        state = [x, y, theta]
        goal = [x_goal, y_goal]
    Output:
        v_limited, w_limited, omega_left, omega_right, distance_error
    """

    x, y, theta = state

    # Position error
    dx = goal[0] - x
    dy = goal[1] - y
    distance_error = np.sqrt(dx**2 + dy**2)

    # Desired heading angle
    desired_theta = np.arctan2(dy, dx)

    # Heading error
    angle_error = desired_theta - theta
    angle_error = normalize_angle(angle_error)

    # Raw P controller commands
    v_cmd = KV * distance_error
    w_cmd = KW * angle_error

    # Saturation
    v_cmd = np.clip(v_cmd, 0.0, V_MAX)
    w_cmd = np.clip(w_cmd, -W_MAX, W_MAX)

    # Rate limiting
    dv_max = MAX_ACCEL * DT
    dw_max = MAX_ALPHA * DT

    v_limited = v_prev + np.clip(v_cmd - v_prev, -dv_max, dv_max)
    w_limited = w_prev + np.clip(w_cmd - w_prev, -dw_max, dw_max)

    # Inverse kinematics: body velocity to wheel angular velocities
    omega_right = (v_limited + (w_limited * WHEEL_BASE / 2.0)) / WHEEL_RADIUS
    omega_left = (v_limited - (w_limited * WHEEL_BASE / 2.0)) / WHEEL_RADIUS

    return v_limited, w_limited, omega_left, omega_right, distance_error


def update_robot_state(state, v, w):
    """
    Euler integration for the differential-drive kinematic model.
    """

    x, y, theta = state

    x_new = x + v * np.cos(theta) * DT
    y_new = y + v * np.sin(theta) * DT
    theta_new = theta + w * DT

    theta_new = normalize_angle(theta_new)

    return np.array([x_new, y_new, theta_new])


# -----------------------------
# Matplotlib Figure Setup
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_title("Online Differential-Drive Robot Point Control")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.grid(True)
ax.axis("equal")

# Set plot limits
margin = 100
x_min = min(state[0], goal[0]) - margin
x_max = max(state[0], goal[0]) + margin
y_min = min(state[1], goal[1]) - margin
y_max = max(state[1], goal[1]) + margin

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Goal marker
ax.plot(goal[0], goal[1], "rx", markersize=14, label="Goal")

# Robot path
path_line, = ax.plot([], [], "b-", linewidth=2, label="Robot Path")

# Robot position
robot_point, = ax.plot([], [], "ko", markersize=8, label="Robot")

# Robot heading arrow
heading_arrow = ax.arrow(
    state[0],
    state[1],
    1.0,
    0.0,
    head_width=10,
    head_length=15,
    fc="green",
    ec="green"
)

# Text information
info_text = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top"
)

ax.legend()


def init_animation():
    path_line.set_data([], [])
    robot_point.set_data([], [])
    info_text.set_text("")
    return path_line, robot_point, info_text


def update_animation(frame):
    global state
    global v_prev
    global w_prev
    global heading_arrow
    global robot_reached_goal

    if robot_reached_goal:
        return path_line, robot_point, heading_arrow, info_text

    # Save current state
    x_history.append(state[0])
    y_history.append(state[1])
    theta_history.append(state[2])

    # Controller
    v, w, omega_left, omega_right, distance_error = point_controller(
        state,
        goal,
        v_prev,
        w_prev
    )

    # Stop condition
    if distance_error < DIST_TOLERANCE:
        v = 0.0
        w = 0.0
        omega_left = 0.0
        omega_right = 0.0
        robot_reached_goal = True

    # Update previous commands
    v_prev = v
    w_prev = w

    # Update robot state
    if not robot_reached_goal:
        state = update_robot_state(state, v, w)

    # Update path
    path_line.set_data(x_history, y_history)

    # Update robot point
    robot_point.set_data([state[0]], [state[1]])

    # Remove old arrow and draw new one
    heading_arrow.remove()

    arrow_length = 40.0
    dx_arrow = arrow_length * np.cos(state[2])
    dy_arrow = arrow_length * np.sin(state[2])

    heading_arrow = ax.arrow(
        state[0],
        state[1],
        dx_arrow,
        dy_arrow,
        head_width=10,
        head_length=15,
        fc="green",
        ec="green"
    )

    # Update information text
    info_text.set_text(
        f"Step: {frame}\n"
        f"x: {state[0]:.2f}\n"
        f"y: {state[1]:.2f}\n"
        f"theta: {state[2]:.2f} rad\n"
        f"distance error: {distance_error:.2f}\n"
        f"v: {v:.2f}\n"
        f"w: {w:.2f}\n"
        f"omega_left: {omega_left:.2f}\n"
        f"omega_right: {omega_right:.2f}"
    )

    return path_line, robot_point, heading_arrow, info_text


animation = FuncAnimation(
    fig,
    update_animation,
    frames=MAX_STEPS,
    init_func=init_animation,
    interval=DT * 1000,
    blit=False,
    repeat=False
)

plt.show()

