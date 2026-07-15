import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Button
from matplotlib.gridspec import GridSpec


# =========================
# Global Parameters
# =========================

# Robot Parameters
WHEEL_RADIUS = 15.0
WHEEL_BASE = 60.0

# Simulation Parameters
DT = 0.05
TOTAL_TIME = 100.0
MAX_STEPS = int(TOTAL_TIME / DT)

# Initial Pose and Goal
state = np.array([500.0, 500.0, 0.0])  # [x, y, theta]
goal = np.array([400.0, 800.0])

# Controller Gains
KV = 0.8
KW = 4.0

# Saturation Limits
V_MAX = 50.0
W_MAX = 2.0

# Rate Limits
MAX_ACCEL = 30.0
MAX_ALPHA = 4.0

# Stop Condition
DIST_TOLERANCE = 5.0

# Previous Commands
v_prev = 0.0
w_prev = 0.0

# Data Storage
x_history = []
y_history = []
theta_history = []

robot_reached_goal = False


# =========================
# Utility Functions
# =========================

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def point_controller(state, goal, v_prev, w_prev):
    x, y, theta = state
    goal_x, goal_y = goal

    dx = goal_x - x
    dy = goal_y - y
    distance_error = np.sqrt(dx**2 + dy**2)

    target_theta = np.arctan2(dy, dx)
    heading_error = normalize_angle(target_theta - theta)

    v_desired = KV * distance_error
    w_desired = KW * heading_error

    v_desired = np.clip(v_desired, -V_MAX, V_MAX)
    w_desired = np.clip(w_desired, -W_MAX, W_MAX)

    dv = np.clip(v_desired - v_prev, -MAX_ACCEL * DT, MAX_ACCEL * DT)
    dw = np.clip(w_desired - w_prev, -MAX_ALPHA * DT, MAX_ALPHA * DT)

    v = v_prev + dv
    w = w_prev + dw

    omega_left = (2 * v - w * WHEEL_BASE) / (2 * WHEEL_RADIUS)
    omega_right = (2 * v + w * WHEEL_BASE) / (2 * WHEEL_RADIUS)

    return v, w, omega_left, omega_right, distance_error


def update_robot_state(state, v, w):
    x, y, theta = state

    x_new = x + v * np.cos(theta) * DT
    y_new = y + v * np.sin(theta) * DT
    theta_new = theta + w * DT

    theta_new = normalize_angle(theta_new)

    return np.array([x_new, y_new, theta_new])


# =========================
# Matplotlib Figure Setup
# =========================

fig = plt.figure(figsize=(11, 8))
gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

ax = fig.add_subplot(gs[0, 0])
sidebar_ax = fig.add_subplot(gs[0, 1])
sidebar_ax.axis("off")

ax.set_title("Online Differential-Drive Robot Point Control")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.grid(True)
ax.axis("equal")

margin = 100
x_min = min(state[0], goal[0]) - margin
x_max = max(state[0], goal[0]) + margin
y_min = min(state[1], goal[1]) - margin
y_max = max(state[1], goal[1]) + margin

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Goal marker
goal_marker, = ax.plot(
    [goal[0]],
    [goal[1]],
    "rx",
    markersize=14,
    label="Goal"
)

# Robot path
path_line, = ax.plot([], [], "b-", linewidth=2, label="Robot Path")

# Robot position
robot_point, = ax.plot([], [], "ko", markersize=8, label="Robot")

# Heading arrow
heading_arrow = ax.arrow(
    state[0],
    state[1],
    40.0 * np.cos(state[2]),
    40.0 * np.sin(state[2]),
    head_width=10,
    head_length=15,
    fc="green",
    ec="green"
)

info_text = ax.text(
    0.02,
    0.98,
    "",
    transform=ax.transAxes,
    va="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

ax.legend(loc="lower right")


# =========================
# Sidebar Title
# =========================

fig.text(
    0.895, 0.93, "Parameters",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold"
)


# =========================
# Sidebar Widgets
# =========================

ax_goal_x = fig.add_axes([0.82, 0.84, 0.15, 0.04])
tb_goal_x = TextBox(ax_goal_x, "Goal X", initial=str(goal[0]))

ax_goal_y = fig.add_axes([0.82, 0.78, 0.15, 0.04])
tb_goal_y = TextBox(ax_goal_y, "Goal Y", initial=str(goal[1]))

ax_kv = fig.add_axes([0.82, 0.70, 0.15, 0.04])
tb_kv = TextBox(ax_kv, "KV", initial=str(KV))

ax_kw = fig.add_axes([0.82, 0.64, 0.15, 0.04])
tb_kw = TextBox(ax_kw, "KW", initial=str(KW))

ax_vmax = fig.add_axes([0.82, 0.56, 0.15, 0.04])
tb_vmax = TextBox(ax_vmax, "V Max", initial=str(V_MAX))

ax_wmax = fig.add_axes([0.82, 0.50, 0.15, 0.04])
tb_wmax = TextBox(ax_wmax, "W Max", initial=str(W_MAX))

ax_tol = fig.add_axes([0.82, 0.42, 0.15, 0.04])
tb_tol = TextBox(ax_tol, "Tol", initial=str(DIST_TOLERANCE))

ax_apply = fig.add_axes([0.82, 0.30, 0.15, 0.05])
btn_apply = Button(ax_apply, "Apply")

ax_reset = fig.add_axes([0.82, 0.22, 0.15, 0.05])
btn_reset = Button(ax_reset, "Reset")


# =========================
# Animation Functions
# =========================

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

    x_history.append(state[0])
    y_history.append(state[1])
    theta_history.append(state[2])

    v, w, omega_left, omega_right, distance_error = point_controller(
        state,
        goal,
        v_prev,
        w_prev
    )

    if distance_error < DIST_TOLERANCE:
        v = 0.0
        w = 0.0
        omega_left = 0.0
        omega_right = 0.0
        robot_reached_goal = True

    v_prev = v
    w_prev = w

    if not robot_reached_goal:
        state = update_robot_state(state, v, w)

    path_line.set_data(x_history, y_history)
    robot_point.set_data([state[0]], [state[1]])

    try:
        heading_arrow.remove()
    except Exception:
        pass

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


# =========================
# Callbacks
# =========================

def apply_parameters(event):
    global goal, KV, KW, V_MAX, W_MAX, DIST_TOLERANCE
    global state, v_prev, w_prev, x_history, y_history, theta_history
    global robot_reached_goal

    try:
        new_goal_x = float(tb_goal_x.text)
        new_goal_y = float(tb_goal_y.text)
        KV = float(tb_kv.text)
        KW = float(tb_kw.text)
        V_MAX = float(tb_vmax.text)
        W_MAX = float(tb_wmax.text)
        DIST_TOLERANCE = float(tb_tol.text)

        goal = np.array([new_goal_x, new_goal_y])

        state = np.array([500.0, 500.0, 0.0])
        v_prev = 0.0
        w_prev = 0.0
        x_history.clear()
        y_history.clear()
        theta_history.clear()
        robot_reached_goal = False

        goal_marker.set_data([goal[0]], [goal[1]])

        margin = 100
        x_min = min(state[0], goal[0]) - margin
        x_max = max(state[0], goal[0]) + margin
        y_min = min(state[1], goal[1]) - margin
        y_max = max(state[1], goal[1]) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        fig.canvas.draw_idle()

    except ValueError:
        print("Invalid input in sidebar.")


def reset_simulation(event):
    global goal, KV, KW, V_MAX, W_MAX, DIST_TOLERANCE
    global state, v_prev, w_prev, x_history, y_history, theta_history
    global robot_reached_goal

    goal = np.array([400.0, 800.0])
    KV = 0.8
    KW = 4.0
    V_MAX = 50.0
    W_MAX = 2.0
    DIST_TOLERANCE = 5.0

    state = np.array([500.0, 500.0, 0.0])
    v_prev = 0.0
    w_prev = 0.0
    x_history.clear()
    y_history.clear()
    theta_history.clear()
    robot_reached_goal = False

    tb_goal_x.set_val(str(goal[0]))
    tb_goal_y.set_val(str(goal[1]))
    tb_kv.set_val(str(KV))
    tb_kw.set_val(str(KW))
    tb_vmax.set_val(str(V_MAX))
    tb_wmax.set_val(str(W_MAX))
    tb_tol.set_val(str(DIST_TOLERANCE))

    goal_marker.set_data([goal[0]], [goal[1]])

    margin = 100
    x_min = min(state[0], goal[0]) - margin
    x_max = max(state[0], goal[0]) + margin
    y_min = min(state[1], goal[1]) - margin
    y_max = max(state[1], goal[1]) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.canvas.draw_idle()


# Connect callbacks AFTER function definitions
btn_apply.on_clicked(apply_parameters)
btn_reset.on_clicked(reset_simulation)


# =========================
# Start Animation
# =========================

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

