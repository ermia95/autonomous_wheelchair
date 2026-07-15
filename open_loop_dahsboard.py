"""Forward kinematics simulation for a differential drive mobile robot.

Author: Amir
Date: 2026-07-13

Original core: odeint batch simulation -> simulationData.npy -> offline plots.
Added: real-time animated window with sidebar controls
       (wheel velocities, start pose, goal point) + live x/y/theta plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, TextBox, Button
from scipy.integrate import odeint

# ============================================================================
# CORE (unchanged)
# ============================================================================

# Robot physical parameters
WHEEL_RADIUS = 15                 # Wheel radius
WHEEL_BASE = 4 * WHEEL_RADIUS     # Distance between wheel centers

# Simulation time configurations
SIMULATION_TIME = 100.0
NUM_SAMPLES = 10000
time_steps = np.linspace(0.0, SIMULATION_TIME, NUM_SAMPLES)

# Input wheel angular velocity profiles (rad/s)
left_wheel_speeds = 2.0 * np.ones(time_steps.shape)
right_wheel_speeds = 1.4 * np.ones(time_steps.shape)


# state[0]=x_dot, state[1]=y_dot, state[2]=theta_dot
def robot_kinematics_model(state, t, time_vector, wheel_base, wheel_radius,
                           left_speed_profile, right_speed_profile):
    """Calculates time derivatives of the robot pose (x, y, theta)."""
    x, y, theta = state

    omega_left = np.interp(t, time_vector, left_speed_profile)
    omega_right = np.interp(t, time_vector, right_speed_profile)

    v_left = wheel_radius * omega_left
    v_right = wheel_radius * omega_right

    linear_velocity = (v_right + v_left) / 2.0
    angular_velocity = (v_right - v_left) / wheel_base

    dx_dt = linear_velocity * np.cos(theta)
    dy_dt = linear_velocity * np.sin(theta)
    dtheta_dt = angular_velocity

    return [dx_dt, dy_dt, dtheta_dt]


# Initial robot pose [x, y, theta]
initial_pose = [500.0, 500.0, 0.0]

# Solve the ordinary differential equations (original batch run)
pose_trajectory = odeint(
    robot_kinematics_model, initial_pose, time_steps,
    args=(time_steps, WHEEL_BASE, WHEEL_RADIUS,
          left_wheel_speeds, right_wheel_speeds),
)

# save the simulation data
np.save('simulationData.npy', pose_trajectory)

# Extract trajectory coordinates (original offline plots)
x_traj = pose_trajectory[:, 0]
y_traj = pose_trajectory[:, 1]
theta_traj = pose_trajectory[:, 2]

fig0, axs0 = plt.subplots(2, 2, figsize=(10, 7))
fig0.suptitle('Offline Simulation Results (odeint)')
axs0[0, 0].plot(x_traj, y_traj); axs0[0, 0].set_title('Trajectory (x-y)')
axs0[0, 1].plot(time_steps, x_traj); axs0[0, 1].set_title('x(t)')
axs0[1, 0].plot(time_steps, y_traj); axs0[1, 0].set_title('y(t)')
axs0[1, 1].plot(time_steps, theta_traj); axs0[1, 1].set_title('theta(t)')
for a in axs0.flat:
    a.grid(True)
fig0.tight_layout()

# ============================================================================
# REAL-TIME SIMULATION WINDOW (new)
# ============================================================================

DT = 0.05                # real-time integration step (s)
GOAL_TOLERANCE = 25.0    # distance to consider goal reached


class RealTimeSim:
    def __init__(self):
        self.state = np.array(initial_pose, dtype=float)
        self.t = 0.0
        self.omega_l = 2.0
        self.omega_r = 1.4
        self.goal = np.array([2000.0, 2000.0])
        self.running = True
        self.hist_t, self.hist_x, self.hist_y, self.hist_th = [], [], [], []

    def step(self):
        if not self.running:
            return
        # RK4 step with constant (current slider) wheel speeds
        def f(s):
            _, _, th = s
            v = WHEEL_RADIUS * (self.omega_r + self.omega_l) / 2.0
            w = WHEEL_RADIUS * (self.omega_r - self.omega_l) / WHEEL_BASE
            return np.array([v * np.cos(th), v * np.sin(th), w])

        s = self.state
        k1 = f(s)
        k2 = f(s + 0.5 * DT * k1)
        k3 = f(s + 0.5 * DT * k2)
        k4 = f(s + DT * k3)
        self.state = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += DT

        self.hist_t.append(self.t)
        self.hist_x.append(self.state[0])
        self.hist_y.append(self.state[1])
        self.hist_th.append(self.state[2])

        if np.hypot(*(self.state[:2] - self.goal)) < GOAL_TOLERANCE:
            self.running = False   # goal reached

    def reset(self, x0, y0, th0):
        self.state = np.array([x0, y0, th0], dtype=float)
        self.t = 0.0
        self.hist_t, self.hist_x, self.hist_y, self.hist_th = [], [], [], []
        self.running = True


sim = RealTimeSim()

# ---- Figure layout: arena top-left, 3 plots bottom-left, sidebar right ----
fig = plt.figure(figsize=(15, 8))
fig.suptitle('Real-Time Differential Drive Simulation')

ax_arena = fig.add_axes([0.06, 0.42, 0.55, 0.50])
ax_x     = fig.add_axes([0.06, 0.28, 0.55, 0.09])
ax_y     = fig.add_axes([0.06, 0.16, 0.55, 0.09])
ax_th    = fig.add_axes([0.06, 0.04, 0.55, 0.09])

ax_arena.set_aspect('equal')
ax_arena.grid(True, alpha=0.3)
ax_arena.set_xlabel('x'); ax_arena.set_ylabel('y')
ax_x.set_ylabel('x');  ax_x.grid(True, alpha=0.3)
ax_y.set_ylabel('y');  ax_y.grid(True, alpha=0.3)
ax_th.set_ylabel(r'$\theta$'); ax_th.set_xlabel('t (s)'); ax_th.grid(True, alpha=0.3)

(path_line,)  = ax_arena.plot([], [], 'b-', lw=1.2, label='path')
(robot_dot,)  = ax_arena.plot([], [], 'ko', ms=8)
(heading_ln,) = ax_arena.plot([], [], 'r-', lw=2)
(goal_mark,)  = ax_arena.plot(sim.goal[0], sim.goal[1], 'g*', ms=16, label='goal')
ax_arena.legend(loc='upper right', fontsize=8)

(line_x,)  = ax_x.plot([], [], 'b')
(line_y,)  = ax_y.plot([], [], 'r')
(line_th,) = ax_th.plot([], [], 'm')

# ---- Sidebar widgets ----
ax_sl_l = fig.add_axes([0.72, 0.88, 0.22, 0.03])
ax_sl_r = fig.add_axes([0.72, 0.82, 0.22, 0.03])
sl_l = Slider(ax_sl_l, r'$\omega_L$', -5.0, 5.0, valinit=sim.omega_l)
sl_r = Slider(ax_sl_r, r'$\omega_R$', -5.0, 5.0, valinit=sim.omega_r)
sl_l.on_changed(lambda v: setattr(sim, 'omega_l', v))
sl_r.on_changed(lambda v: setattr(sim, 'omega_r', v))

tb_x0 = TextBox(fig.add_axes([0.78, 0.70, 0.14, 0.04]), 'start x0  ', initial='500')
tb_y0 = TextBox(fig.add_axes([0.78, 0.64, 0.14, 0.04]), 'start y0  ', initial='500')
tb_t0 = TextBox(fig.add_axes([0.78, 0.58, 0.14, 0.04]), 'start th0 ', initial='0')
tb_gx = TextBox(fig.add_axes([0.78, 0.48, 0.14, 0.04]), 'goal x  ', initial='2000')
tb_gy = TextBox(fig.add_axes([0.78, 0.42, 0.14, 0.04]), 'goal y  ', initial='2000')


def apply_goal(_=None):
    try:
        sim.goal = np.array([float(tb_gx.text), float(tb_gy.text)])
        goal_mark.set_data([sim.goal[0]], [sim.goal[1]])
        sim.running = True
    except ValueError:
        pass


def apply_reset(_=None):
    try:
        sim.reset(float(tb_x0.text), float(tb_y0.text), float(tb_t0.text))
        apply_goal()
    except ValueError:
        pass


btn_reset = Button(fig.add_axes([0.72, 0.30, 0.22, 0.05]), 'Reset / Apply')
btn_reset.on_clicked(apply_reset)

status_ax = fig.add_axes([0.72, 0.22, 0.22, 0.05])
status_ax.axis('off')
status_txt = status_ax.text(0.5, 0.5, 'Running…', ha='center', va='center',
                             transform=status_ax.transAxes, fontsize=10)

tb_gx.on_submit(apply_goal)
tb_gy.on_submit(apply_goal)


# ---- Animation update ----
def update(_frame):
    for _ in range(5):          # advance multiple steps per frame for speed
        sim.step()

    if len(sim.hist_x) < 2:
        return path_line, robot_dot, heading_ln, line_x, line_y, line_th

    x, y, th = sim.state
    ht, hx, hy, hth = sim.hist_t, sim.hist_x, sim.hist_y, sim.hist_th

    # Arena
    path_line.set_data(hx, hy)
    robot_dot.set_data([x], [y])
    arrow_len = WHEEL_BASE * 1.5
    heading_ln.set_data([x, x + arrow_len * np.cos(th)],
                        [y, y + arrow_len * np.sin(th)])
    goal_mark.set_data([sim.goal[0]], [sim.goal[1]])

    margin = 200
    all_x = hx + [sim.goal[0]]; all_y = hy + [sim.goal[1]]
    ax_arena.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax_arena.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Time-series plots
    for ax, line, data, col in [
        (ax_x,  line_x,  hx,  'b'),
        (ax_y,  line_y,  hy,  'r'),
        (ax_th, line_th, hth, 'm'),
    ]:
        line.set_data(ht, data)
        ax.set_xlim(0, max(ht) + 1)
        ax.set_ylim(min(data) - 1, max(data) + 1)

    status_txt.set_text('Goal reached!' if not sim.running else
                        f't = {sim.t:.1f}s')

    return path_line, robot_dot, heading_ln, line_x, line_y, line_th


ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

plt.show()

