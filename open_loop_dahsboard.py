"""Forward kinematics simulation for a differential drive mobile robot.

Part 1: Batch simulation (odeint) + two static figures
Part 2: Real-time interactive simulation with sidebar controls

Author: Amir
Date: 2026-07-15
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, TextBox, Button
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# ─────────────────────────── Robot physical parameters ───────────────────────
WHEEL_RADIUS = 15.0                  # Wheel radius
WHEEL_BASE   = 4 * WHEEL_RADIUS      # Distance between wheel centers

# ─────────────────────────── Simulation configuration ────────────────────────
SIMULATION_TIME = 100.0
NUM_SAMPLES     = 10000
time_steps = np.linspace(0.0, SIMULATION_TIME, NUM_SAMPLES)

# Input wheel angular velocity profiles (rad/s)
left_wheel_speeds  = 2.0 * np.ones(time_steps.shape)
right_wheel_speeds = 1.4 * np.ones(time_steps.shape)


def robot_kinematics_model(state, t, time_vector, wheel_base, wheel_radius,
                           left_speed_profile, right_speed_profile):
    """Calculates time derivatives of the robot pose (x, y, theta)."""
    x, y, theta = state

    omega_left  = np.interp(t, time_vector, left_speed_profile)
    omega_right = np.interp(t, time_vector, right_speed_profile)

    v_left  = wheel_radius * omega_left
    v_right = wheel_radius * omega_right

    linear_velocity  = (v_right + v_left) / 2.0
    angular_velocity = (v_right - v_left) / wheel_base

    return [linear_velocity * np.cos(theta),
            linear_velocity * np.sin(theta),
            angular_velocity]


# ═══════════════════ PART 1: Batch simulation + static plots ═════════════════
initial_pose = [500.0, 500.0, 0.0]

pose_trajectory = odeint(
    robot_kinematics_model, initial_pose, time_steps,
    args=(time_steps, WHEEL_BASE, WHEEL_RADIUS,
          left_wheel_speeds, right_wheel_speeds),
)

np.save('simulationData.npy', pose_trajectory)

x_coords     = pose_trajectory[:, 0]
y_coords     = pose_trajectory[:, 1]
theta_coords = pose_trajectory[:, 2]

# ── Figure 1: x(t), y(t), θ(t) with dashed reference line for initial θ ──────
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(time_steps, x_coords,     'b', label='x')
ax1.plot(time_steps, y_coords,     'r', label='y')
ax1.plot(time_steps, theta_coords, 'm', label='θ')
ax1.axhline(y=initial_pose[2], color='k', linestyle='--',
            linewidth=1.5, label='θ_reference')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('x, y, θ')
ax1.set_title('Robot States vs Time')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.savefig('simulationResult.png', dpi=600)

# ── Figure 2: 2-D trajectory with EQUAL axis lengths (circle stays a circle) ─
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.plot(x_coords, y_coords, 'g', linewidth=1.5, label='trajectory')
ax2.plot(x_coords[0],  y_coords[0],  'ko', label='start')
ax2.plot(x_coords[-1], y_coords[-1], 'rs', label='end')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Robot 2D Trajectory')
ax2.set_aspect('equal', adjustable='datalim')   # ← جلوگیری از بیضی شدن دایره
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.savefig('trajectoryResult.png', dpi=600)

plt.show()


# ═══════════════════ PART 2: Real-time interactive simulation ════════════════
class RealTimeSim:
    """Euler-integrated real-time simulation state."""

    DT = 0.05  # integration / animation time step (s)

    def __init__(self):
        self.omega_l = 2.0
        self.omega_r = 1.4
        self.start_pose = [500.0, 500.0, 0.0]
        self.goal = [1200.0, 900.0]
        self.running = False
        self.reset()

    def reset(self):
        self.x, self.y, self.theta = self.start_pose
        self.t = 0.0
        self.t_hist  = [0.0]
        self.x_hist  = [self.x]
        self.y_hist  = [self.y]
        self.th_hist = [self.theta]

    def step(self):
        v = WHEEL_RADIUS * (self.omega_r + self.omega_l) / 2.0
        w = WHEEL_RADIUS * (self.omega_r - self.omega_l) / WHEEL_BASE
        self.x     += v * np.cos(self.theta) * self.DT
        self.y     += v * np.sin(self.theta) * self.DT
        self.theta += w * self.DT
        self.t     += self.DT
        self.t_hist.append(self.t)
        self.x_hist.append(self.x)
        self.y_hist.append(self.y)
        self.th_hist.append(self.theta)

    def at_goal(self, tol=20.0):
        return np.hypot(self.x - self.goal[0], self.y - self.goal[1]) < tol


sim = RealTimeSim()

# ── Figure layout: left = arena + live plots, right = sidebar ─────────────────
fig = plt.figure(figsize=(15, 8))
fig.canvas.manager.set_window_title('Real-Time Differential Drive Simulation')

gs = gridspec.GridSpec(2, 1, figure=fig,
                       left=0.06, right=0.64, top=0.95, bottom=0.08,
                       hspace=0.35, height_ratios=[2, 1])
ax_arena = fig.add_subplot(gs[0])   # بالا-چپ: حرکت ربات
ax_live  = fig.add_subplot(gs[1])   # پایین-چپ: نمودار زنده x, y, θ

ax_arena.set_title('Robot Arena')
ax_arena.set_xlabel('x')
ax_arena.set_ylabel('y')
ax_arena.set_aspect('equal', adjustable='datalim')
ax_arena.grid(True, alpha=0.3)

ax_live.set_title('Live states: x(t), y(t), θ(t)')
ax_live.set_xlabel('time (s)')
ax_live.grid(True, alpha=0.3)

# Arena artists
(trail_line,) = ax_arena.plot([], [], 'g-', lw=1.2, label='path')
(robot_dot,)  = ax_arena.plot([], [], 'bo', ms=10, label='robot')
(heading_ln,) = ax_arena.plot([], [], 'b-', lw=2)
(goal_dot,)   = ax_arena.plot([], [], 'r*', ms=15, label='goal')
ax_arena.legend(loc='upper right', fontsize=8)

# Live plot artists
(lx,)  = ax_live.plot([], [], 'b', label='x')
(ly,)  = ax_live.plot([], [], 'r', label='y')
(lth,) = ax_live.plot([], [], 'm', label='θ')
ax_live.legend(loc='upper left', fontsize=8)

# ─────────────────────────────── Sidebar (right) ─────────────────────────────
SB_L, SB_W = 0.70, 0.24

fig.text(SB_L + SB_W / 2, 0.96, 'Controls',
         fontsize=12, fontweight='bold', ha='center')

# — Wheel speeds: slider + numeric TextBox (هم اسلایدر، هم ورود عدد) —
ax_sl_l = fig.add_axes([SB_L, 0.89, SB_W * 0.62, 0.03])
sl_oml  = Slider(ax_sl_l, 'ωL ', -5.0, 5.0, valinit=sim.omega_l)
ax_tb_l = fig.add_axes([SB_L + SB_W * 0.72, 0.89, SB_W * 0.28, 0.03])
tb_oml  = TextBox(ax_tb_l, '', initial=f'{sim.omega_l:.2f}')

ax_sl_r = fig.add_axes([SB_L, 0.83, SB_W * 0.62, 0.03])
sl_omr  = Slider(ax_sl_r, 'ωR ', -5.0, 5.0, valinit=sim.omega_r)
ax_tb_r = fig.add_axes([SB_L + SB_W * 0.72, 0.83, SB_W * 0.28, 0.03])
tb_omr  = TextBox(ax_tb_r, '', initial=f'{sim.omega_r:.2f}')

# — Start pose x0, y0, θ0 —
fig.text(SB_L, 0.77, 'Start pose:', fontsize=9, fontweight='bold')
ax_x0 = fig.add_axes([SB_L + 0.03, 0.72, 0.055, 0.03])
tb_x0 = TextBox(ax_x0, 'x₀ ', initial='500')
ax_y0 = fig.add_axes([SB_L + 0.115, 0.72, 0.055, 0.03])
tb_y0 = TextBox(ax_y0, 'y₀ ', initial='500')
ax_th0 = fig.add_axes([SB_L + 0.20, 0.72, 0.055, 0.03])
tb_th0 = TextBox(ax_th0, 'θ₀ ', initial='0.0')

# — Goal point x_g, y_g —
fig.text(SB_L, 0.66, 'Goal point:', fontsize=9, fontweight='bold')
ax_xg = fig.add_axes([SB_L + 0.03, 0.61, 0.07, 0.03])
tb_xg = TextBox(ax_xg, 'x_g ', initial='1200')
ax_yg = fig.add_axes([SB_L + 0.14, 0.61, 0.07, 0.03])
tb_yg = TextBox(ax_yg, 'y_g ', initial='900')

# — Buttons: Start/Pause & Reset —
ax_btn_run = fig.add_axes([SB_L, 0.52, SB_W * 0.45, 0.05])
btn_run    = Button(ax_btn_run, 'Start / Pause', color='lightgreen')
ax_btn_rst = fig.add_axes([SB_L + SB_W * 0.55, 0.52, SB_W * 0.45, 0.05])
btn_rst    = Button(ax_btn_rst, 'Reset', color='lightcoral')

# — Robot specifications display —
fig.text(SB_L, 0.44, 'Robot specifications:', fontsize=9, fontweight='bold')
fig.text(SB_L + 0.01, 0.40, f'Wheel radius R = {WHEEL_RADIUS:.1f}', fontsize=9)
fig.text(SB_L + 0.01, 0.37, f'Wheel base   L = {WHEEL_BASE:.1f}', fontsize=9)

# — Live status readout —
status_text = fig.text(SB_L, 0.28, '', fontsize=9, family='monospace',
                       verticalalignment='top')

# ────────────────────────────── Callbacks ─────────────────────────────────────
def _sync_omega_l(val):
    sim.omega_l = sl_oml.val
    tb_oml.set_val(f'{sl_oml.val:.2f}')

def _sync_omega_r(val):
    sim.omega_r = sl_omr.val
    tb_omr.set_val(f'{sl_omr.val:.2f}')

def _tb_omega_l(text):
    try:
        v = float(text)
        sim.omega_l = v
        sl_oml.eventson = False
        sl_oml.set_val(np.clip(v, sl_oml.valmin, sl_oml.valmax))
        sl_oml.eventson = True
    except ValueError:
        pass

def _tb_omega_r(text):
    try:
        v = float(text)
        sim.omega_r = v
        sl_omr.eventson = False
        sl_omr.set_val(np.clip(v, sl_omr.valmin, sl_omr.valmax))
        sl_omr.eventson = True
    except ValueError:
        pass

sl_oml.on_changed(_sync_omega_l)
sl_omr.on_changed(_sync_omega_r)
tb_oml.on_submit(_tb_omega_l)
tb_omr.on_submit(_tb_omega_r)


def _apply_start_pose():
    try:
        sim.start_pose = [float(tb_x0.text),
                          float(tb_y0.text),
                          float(tb_th0.text)]
    except ValueError:
        pass

def _apply_goal():
    try:
        sim.goal = [float(tb_xg.text), float(tb_yg.text)]
    except ValueError:
        pass

for tb in (tb_x0, tb_y0, tb_th0):
    tb.on_submit(lambda _t: _apply_start_pose())
for tb in (tb_xg, tb_yg):
    tb.on_submit(lambda _t: _apply_goal())


def _toggle_run(event):
    sim.running = not sim.running

def _do_reset(event):
    _apply_start_pose()
    _apply_goal()
    sim.reset()
    sim.running = False
    trail_line.set_data([], [])
    lx.set_data([], [])
    ly.set_data([], [])
    lth.set_data([], [])
    fig.canvas.draw_idle()

btn_run.on_clicked(_toggle_run)
btn_rst.on_clicked(_do_reset)

# ─────────────────────────── Animation update loop ───────────────────────────
def _update(frame):
    if sim.running and not sim.at_goal():
        sim.step()

    # Arena: trail + robot + heading arrow + goal
    trail_line.set_data(sim.x_hist, sim.y_hist)
    robot_dot.set_data([sim.x], [sim.y])
    hl = 2.0 * WHEEL_RADIUS
    heading_ln.set_data([sim.x, sim.x + hl * np.cos(sim.theta)],
                        [sim.y, sim.y + hl * np.sin(sim.theta)])
    goal_dot.set_data([sim.goal[0]], [sim.goal[1]])

    ax_arena.relim()
    ax_arena.autoscale_view()

    # Live time-series plots
    lx.set_data(sim.t_hist, sim.x_hist)
    ly.set_data(sim.t_hist, sim.y_hist)
    lth.set_data(sim.t_hist, sim.th_hist)
    ax_live.relim()
    ax_live.autoscale_view()

    goal_flag = 'REACHED!' if sim.at_goal() else '---'
    status_text.set_text(
        f'Status\n'
        f'  t     = {sim.t:8.2f} s\n'
        f'  x     = {sim.x:8.2f}\n'
        f'  y     = {sim.y:8.2f}\n'
        f'  θ     = {sim.theta:8.3f} rad\n'
        f'  goal  : {goal_flag}'
    )
    return (trail_line, robot_dot, heading_ln, goal_dot,
            lx, ly, lth, status_text)


ani = FuncAnimation(fig, _update, interval=int(sim.DT * 1000),
                    blit=False, cache_frame_data=False)

plt.show()

