import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -----------------------------
# 1) Straight line
# -----------------------------
x_line = np.linspace(0, 10, 200)
y_line = np.zeros_like(x_line)

# -----------------------------
# 2) Circular path
# -----------------------------
R = 3
theta = np.linspace(0, 2*np.pi, 300)
x_circle = 5 + R * np.cos(theta)
y_circle = 5 + R * np.sin(theta)

# -----------------------------
# 3) Spline path
# -----------------------------
waypoints_x = np.array([0, 1.5, 3.0, 5.0, 7.0, 9.0])
waypoints_y = np.array([0, 1.0, -0.5, 2.0, 1.0, 3.0])

t_wp = np.arange(len(waypoints_x))
t_fine = np.linspace(t_wp[0], t_wp[-1], 300)

cs_x = CubicSpline(t_wp, waypoints_x)
cs_y = CubicSpline(t_wp, waypoints_y)

x_spline = cs_x(t_fine)
y_spline = cs_y(t_fine)

# -----------------------------
# 4) N-Arc path
#   Build several circular arcs and concatenate them
# -----------------------------
def arc(center, radius, start_angle, end_angle, n=100):
    a = np.linspace(start_angle, end_angle, n)
    x = center[0] + radius * np.cos(a)
    y = center[1] + radius * np.sin(a)
    return x, y

x_arc1, y_arc1 = arc(center=(0, 0), radius=2, start_angle=0, end_angle=np.pi/2, n=80)
x_arc2, y_arc2 = arc(center=(2, 2), radius=2, start_angle=np.pi, end_angle=np.pi/2, n=80)
x_arc3, y_arc3 = arc(center=(4, 0), radius=2, start_angle=np.pi, end_angle=3*np.pi/2, n=80)

x_narc = np.concatenate([x_arc1, x_arc2, x_arc3])
y_narc = np.concatenate([y_arc1, y_arc2, y_arc3])

# -----------------------------
# 5) Corridor / detour path
#   Straight line with smooth bump in the middle
# -----------------------------
x_corr = np.linspace(0, 12, 300)
y_corr = 1.5 * np.exp(-0.5 * ((x_corr - 6)/1.2)**2)  # smooth detour bump

# -----------------------------
# Plot all in separate subplots
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.ravel()

# Straight line
axes[0].plot(x_line, y_line, 'b', linewidth=2)
axes[0].set_title('1) Straight Line Path')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True)
axes[0].axis('equal')

# Circle
axes[1].plot(x_circle, y_circle, 'r', linewidth=2)
axes[1].set_title('2) Circular Path')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True)
axes[1].axis('equal')

# Spline
axes[2].plot(x_spline, y_spline, 'g', linewidth=2, label='Spline path')
axes[2].scatter(waypoints_x, waypoints_y, c='k', s=25, label='Waypoints')
axes[2].set_title('3) Spline Path')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].grid(True)
axes[2].legend()
axes[2].axis('equal')

# N-Arc
axes[3].plot(x_narc, y_narc, 'm', linewidth=2)
axes[3].set_title('4) N-Arc Path')
axes[3].set_xlabel('X')
axes[3].set_ylabel('Y')
axes[3].grid(True)
axes[3].axis('equal')

# Corridor detour
axes[4].plot(x_corr, y_corr, 'c', linewidth=2)
axes[4].set_title('5) Corridor / Smooth Detour Path')
axes[4].set_xlabel('X')
axes[4].set_ylabel('Y')
axes[4].grid(True)
axes[4].axis('equal')

# Hide the last empty subplot
axes[5].axis('off')

plt.tight_layout()
plt.show()
