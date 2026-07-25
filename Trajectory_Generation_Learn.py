import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline



# ================= Waypoints ==============================

waypoints = np.array([
    [0.0, 0.0],
    [1.0, 0.2],
    [2.0, 0.8],
    [3.0, 2.0],
    [4.5, 2.2],
    [6.0, 3.0],
    [7.0, 4.0],
])


x_waypoints = waypoints[:, 0]
y_waypoints = waypoints[:, 1]



plt.figure()
plt.plot(
    x_waypoints,
    y_waypoints,
    "o--",
    label="Raw waypoints",
)
plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.show()


# ==================== Remove Duplicate Points ============================

def remove_duplicate_points(points, min_distance=0.01):
    """
    حذف نقاط متوالی که فاصله آن‌ها بسیار کم است.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        نقاط مسیر.
    min_distance : float
        حداقل فاصله مجاز بین نقاط متوالی.

    Returns
    -------
    filtered_points : ndarray
        نقاط فیلترشده.
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        raise ValueError("At least two points are required.")

    filtered_points = [points[0]]

    for point in points[1:]:
        distance = np.linalg.norm(point - filtered_points[-1])

        if distance > min_distance:
            filtered_points.append(point)

    filtered_points = np.asarray(filtered_points)

    if len(filtered_points) < 2:
        raise ValueError("Not enough distinct points.")

    return filtered_points


waypoints = remove_duplicate_points(waypoints)



# =========================== پارامتر دهی waypoint ها بر اساس فاصبه =================================

def chord_length_parameterization(points):
    """
    پارامتردهی نقاط براساس فاصله تجمعی chord length.
    """
    delta_points = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(delta_points, axis=1)

    cumulative_distance = np.concatenate([
        [0.0],
        np.cumsum(segment_lengths),
    ])

    total_distance = cumulative_distance[-1]

    if total_distance <= 0.0:
        raise ValueError("Path length must be positive.")

    u = cumulative_distance / total_distance

    return u

u_waypoints = chord_length_parameterization(waypoints)

# =====================================   Cubic Spline ========================================================

spline_x = CubicSpline(
    u_waypoints,
    waypoints[:, 0],
    bc_type="natural",
)

spline_y = CubicSpline(
    u_waypoints,
    waypoints[:, 1],
    bc_type="natural",
)

u_dense = np.linspace(0.0, 1.0, 5000)

x_dense = spline_x(u_dense)
y_dense = spline_y(u_dense)

plt.figure(figsize=(8, 6))

plt.plot(
    waypoints[:, 0],
    waypoints[:, 1],
    "ro--",
    label="Raw waypoints",
)

plt.plot(
    x_dense,
    y_dense,
    "b",
    linewidth=2,
    label="Cubic spline",
)

plt.axis("equal")
plt.grid(True)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.title("Raw path and smoothed path")
plt.show()

# ================================= Arch length ============================================

dx_dense = np.diff(x_dense)
dy_dense = np.diff(y_dense)

ds_dense = np.sqrt(dx_dense**2 + dy_dense**2)

s_dense = np.concatenate([
    [0.0],
    np.cumsum(ds_dense),
])

path_length = s_dense[-1]

print("Path length:", path_length, "m")

# =============================== Resampling points based on the arc length ==================

ds_target = 0.02  # meter

number_of_samples = int(np.ceil(path_length / ds_target)) + 1

s_path = np.linspace(
    0.0,
    path_length,
    number_of_samples,
)


u_path = np.interp(
    s_path,
    s_dense,
    u_dense,
)

x_path = spline_x(u_path)
y_path = spline_y(u_path)

# ======================================= 1st and 2nd derivative of spline Calcualtion ========================

dx_du = spline_x(u_path, 1)
dy_du = spline_y(u_path, 1)

d2x_du2 = spline_x(u_path, 2)
d2y_du2 = spline_y(u_path, 2)

# =======================================  Path Angle Calcualtion ===================================================

theta_path = np.arctan2(dy_du, dx_du)

theta_path = np.unwrap(theta_path)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# ======================================= Path Curvature Calculation ==============================================

numerator = (
    dx_du * d2y_du2
    - dy_du * d2x_du2
)

denominator = (
    dx_du**2 + dy_du**2
) ** 1.5

epsilon = 1e-9

curvature_path = numerator / np.maximum(
    denominator,
    epsilon,
)

plt.figure(figsize=(9, 4))
plt.plot(s_path, curvature_path)
plt.grid(True)
plt.xlabel("Path distance s [m]")
plt.ylabel("Curvature kappa [1/m]")
plt.title("Path curvature")
plt.show()


# ===================================== Wheelchair Physical Limitation ==============================================

V_MAX = 1.0              # m/s
OMEGA_MAX = 0.8          # rad/s
A_ACCEL_MAX = 0.5        # m/s^2
A_DECEL_MAX = 0.7        # m/s^2
A_LATERAL_MAX = 0.6      # m/s^2

WHEEL_BASE = 0.60        # m
WHEEL_RADIUS = 0.16      # m
WHEEL_ANGULAR_MAX = 7.0  # rad/s



# ======================================   OMEGA_MAX Limitation ======================================================

abs_curvature = np.abs(curvature_path)

v_limit_omega = np.full_like(
    curvature_path,
    np.inf,
)

curved_mask = abs_curvature > 1e-6

v_limit_omega[curved_mask] = (
    OMEGA_MAX / abs_curvature[curved_mask]
)


# ========================== Lateral Acceleration Limitation ==========================================================

v_limit_lateral = np.full_like(
    curvature_path,
    np.inf,
)

v_limit_lateral[curved_mask] = np.sqrt(
    A_LATERAL_MAX / abs_curvature[curved_mask]
)

# ========================= Wheel Velocity Limitation ==================================================================

wheel_linear_max = (
    WHEEL_RADIUS * WHEEL_ANGULAR_MAX
)

right_factor = np.abs(
    1.0 + 0.5 * WHEEL_BASE * curvature_path
)

left_factor = np.abs(
    1.0 - 0.5 * WHEEL_BASE * curvature_path
)

largest_wheel_factor = np.maximum(
    right_factor,
    left_factor,
)

v_limit_wheel = (
    wheel_linear_max
    / np.maximum(largest_wheel_factor, 1e-9)
)

# =========================== Combination of Velocity Limitation ==============================

v_limit_body = np.full_like(
    curvature_path,
    V_MAX,
)

v_limit = np.minimum.reduce([
    v_limit_body,
    v_limit_omega,
    v_limit_lateral,
    v_limit_wheel,
])

plt.figure(figsize=(10, 5))

plt.plot(
    s_path,
    v_limit_body,
    label="Body speed limit",
)

plt.plot(
    s_path,
    v_limit_omega,
    label="Angular speed limit",
)

plt.plot(
    s_path,
    v_limit_lateral,
    label="Lateral acceleration limit",
)

plt.plot(
    s_path,
    v_limit_wheel,
    label="Wheel speed limit",
)

plt.plot(
    s_path,
    v_limit,
    "k",
    linewidth=2,
    label="Final limit",
)

plt.ylim(0.0, 1.5 * V_MAX)
plt.grid(True)
plt.xlabel("Path distance s [m]")
plt.ylabel("Allowed speed [m/s]")
plt.legend()
plt.title("Local speed limits")
plt.show()

# ============================== Acceleration Limit/ Forward Pass ======================================

v_profile = v_limit.copy()

# شروع از سکون
v_profile[0] = 0.0

for i in range(len(s_path) - 1):
    ds = s_path[i + 1] - s_path[i]

    reachable_speed = np.sqrt(
        v_profile[i]**2
        + 2.0 * A_ACCEL_MAX * ds
    )

    v_profile[i + 1] = min(
        v_profile[i + 1],
        reachable_speed,
    )


# =============================== Decceleration Limit / Backward pass ========================================

v_profile[-1] = 0.0

for i in range(len(s_path) - 2, -1, -1):
    ds = s_path[i + 1] - s_path[i]

    reachable_speed = np.sqrt(
        v_profile[i + 1]**2
        + 2.0 * A_DECEL_MAX * ds
    )

    v_profile[i] = min(
        v_profile[i],
        reachable_speed,
    )

plt.figure(figsize=(10, 5))

plt.plot(
    s_path,
    v_limit,
    "--",
    label="Local speed limit",
)

plt.plot(
    s_path,
    v_profile,
    linewidth=2,
    label="Feasible speed profile",
)

plt.grid(True)
plt.xlabel("Path distance s [m]")
plt.ylabel("Speed v [m/s]")
plt.legend()
plt.title("Velocity planning")
plt.show()


# ============================= Convert Local_based profile to time_based profile ========================

time_path = np.zeros_like(s_path)

for i in range(len(s_path) - 1):
    ds = s_path[i + 1] - s_path[i]

    speed_sum = (
        v_profile[i] + v_profile[i + 1]
    )

    if speed_sum < 1e-9:
        raise ValueError(
            "Two consecutive path points have zero speed."
        )

    dt_segment = 2.0 * ds / speed_sum

    time_path[i + 1] = (
        time_path[i] + dt_segment
    )
total_time = time_path[-1]

print("Total trajectory time:", total_time, "s")

# ======================================================== Sampling by fixed time step ============================

DT = 0.02  # 50 Hz

t_trajectory = np.arange(
    0.0,
    total_time,
    DT,
)

if (
    len(t_trajectory) == 0
    or t_trajectory[-1] < total_time
):
    t_trajectory = np.append(
        t_trajectory,
        total_time,
    )

s_trajectory = np.interp(
    t_trajectory,
    time_path,
    s_path,
)

# ========================================= Generate refrence Pose (Position + Orientation) ==============================


x_trajectory = np.interp(
    s_trajectory,
    s_path,
    x_path,
)

y_trajectory = np.interp(
    s_trajectory,
    s_path,
    y_path,
)

theta_trajectory = np.interp(
    s_trajectory,
    s_path,
    theta_path,
)

theta_wrapped = wrap_to_pi(
    theta_trajectory
)

# =================================== Linear and Angular Velocity Calcualtion ======================================
v_trajectory = np.interp(
    s_trajectory,
    s_path,
    v_profile,
)

curvature_trajectory = np.interp(
    s_trajectory,
    s_path,
    curvature_path,
)

omega_trajectory = (
    v_trajectory * curvature_trajectory
)

#====================================== Acceleration/Deceleration Calcualtion =======================================

# because the last sample may have shorter duration --> better to calcualte based on real time instead of interp

a_trajectory = np.gradient(
    v_trajectory,
    t_trajectory,
)

alpha_trajectory = np.gradient(
    omega_trajectory,
    t_trajectory,
)
# =========================================== Wheel Speed Limitation ===============================================
right_wheel_linear = (
    v_trajectory
    + 0.5 * WHEEL_BASE * omega_trajectory
)

left_wheel_linear = (
    v_trajectory
    - 0.5 * WHEEL_BASE * omega_trajectory
)

right_wheel_angular = (
    right_wheel_linear / WHEEL_RADIUS
)

left_wheel_angular = (
    left_wheel_linear / WHEEL_RADIUS
)

