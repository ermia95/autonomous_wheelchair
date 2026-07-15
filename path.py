import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# =========================================================
# USER-SELECTABLE PARAMETERS
# =========================================================

# -----------------------------
# 1) Straight line parameters
# -----------------------------
LINE_X_START = 0.0
LINE_X_END = 10.0
LINE_Y_VALUE = 0.0
LINE_NUM_POINTS = 200

# -----------------------------
# 2) Circular path parameters
# -----------------------------
CIRCLE_RADIUS = 3.0
CIRCLE_CENTER_X = 5.0
CIRCLE_CENTER_Y = 5.0
CIRCLE_THETA_START = 0.0
CIRCLE_THETA_END = 2 * np.pi
CIRCLE_NUM_POINTS = 300

# -----------------------------
# 3) Spline path parameters
# -----------------------------
SPLINE_WAYPOINTS_X = np.array([0, 1.5, 3.0, 5.0, 7.0, 9.0], dtype=float)
SPLINE_WAYPOINTS_Y = np.array([0, 1.0, -0.5, 2.0, 1.0, 3.0], dtype=float)
SPLINE_NUM_POINTS = 300

# -----------------------------
# 4) N-Arc path parameters
# -----------------------------
NARC_1_CENTER = (0.0, 0.0)
NARC_1_RADIUS = 2.0
NARC_1_START_ANGLE = 0.0
NARC_1_END_ANGLE = np.pi / 2
NARC_1_NUM_POINTS = 80

NARC_2_CENTER = (2.0, 2.0)
NARC_2_RADIUS = 2.0
NARC_2_START_ANGLE = np.pi
NARC_2_END_ANGLE = np.pi / 2
NARC_2_NUM_POINTS = 80

NARC_3_CENTER = (4.0, 0.0)
NARC_3_RADIUS = 2.0
NARC_3_START_ANGLE = np.pi
NARC_3_END_ANGLE = 3 * np.pi / 2
NARC_3_NUM_POINTS = 80

# -----------------------------
# 5) Corridor / detour parameters
# -----------------------------
CORR_X_START = 0.0
CORR_X_END = 12.0
CORR_NUM_POINTS = 300
CORR_BUMP_AMPLITUDE = 1.5
CORR_BUMP_CENTER = 6.0
CORR_BUMP_WIDTH = 1.2


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def arc(center, radius, start_angle, end_angle, n=100):
    a = np.linspace(start_angle, end_angle, n)
    x = center[0] + radius * np.cos(a)
    y = center[1] + radius * np.sin(a)
    return x, y


def compute_path_outputs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)
    ds_segments = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds_segments)))

    dx_ds = np.gradient(x, s, edge_order=2)
    dy_ds = np.gradient(y, s, edge_order=2)

    theta = np.unwrap(np.arctan2(dy_ds, dx_ds))

    d2x_ds2 = np.gradient(dx_ds, s, edge_order=2)
    d2y_ds2 = np.gradient(dy_ds, s, edge_order=2)

    denom = (dx_ds**2 + dy_ds**2) ** 1.5
    kappa = np.divide(
        dx_ds * d2y_ds2 - dy_ds * d2x_ds2,
        denom,
        out=np.zeros_like(denom),
        where=denom > 1e-12
    )

    return {
        "x": x,
        "y": y,
        "s": s,
        "theta": theta,
        "kappa": kappa
    }


def save_path_to_csv(path_name, path_data):
    file_name = f"{path_name}_path_data.csv"
    data = np.column_stack((
        path_data["x"],
        path_data["y"],
        path_data["s"],
        path_data["theta"],
        path_data["kappa"]
    ))

    header = "x,y,s,theta,kappa"
    np.savetxt(file_name, data, delimiter=",", header=header, comments="")
    print(f"Saved: {file_name}")


def parse_paths_to_save(user_text, valid_names):
    user_text = user_text.strip().lower()

    if user_text == "all":
        return valid_names

    selected = [item.strip() for item in user_text.split(",") if item.strip()]
    selected = list(dict.fromkeys(selected))

    invalid = [name for name in selected if name not in valid_names]
    if invalid:
        raise ValueError(
            "Invalid path name(s): " + ", ".join(invalid) +
            ". Valid names are: " + ", ".join(valid_names)
        )

    return selected


# =========================================================
# 1) Straight line
# =========================================================
x_line = np.linspace(LINE_X_START, LINE_X_END, LINE_NUM_POINTS)
y_line = np.full_like(x_line, LINE_Y_VALUE)

# =========================================================
# 2) Circular path
# =========================================================
theta_circle = np.linspace(CIRCLE_THETA_START, CIRCLE_THETA_END, CIRCLE_NUM_POINTS)
x_circle = CIRCLE_CENTER_X + CIRCLE_RADIUS * np.cos(theta_circle)
y_circle = CIRCLE_CENTER_Y + CIRCLE_RADIUS * np.sin(theta_circle)

# =========================================================
# 3) Spline path
# =========================================================
t_wp = np.arange(len(SPLINE_WAYPOINTS_X))
t_fine = np.linspace(t_wp[0], t_wp[-1], SPLINE_NUM_POINTS)

cs_x = CubicSpline(t_wp, SPLINE_WAYPOINTS_X)
cs_y = CubicSpline(t_wp, SPLINE_WAYPOINTS_Y)

x_spline = cs_x(t_fine)
y_spline = cs_y(t_fine)

# =========================================================
# 4) N-Arc path
# =========================================================
x_arc1, y_arc1 = arc(
    center=NARC_1_CENTER,
    radius=NARC_1_RADIUS,
    start_angle=NARC_1_START_ANGLE,
    end_angle=NARC_1_END_ANGLE,
    n=NARC_1_NUM_POINTS
)

x_arc2, y_arc2 = arc(
    center=NARC_2_CENTER,
    radius=NARC_2_RADIUS,
    start_angle=NARC_2_START_ANGLE,
    end_angle=NARC_2_END_ANGLE,
    n=NARC_2_NUM_POINTS
)

x_arc3, y_arc3 = arc(
    center=NARC_3_CENTER,
    radius=NARC_3_RADIUS,
    start_angle=NARC_3_START_ANGLE,
    end_angle=NARC_3_END_ANGLE,
    n=NARC_3_NUM_POINTS
)

x_narc = np.concatenate([x_arc1, x_arc2, x_arc3])
y_narc = np.concatenate([y_arc1, y_arc2, y_arc3])

# =========================================================
# 5) Corridor / detour path
# =========================================================
x_corr = np.linspace(CORR_X_START, CORR_X_END, CORR_NUM_POINTS)
y_corr = CORR_BUMP_AMPLITUDE * np.exp(
    -0.5 * ((x_corr - CORR_BUMP_CENTER) / CORR_BUMP_WIDTH) ** 2
)

# =========================================================
# BUILD OUTPUT DATA FOR TRAJECTORY STAGE
# =========================================================
paths_data = {
    "straight": compute_path_outputs(x_line, y_line),
    "circle": compute_path_outputs(x_circle, y_circle),
    "spline": compute_path_outputs(x_spline, y_spline),
    "narc": compute_path_outputs(x_narc, y_narc),
    "corridor": compute_path_outputs(x_corr, y_corr)
}

# =========================================================
# PLOT ALL IN SEPARATE SUBPLOTS
# =========================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.ravel()

# Straight line
axes[0].plot(x_line, y_line, "b", linewidth=2)
axes[0].set_title("1) Straight Line Path")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid(True)
axes[0].axis("equal")

# Circle
axes[1].plot(x_circle, y_circle, "r", linewidth=2)
axes[1].set_title("2) Circular Path")
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].grid(True)
axes[1].axis("equal")

# Spline
axes[2].plot(x_spline, y_spline, "g", linewidth=2, label="Spline path")
axes[2].scatter(SPLINE_WAYPOINTS_X, SPLINE_WAYPOINTS_Y, c="k", s=25, label="Waypoints")
axes[2].set_title("3) Spline Path")
axes[2].set_xlabel("X")
axes[2].set_ylabel("Y")
axes[2].grid(True)
axes[2].legend()
axes[2].axis("equal")

# N-Arc
axes[3].plot(x_narc, y_narc, "m", linewidth=2)
axes[3].set_title("4) N-Arc Path")
axes[3].set_xlabel("X")
axes[3].set_ylabel("Y")
axes[3].grid(True)
axes[3].axis("equal")

# Corridor detour
axes[4].plot(x_corr, y_corr, "c", linewidth=2)
axes[4].set_title("5) Corridor / Smooth Detour Path")
axes[4].set_xlabel("X")
axes[4].set_ylabel("Y")
axes[4].grid(True)
axes[4].axis("equal")

# Hide the last empty subplot
axes[5].axis("off")

plt.tight_layout()
plt.show()
# =========================================================
# ASK USER WHICH PATHS TO SAVE
# =========================================================
valid_path_names = ["straight", "circle", "spline", "narc", "corridor"]

print("\nAvailable paths to save:")
for i, name in enumerate(valid_path_names, start=1):
    print(f"{i}) {name}")
print("0) all")

user_choice = input("Enter path number(s) separated by commas: ").strip()


def parse_path_numbers(user_text, valid_names):
    user_text = user_text.strip().lower()

    if user_text == "0":
        return valid_names

    items = [item.strip() for item in user_text.split(",") if item.strip()]

    try:
        indices = [int(item) for item in items]
    except ValueError:
        raise ValueError("Please enter only numbers separated by commas.")

    invalid = [idx for idx in indices if idx < 1 or idx > len(valid_names)]
    if invalid:
        raise ValueError(
            f"Invalid number(s): {invalid}. Choose between 1 and {len(valid_names)}, or 0 for all."
        )

    selected = [valid_names[idx - 1] for idx in indices]
    selected = list(dict.fromkeys(selected))  # remove duplicates, keep order
    return selected


try:
    selected_paths = parse_path_numbers(user_choice, valid_path_names)
    for path_name in selected_paths:
        save_path_to_csv(path_name, paths_data[path_name])
except ValueError as error:
    print(error)

