"""Interactive initial-condition pickers for mechanics demos."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def _ensure_gui_backend() -> None:
    """Raise if the current Matplotlib backend cannot open interactive windows."""

    backend = matplotlib.get_backend().lower()
    noninteractive_backends = {
        "agg",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
    if backend in noninteractive_backends or backend.endswith("agg"):
        raise RuntimeError(
            "Interactive initial-condition picking requires a GUI Matplotlib backend. "
            "Remove MPLBACKEND=Agg and rerun with --pick-ic."
        )


def _pendulum_position(length: float, angle: float) -> tuple[float, float]:
    """Return the bob position for a pendulum angle."""

    return length * float(np.sin(angle)), -length * float(np.cos(angle))


def _double_pendulum_positions(
    length1: float,
    length2: float,
    angle1: float,
    angle2: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the first and second bob positions for a double pendulum."""

    x1, y1 = _pendulum_position(length1, angle1)
    dx2, dy2 = _pendulum_position(length2, angle2)
    return (x1, y1), (x1 + dx2, y1 + dy2)


def _angle_from_position(x: float, y: float) -> float:
    """Convert a Cartesian position into the pendulum angle convention."""

    return float(np.arctan2(x, -y))


def pick_pendulum_initial_conditions(
    *,
    length: float,
    theta0: float = 1.1,
    omega0: float = 0.0,
    omega_limits: tuple[float, float] = (-8.0, 8.0),
) -> tuple[float, float]:
    """Interactively pick simple-pendulum initial conditions."""

    _ensure_gui_backend()

    fig, ax_motion = plt.subplots(figsize=(6.0, 7.0))
    fig.suptitle("Simple pendulum setup")
    fig.subplots_adjust(bottom=0.30)

    radius = 1.2 * length
    ax_motion.set_xlim(-radius, radius)
    ax_motion.set_ylim(-radius, 0.35 * length)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Drag the bob, then click Run")
    ax_motion.grid(alpha=0.25)
    ax_motion.add_patch(
        plt.Circle((0.0, 0.0), length, fill=False, linestyle="--", color="0.75")
    )

    rod_line, = ax_motion.plot([], [], lw=2.5, color="tab:blue")
    bob = ax_motion.scatter([], [], s=160, color="tab:orange", zorder=3)
    info_text = ax_motion.text(
        0.03,
        0.97,
        "",
        transform=ax_motion.transAxes,
        va="top",
    )

    state = {
        "theta": float(theta0),
        "dragging": False,
    }

    omega_ax = fig.add_axes([0.17, 0.17, 0.66, 0.04])
    reset_ax = fig.add_axes([0.18, 0.07, 0.22, 0.06])
    run_ax = fig.add_axes([0.60, 0.07, 0.22, 0.06])

    omega_slider = Slider(
        omega_ax,
        "omega0",
        omega_limits[0],
        omega_limits[1],
        valinit=float(omega0),
    )
    reset_button = Button(reset_ax, "Reset")
    run_button = Button(run_ax, "Run")

    def update_pose() -> None:
        x, y = _pendulum_position(length, state["theta"])
        rod_line.set_data([0.0, x], [0.0, y])
        bob.set_offsets(np.array([[x, y]]))
        info_text.set_text(
            f"theta0 = {state['theta']:.3f} rad\nomega0 = {omega_slider.val:.3f} rad/s"
        )
        fig.canvas.draw_idle()

    def update_from_event(event) -> None:
        if event.inaxes != ax_motion or event.xdata is None or event.ydata is None:
            return
        state["theta"] = _angle_from_position(event.xdata, event.ydata)
        update_pose()

    def on_press(event) -> None:
        if event.inaxes != ax_motion or event.button != 1:
            return
        contains, _ = bob.contains(event)
        if not contains:
            return
        state["dragging"] = True
        update_from_event(event)

    def on_motion(event) -> None:
        if state["dragging"]:
            update_from_event(event)

    def on_release(_event) -> None:
        state["dragging"] = False

    def on_reset(_event) -> None:
        state["theta"] = float(theta0)
        omega_slider.reset()
        update_pose()

    def on_run(_event) -> None:
        plt.close(fig)

    omega_slider.on_changed(lambda _value: update_pose())
    reset_button.on_clicked(on_reset)
    run_button.on_clicked(on_run)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    update_pose()
    plt.show(block=True)

    return float(state["theta"]), float(omega_slider.val)


def pick_double_pendulum_initial_conditions(
    *,
    length1: float,
    length2: float,
    theta1_0: float = 1.4,
    theta2_0: float = 0.9,
    omega1_0: float = 0.0,
    omega2_0: float = 0.0,
    omega_limits: tuple[float, float] = (-10.0, 10.0),
) -> tuple[float, float, float, float]:
    """Interactively pick double-pendulum initial conditions."""

    _ensure_gui_backend()

    fig, ax_motion = plt.subplots(figsize=(6.5, 7.5))
    fig.suptitle("Double pendulum setup")
    fig.subplots_adjust(bottom=0.38)

    total_length = length1 + length2
    radius = 1.15 * total_length
    ax_motion.set_xlim(-radius, radius)
    ax_motion.set_ylim(-radius, 0.3 * total_length)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Drag either bob, then click Run")
    ax_motion.grid(alpha=0.25)

    upper_rod, = ax_motion.plot([], [], lw=2.5, color="tab:blue")
    lower_rod, = ax_motion.plot([], [], lw=2.5, color="tab:cyan")
    bob1 = ax_motion.scatter([], [], s=120, color="tab:orange", zorder=3)
    bob2 = ax_motion.scatter([], [], s=150, color="tab:red", zorder=4)
    trace_line, = ax_motion.plot([], [], lw=1.2, color="tab:red", alpha=0.25)
    info_text = ax_motion.text(
        0.03,
        0.97,
        "",
        transform=ax_motion.transAxes,
        va="top",
    )

    state = {
        "theta1": float(theta1_0),
        "theta2": float(theta2_0),
        "active": None,
        "trace_x": [],
        "trace_y": [],
    }

    omega1_ax = fig.add_axes([0.17, 0.22, 0.66, 0.04])
    omega2_ax = fig.add_axes([0.17, 0.14, 0.66, 0.04])
    reset_ax = fig.add_axes([0.18, 0.05, 0.22, 0.06])
    run_ax = fig.add_axes([0.60, 0.05, 0.22, 0.06])

    omega1_slider = Slider(
        omega1_ax,
        "omega1_0",
        omega_limits[0],
        omega_limits[1],
        valinit=float(omega1_0),
    )
    omega2_slider = Slider(
        omega2_ax,
        "omega2_0",
        omega_limits[0],
        omega_limits[1],
        valinit=float(omega2_0),
    )
    reset_button = Button(reset_ax, "Reset")
    run_button = Button(run_ax, "Run")

    def update_pose() -> None:
        (x1, y1), (x2, y2) = _double_pendulum_positions(
            length1,
            length2,
            state["theta1"],
            state["theta2"],
        )
        upper_rod.set_data([0.0, x1], [0.0, y1])
        lower_rod.set_data([x1, x2], [y1, y2])
        bob1.set_offsets(np.array([[x1, y1]]))
        bob2.set_offsets(np.array([[x2, y2]]))
        trace_line.set_data(state["trace_x"], state["trace_y"])
        info_text.set_text(
            "\n".join(
                [
                    f"theta1_0 = {state['theta1']:.3f} rad",
                    f"theta2_0 = {state['theta2']:.3f} rad",
                    f"omega1_0 = {omega1_slider.val:.3f} rad/s",
                    f"omega2_0 = {omega2_slider.val:.3f} rad/s",
                ]
            )
        )
        fig.canvas.draw_idle()

    def on_press(event) -> None:
        if event.inaxes != ax_motion or event.button != 1:
            return
        contains2, _ = bob2.contains(event)
        contains1, _ = bob1.contains(event)
        if contains2:
            state["active"] = "bob2"
        elif contains1:
            state["active"] = "bob1"

    def on_motion(event) -> None:
        if state["active"] is None:
            return
        if event.inaxes != ax_motion or event.xdata is None or event.ydata is None:
            return

        if state["active"] == "bob1":
            state["theta1"] = _angle_from_position(event.xdata, event.ydata)
        else:
            (x1, y1), _ = _double_pendulum_positions(
                length1,
                length2,
                state["theta1"],
                state["theta2"],
            )
            state["theta2"] = _angle_from_position(event.xdata - x1, event.ydata - y1)
            state["trace_x"].append(float(event.xdata))
            state["trace_y"].append(float(event.ydata))
        update_pose()

    def on_release(_event) -> None:
        state["active"] = None

    def on_reset(_event) -> None:
        state["theta1"] = float(theta1_0)
        state["theta2"] = float(theta2_0)
        state["trace_x"].clear()
        state["trace_y"].clear()
        omega1_slider.reset()
        omega2_slider.reset()
        update_pose()

    def on_run(_event) -> None:
        plt.close(fig)

    omega1_slider.on_changed(lambda _value: update_pose())
    omega2_slider.on_changed(lambda _value: update_pose())
    reset_button.on_clicked(on_reset)
    run_button.on_clicked(on_run)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    update_pose()
    plt.show(block=True)

    return (
        float(state["theta1"]),
        float(state["theta2"]),
        float(omega1_slider.val),
        float(omega2_slider.val),
    )
