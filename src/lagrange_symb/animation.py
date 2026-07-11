"""Animation helpers for mechanics demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animate_pendulum(
    times: np.ndarray,
    angles: np.ndarray,
    angular_velocities: np.ndarray,
    *,
    length: float = 1.0,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Render a pendulum animation with a phase-space subplot."""

    times = np.asarray(times)
    angles = np.asarray(angles)
    angular_velocities = np.asarray(angular_velocities)

    x = length * np.sin(angles)
    y = -length * np.cos(angles)

    fig, (ax_motion, ax_phase) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Simple pendulum")

    ax_motion.set_xlim(-1.2 * length, 1.2 * length)
    ax_motion.set_ylim(-1.2 * length, 0.3 * length)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Motion")
    ax_motion.grid(alpha=0.25)

    ax_phase.set_title("Phase portrait")
    ax_phase.set_xlabel(r"$\theta$ (rad)")
    ax_phase.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax_phase.plot(angles, angular_velocities, color="tab:purple", alpha=0.3)
    ax_phase.grid(alpha=0.25)

    rod_line, = ax_motion.plot([], [], lw=2.5, color="tab:blue")
    bob = ax_motion.scatter([], [], s=100, color="tab:orange", zorder=3)
    trail_line, = ax_motion.plot([], [], lw=1.5, color="tab:orange", alpha=0.35)
    phase_point = ax_phase.scatter([], [], s=45, color="tab:red", zorder=3)
    time_text = ax_motion.text(
        0.04,
        0.94,
        "",
        transform=ax_motion.transAxes,
        va="top",
        fontsize=10,
    )

    def update(frame: int):
        rod_line.set_data([0.0, x[frame]], [0.0, y[frame]])
        bob.set_offsets(np.array([[x[frame], y[frame]]]))
        trail_line.set_data(x[: frame + 1], y[: frame + 1])
        phase_point.set_offsets(np.array([[angles[frame], angular_velocities[frame]]]))
        time_text.set_text(f"t = {times[frame]:.2f} s")
        return rod_line, bob, trail_line, phase_point, time_text

    frame_interval_ms = 1000 * float(np.mean(np.diff(times))) if len(times) > 1 else 20.0
    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=frame_interval_ms,
        blit=True,
    )

    if save_path:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1000.0 / frame_interval_ms)))
        animation.save(destination, writer="pillow", fps=fps)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def animate_double_pendulum(
    times: np.ndarray,
    angles1: np.ndarray,
    angles2: np.ndarray,
    angular_velocities1: np.ndarray,
    angular_velocities2: np.ndarray,
    *,
    length1: float = 1.0,
    length2: float = 1.0,
    save_path: str | None = None,
    show: bool = False,
) -> FuncAnimation:
    """Render a double-pendulum animation with angle traces."""

    times = np.asarray(times)
    angles1 = np.asarray(angles1)
    angles2 = np.asarray(angles2)
    angular_velocities1 = np.asarray(angular_velocities1)
    angular_velocities2 = np.asarray(angular_velocities2)

    x1 = length1 * np.sin(angles1)
    y1 = -length1 * np.cos(angles1)
    x2 = x1 + length2 * np.sin(angles2)
    y2 = y1 - length2 * np.cos(angles2)

    total_length = length1 + length2
    fig, (ax_motion, ax_angles) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.suptitle("Double pendulum")

    ax_motion.set_xlim(-1.15 * total_length, 1.15 * total_length)
    ax_motion.set_ylim(-1.15 * total_length, 0.3 * total_length)
    ax_motion.set_aspect("equal")
    ax_motion.set_title("Motion")
    ax_motion.grid(alpha=0.25)

    all_angles = np.concatenate((angles1, angles2))
    angle_pad = max(0.2, 0.1 * np.max(np.abs(all_angles)))
    ax_angles.set_title("Angle traces")
    ax_angles.set_xlabel("t (s)")
    ax_angles.set_ylabel("angle (rad)")
    ax_angles.set_xlim(times[0], times[-1] if len(times) > 1 else 1.0)
    ax_angles.set_ylim(all_angles.min() - angle_pad, all_angles.max() + angle_pad)
    ax_angles.plot(times, angles1, color="tab:blue", alpha=0.35, label=r"$\theta_1$")
    ax_angles.plot(times, angles2, color="tab:red", alpha=0.35, label=r"$\theta_2$")
    ax_angles.grid(alpha=0.25)
    ax_angles.legend(loc="upper right")

    upper_rod, = ax_motion.plot([], [], lw=2.6, color="tab:blue")
    lower_rod, = ax_motion.plot([], [], lw=2.6, color="tab:cyan")
    bob1 = ax_motion.scatter([], [], s=90, color="tab:orange", zorder=3)
    bob2 = ax_motion.scatter([], [], s=110, color="tab:red", zorder=4)
    trail_line, = ax_motion.plot([], [], lw=1.2, color="tab:red", alpha=0.3)
    marker1 = ax_angles.scatter([], [], s=50, color="tab:blue", zorder=3)
    marker2 = ax_angles.scatter([], [], s=50, color="tab:red", zorder=3)
    speed_text = ax_motion.text(
        0.04,
        0.94,
        "",
        transform=ax_motion.transAxes,
        va="top",
        fontsize=10,
    )

    def update(frame: int):
        upper_rod.set_data([0.0, x1[frame]], [0.0, y1[frame]])
        lower_rod.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        bob1.set_offsets(np.array([[x1[frame], y1[frame]]]))
        bob2.set_offsets(np.array([[x2[frame], y2[frame]]]))
        trail_line.set_data(x2[: frame + 1], y2[: frame + 1])
        marker1.set_offsets(np.array([[times[frame], angles1[frame]]]))
        marker2.set_offsets(np.array([[times[frame], angles2[frame]]]))
        speed_text.set_text(
            "\n".join(
                [
                    f"t = {times[frame]:.2f} s",
                    f"omega1 = {angular_velocities1[frame]:.2f} rad/s",
                    f"omega2 = {angular_velocities2[frame]:.2f} rad/s",
                ]
            )
        )
        return upper_rod, lower_rod, bob1, bob2, trail_line, marker1, marker2, speed_text

    frame_interval_ms = 1000 * float(np.mean(np.diff(times))) if len(times) > 1 else 20.0
    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=frame_interval_ms,
        blit=True,
    )

    if save_path:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1000.0 / frame_interval_ms)))
        animation.save(destination, writer="pillow", fps=fps)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation
