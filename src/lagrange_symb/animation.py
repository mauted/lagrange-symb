"""Animation helpers for mechanics demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Black showcase theme; mathtext uses Computer Modern for a LaTeX look
# without paying the per-frame cost of text.usetex during animation.
_DARK_LATEX_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "text.color": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "axes.titlecolor": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "axes.edgecolor": "#ffffff",
    "figure.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "savefig.facecolor": "#000000",
    "grid.color": "#737373",
    "legend.facecolor": "#000000",
    "legend.edgecolor": "#525252",
    "legend.labelcolor": "#ffffff",
}

_ACCENT = {
    "rod1": "#7dd3fc",
    "rod2": "#67e8f9",
    "bob1": "#fbbf24",
    "bob2": "#fb7185",
    "trail": "#fb7185",
    "phase": "#c4b5fd",
    "phase_point": "#f87171",
}


def _style_axes(ax) -> None:
    """Apply dark spines, ticks, and grid to an axes."""

    ax.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#ffffff", labelsize=10)
    ax.grid(True, alpha=0.22, color="#737373", linewidth=0.7)
    # Render tick numbers with Computer Modern mathtext.
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: rf"${v:g}$"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: rf"${v:g}$"))


def _frame_indices(n_frames: int, frame_stride: int) -> np.ndarray:
    """Return sample indices, always including the final frame."""

    stride = max(1, int(frame_stride))
    indices = np.arange(0, n_frames, stride, dtype=int)
    if len(indices) == 0 or indices[-1] != n_frames - 1:
        indices = np.append(indices, n_frames - 1)
    return indices


def _playback_interval_ms(times: np.ndarray, frame_indices: np.ndarray) -> float:
    """Keep wall-clock duration close to the simulated duration after striding."""

    if len(frame_indices) <= 1:
        return 40.0
    sampled_times = times[frame_indices]
    return 1000.0 * float(np.mean(np.diff(sampled_times)))


def _save_animation(
    animation: FuncAnimation,
    fig,
    save_path: str,
    frame_interval_ms: float,
    *,
    dpi: int = 90,
) -> None:
    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1000.0 / frame_interval_ms)))
    animation.save(
        destination,
        writer="pillow",
        fps=fps,
        dpi=dpi,
        savefig_kwargs={"facecolor": fig.get_facecolor(), "edgecolor": "none"},
    )


def animate_pendulum(
    times: np.ndarray,
    angles: np.ndarray,
    angular_velocities: np.ndarray,
    *,
    length: float = 1.0,
    save_path: str | None = None,
    show: bool = False,
    frame_stride: int = 4,
    dpi: int = 90,
) -> FuncAnimation:
    """Render a pendulum animation with a phase-space subplot."""

    times = np.asarray(times)
    angles = np.asarray(angles)
    angular_velocities = np.asarray(angular_velocities)

    x = length * np.sin(angles)
    y = -length * np.cos(angles)
    indices = _frame_indices(len(times), frame_stride)

    with plt.rc_context(_DARK_LATEX_RC):
        fig, (ax_motion, ax_phase) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.patch.set_facecolor("#000000")
        fig.suptitle(r"$\mathrm{Simple\ pendulum}$", color="#ffffff", fontsize=14)

        ax_motion.set_xlim(-1.2 * length, 1.2 * length)
        ax_motion.set_ylim(-1.2 * length, 0.3 * length)
        ax_motion.set_aspect("equal")
        ax_motion.set_title(r"$\mathrm{Motion}$", fontsize=12)
        _style_axes(ax_motion)

        ax_phase.set_title(r"$\mathrm{Phase\ portrait}$", fontsize=12)
        ax_phase.set_xlabel(r"$\theta\ \mathrm{(rad)}$", fontsize=11)
        ax_phase.set_ylabel(r"$\dot{\theta}\ \mathrm{(rad/s)}$", fontsize=11)
        ax_phase.plot(angles, angular_velocities, color=_ACCENT["phase"], alpha=0.35, lw=1.4)
        _style_axes(ax_phase)

        (rod_line,) = ax_motion.plot([], [], lw=2.5, color=_ACCENT["rod1"])
        bob = ax_motion.scatter([], [], s=100, color=_ACCENT["bob1"], zorder=3)
        (trail_line,) = ax_motion.plot([], [], lw=1.5, color=_ACCENT["bob1"], alpha=0.4)
        phase_point = ax_phase.scatter([], [], s=45, color=_ACCENT["phase_point"], zorder=3)
        time_text = ax_motion.text(
            0.04,
            0.94,
            "",
            transform=ax_motion.transAxes,
            va="top",
            fontsize=11,
            color="#ffffff",
        )

        def update(frame: int):
            rod_line.set_data([0.0, x[frame]], [0.0, y[frame]])
            bob.set_offsets(np.array([[x[frame], y[frame]]]))
            trail_line.set_data(x[: frame + 1], y[: frame + 1])
            phase_point.set_offsets(np.array([[angles[frame], angular_velocities[frame]]]))
            time_text.set_text(rf"$t = {times[frame]:.2f}\,\mathrm{{s}}$")
            return rod_line, bob, trail_line, phase_point, time_text

        frame_interval_ms = _playback_interval_ms(times, indices)
        animation = FuncAnimation(
            fig,
            update,
            frames=indices,
            interval=frame_interval_ms,
            blit=True,
        )

        if save_path:
            _save_animation(animation, fig, save_path, frame_interval_ms, dpi=dpi)

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
    frame_stride: int = 4,
    dpi: int = 90,
    trail_length: int = 180,
) -> FuncAnimation:
    """Render a double-pendulum motion animation."""

    times = np.asarray(times)
    angles1 = np.asarray(angles1)
    angles2 = np.asarray(angles2)
    angular_velocities1 = np.asarray(angular_velocities1)
    angular_velocities2 = np.asarray(angular_velocities2)

    x1 = length1 * np.sin(angles1)
    y1 = -length1 * np.cos(angles1)
    x2 = x1 + length2 * np.sin(angles2)
    y2 = y1 - length2 * np.cos(angles2)
    indices = _frame_indices(len(times), frame_stride)
    trail_length = max(2, int(trail_length))

    total_length = length1 + length2
    axis_limit = 2.0 if total_length <= 1.85 else 1.15 * total_length

    with plt.rc_context(_DARK_LATEX_RC):
        fig, ax_motion = plt.subplots(figsize=(5.6, 5.6))
        fig.patch.set_facecolor("#000000")
        fig.suptitle(r"$\mathrm{Double\ pendulum}$", color="#ffffff", fontsize=14)

        ax_motion.set_xlim(-axis_limit, axis_limit)
        ax_motion.set_ylim(-axis_limit, axis_limit)
        ax_motion.set_aspect("equal")
        ax_motion.set_xlabel(r"$x$", fontsize=11)
        ax_motion.set_ylabel(r"$y$", fontsize=11)
        _style_axes(ax_motion)

        (upper_rod,) = ax_motion.plot([], [], lw=2.6, color=_ACCENT["rod1"])
        (lower_rod,) = ax_motion.plot([], [], lw=2.6, color=_ACCENT["rod2"])
        bob1 = ax_motion.scatter([], [], s=90, color=_ACCENT["bob1"], zorder=3)
        bob2 = ax_motion.scatter([], [], s=110, color=_ACCENT["bob2"], zorder=4)
        (trail_line,) = ax_motion.plot([], [], lw=1.2, color=_ACCENT["trail"], alpha=0.45)
        speed_text = ax_motion.text(
            0.04,
            0.96,
            "",
            transform=ax_motion.transAxes,
            va="top",
            fontsize=11,
            color="#ffffff",
            linespacing=1.35,
        )

        def update(frame: int):
            upper_rod.set_data([0.0, x1[frame]], [0.0, y1[frame]])
            lower_rod.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
            bob1.set_offsets(np.array([[x1[frame], y1[frame]]]))
            bob2.set_offsets(np.array([[x2[frame], y2[frame]]]))
            trail_start = max(0, frame + 1 - trail_length)
            trail_line.set_data(x2[trail_start : frame + 1], y2[trail_start : frame + 1])
            speed_text.set_text(
                "\n".join(
                    [
                        rf"$t = {times[frame]:.2f}\,\mathrm{{s}}$",
                        rf"$\omega_1 = {angular_velocities1[frame]:.2f}\,\mathrm{{rad/s}}$",
                        rf"$\omega_2 = {angular_velocities2[frame]:.2f}\,\mathrm{{rad/s}}$",
                    ]
                )
            )
            return upper_rod, lower_rod, bob1, bob2, trail_line, speed_text

        frame_interval_ms = _playback_interval_ms(times, indices)
        animation = FuncAnimation(
            fig,
            update,
            frames=indices,
            interval=frame_interval_ms,
            blit=True,
        )

        if save_path:
            _save_animation(animation, fig, save_path, frame_interval_ms, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

    return animation
