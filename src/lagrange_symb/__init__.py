"""Tools for symbolic and numerical Lagrangian mechanics experiments."""

from .animation import animate_double_pendulum, animate_pendulum
from .dynamics import SimulationResult, simulate_lagrangian
from .interactive import (
    pick_double_pendulum_initial_conditions,
    pick_pendulum_initial_conditions,
)
from .symbolic import (
    SymbolicSystem,
    build_lagrangian,
    build_symbolic_system,
    double_pendulum_symbolics,
    derive_euler_lagrange_equations,
    generalized_coordinates,
    simple_pendulum_symbolics,
)

__all__ = [
    "SimulationResult",
    "SymbolicSystem",
    "animate_double_pendulum",
    "animate_pendulum",
    "build_lagrangian",
    "build_symbolic_system",
    "double_pendulum_symbolics",
    "derive_euler_lagrange_equations",
    "generalized_coordinates",
    "pick_double_pendulum_initial_conditions",
    "pick_pendulum_initial_conditions",
    "simple_pendulum_symbolics",
    "simulate_lagrangian",
]
