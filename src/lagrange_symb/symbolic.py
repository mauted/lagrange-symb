"""Symbolic helpers for constructing and analyzing Lagrangian systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sympy as sp


@dataclass(frozen=True)
class SymbolicSystem:
    """Container for a symbolic mechanics setup."""

    name: str | None
    time: sp.Symbol
    coordinates: tuple[sp.Expr, ...]
    kinetic_energy: sp.Expr
    potential_energy: sp.Expr
    parameters: tuple[sp.Symbol, ...] = ()

    @property
    def lagrangian(self) -> sp.Expr:
        """Return the Lagrangian formed from kinetic and potential energy."""

        return build_lagrangian(self.kinetic_energy, self.potential_energy)

    def equations(self) -> tuple[sp.Equality, ...]:
        """Return the Euler-Lagrange equations for this system."""

        return derive_euler_lagrange_equations(
            self.lagrangian,
            self.coordinates,
            self.time,
        )


def generalized_coordinates(
    names: str | Iterable[str],
    time: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    """Create generalized coordinates as time-dependent SymPy functions."""

    if isinstance(names, str):
        coordinate_names = tuple(token for token in names.replace(",", " ").split() if token)
    else:
        coordinate_names = tuple(names)
    return tuple(sp.Function(name)(time) for name in coordinate_names)


def build_lagrangian(kinetic_energy: sp.Expr, potential_energy: sp.Expr) -> sp.Expr:
    """Return the Lagrangian ``L = T - V``."""

    return sp.simplify(kinetic_energy - potential_energy)


def build_symbolic_system(
    *,
    coordinates: tuple[sp.Expr, ...],
    time: sp.Symbol,
    kinetic_energy: sp.Expr,
    potential_energy: sp.Expr,
    parameters: tuple[sp.Symbol, ...] = (),
    name: str | None = None,
) -> SymbolicSystem:
    """Build a symbolic system from kinetic and potential energy."""

    return SymbolicSystem(
        name=name,
        time=time,
        coordinates=coordinates,
        kinetic_energy=sp.simplify(kinetic_energy),
        potential_energy=sp.simplify(potential_energy),
        parameters=parameters,
    )


def derive_euler_lagrange_equations(
    lagrangian: sp.Expr,
    coordinates: tuple[sp.Expr, ...],
    time: sp.Symbol,
) -> tuple[sp.Equality, ...]:
    """Return the Euler-Lagrange equations for the given Lagrangian."""

    equations: list[sp.Equality] = []
    for coordinate in coordinates:
        generalized_velocity = sp.diff(coordinate, time)
        lhs = sp.diff(sp.diff(lagrangian, generalized_velocity), time)
        rhs = sp.diff(lagrangian, coordinate)
        equations.append(sp.Eq(sp.simplify(lhs - rhs), 0))
    return tuple(equations)


def simple_pendulum_symbolics() -> SymbolicSystem:
    """Create a symbolic simple-pendulum system."""

    time = sp.symbols("t", real=True)
    mass, length, gravity = sp.symbols("m l g", positive=True, real=True)
    (theta,) = generalized_coordinates(("theta",), time)
    theta_dot = sp.diff(theta, time)

    kinetic = sp.Rational(1, 2) * mass * length**2 * theta_dot**2
    potential = mass * gravity * length * (1 - sp.cos(theta))
    return build_symbolic_system(
        name="simple pendulum",
        time=time,
        coordinates=(theta,),
        kinetic_energy=kinetic,
        potential_energy=potential,
        parameters=(mass, length, gravity),
    )


def double_pendulum_symbolics() -> SymbolicSystem:
    """Create a symbolic double-pendulum system."""

    time = sp.symbols("t", real=True)
    mass1, mass2 = sp.symbols("m1 m2", positive=True, real=True)
    length1, length2 = sp.symbols("l1 l2", positive=True, real=True)
    gravity = sp.symbols("g", positive=True, real=True)
    theta1, theta2 = generalized_coordinates(("theta1", "theta2"), time)

    x1 = length1 * sp.sin(theta1)
    y1 = -length1 * sp.cos(theta1)
    x2 = x1 + length2 * sp.sin(theta2)
    y2 = y1 - length2 * sp.cos(theta2)

    vx1 = sp.diff(x1, time)
    vy1 = sp.diff(y1, time)
    vx2 = sp.diff(x2, time)
    vy2 = sp.diff(y2, time)

    kinetic = sp.Rational(1, 2) * mass1 * (vx1**2 + vy1**2)
    kinetic += sp.Rational(1, 2) * mass2 * (vx2**2 + vy2**2)

    potential = mass1 * gravity * length1 * (1 - sp.cos(theta1))
    potential += mass2 * gravity * (
        length1 * (1 - sp.cos(theta1)) + length2 * (1 - sp.cos(theta2))
    )

    return build_symbolic_system(
        name="double pendulum",
        time=time,
        coordinates=(theta1, theta2),
        kinetic_energy=kinetic,
        potential_energy=potential,
        parameters=(mass1, mass2, length1, length2, gravity),
    )
