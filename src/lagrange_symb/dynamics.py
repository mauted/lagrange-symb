"""Numerical simulation for Lagrangian systems using JAX autodiff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

Array = jax.Array
LagrangianFn = Callable[[Array, Array, Any], Array]


@dataclass(frozen=True)
class SimulationResult:
    """Trajectory returned by the numerical simulator."""

    times: Array
    q: Array
    qdot: Array


def _dynamics_from_lagrangian(
    lagrangian: LagrangianFn,
    q: Array,
    qdot: Array,
    params: Any,
) -> Array:
    q = jnp.atleast_1d(q)
    qdot = jnp.atleast_1d(qdot)

    grad_q = jax.grad(lambda q_value: lagrangian(q_value, qdot, params))(q)
    dldqdot = lambda q_value, qdot_value: jax.grad(
        lambda velocity: lagrangian(q_value, velocity, params)
    )(qdot_value)
    mass_matrix = jax.jacobian(lambda velocity: dldqdot(q, velocity))(qdot)
    velocity_term = jax.jacobian(lambda q_value: dldqdot(q_value, qdot))(q) @ qdot

    mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
    qddot = jnp.linalg.solve(mass_matrix, grad_q - velocity_term)
    return jnp.concatenate((qdot, qddot))


def _rk4_step(
    lagrangian: LagrangianFn,
    state: Array,
    dt: float,
    params: Any,
) -> Array:
    derivative = lambda value: _dynamics_from_lagrangian(  # noqa: E731
        lagrangian,
        value[: value.shape[0] // 2],
        value[value.shape[0] // 2 :],
        params,
    )
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * dt * k1)
    k3 = derivative(state + 0.5 * dt * k2)
    k4 = derivative(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_lagrangian(
    lagrangian: LagrangianFn,
    initial_q: Array,
    initial_qdot: Array,
    params: Any,
    *,
    dt: float = 0.01,
    steps: int = 1_000,
) -> SimulationResult:
    """Simulate a Lagrangian system with a fourth-order Runge-Kutta integrator."""

    q0 = jnp.atleast_1d(jnp.asarray(initial_q, dtype=jnp.float32))
    qdot0 = jnp.atleast_1d(jnp.asarray(initial_qdot, dtype=jnp.float32))
    state = jnp.concatenate((q0, qdot0))

    trajectory = [state]
    for _ in range(steps):
        state = _rk4_step(lagrangian, state, dt, params)
        trajectory.append(state)

    states = jnp.stack(trajectory)
    dof = q0.shape[0]
    times = jnp.linspace(0.0, dt * steps, steps + 1)
    return SimulationResult(times=times, q=states[:, :dof], qdot=states[:, dof:])
