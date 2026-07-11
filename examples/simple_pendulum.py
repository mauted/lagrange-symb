"""Simple pendulum demo for symbolic derivation, simulation, and animation."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import sympy as sp

from lagrange_symb.animation import animate_pendulum
from lagrange_symb.dynamics import simulate_lagrangian
from lagrange_symb.interactive import pick_pendulum_initial_conditions
from lagrange_symb.symbolic import simple_pendulum_symbolics


def pendulum_lagrangian(q: jnp.ndarray, qdot: jnp.ndarray, params: dict[str, float]) -> jnp.ndarray:
    theta = q[0]
    omega = qdot[0]
    mass = params["mass"]
    length = params["length"]
    gravity = params["gravity"]

    kinetic = 0.5 * mass * (length**2) * omega**2
    potential = mass * gravity * length * (1.0 - jnp.cos(theta))
    return kinetic - potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta0", type=float, default=1.1, help="Initial angle in radians.")
    parser.add_argument("--omega0", type=float, default=0.0, help="Initial angular velocity in radians per second.")
    parser.add_argument("--length", type=float, default=1.0, help="Pendulum length in meters.")
    parser.add_argument("--mass", type=float, default=1.0, help="Pendulum mass in kilograms.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in meters per second squared.")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step for integration.")
    parser.add_argument("--steps", type=int, default=450, help="Number of integration steps.")
    parser.add_argument(
        "--save",
        default="outputs/simple_pendulum.gif",
        help="Path to save the resulting animation.",
    )
    parser.add_argument(
        "--pick-ic",
        action="store_true",
        help="Open a local interactive picker for the initial angle and angular velocity.",
    )
    parser.add_argument("--show", action="store_true", help="Display the animation window after rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbolic_system = simple_pendulum_symbolics()
    symbolic_equation = symbolic_system.equations()[0]
    mass_symbol, length_symbol, _ = symbolic_system.parameters

    print("Derived Euler-Lagrange equation:")
    print(sp.pretty(symbolic_equation))
    print()

    simplified = sp.simplify(
        symbolic_equation.lhs / (mass_symbol * length_symbol)
    )
    print("Simplified form (dividing by m*l):")
    print(sp.pretty(sp.Eq(simplified, 0)))
    print()

    params = {
        "mass": args.mass,
        "length": args.length,
        "gravity": args.gravity,
    }

    theta0 = args.theta0
    omega0 = args.omega0
    if args.pick_ic:
        theta0, omega0 = pick_pendulum_initial_conditions(
            length=args.length,
            theta0=args.theta0,
            omega0=args.omega0,
        )

    result = simulate_lagrangian(
        pendulum_lagrangian,
        initial_q=jnp.array([theta0]),
        initial_qdot=jnp.array([omega0]),
        params=params,
        dt=args.dt,
        steps=args.steps,
    )

    animate_pendulum(
        result.times,
        result.q[:, 0],
        result.qdot[:, 0],
        length=args.length,
        save_path=args.save,
        show=args.show,
    )

    print(f"Animation saved to {args.save}")


if __name__ == "__main__":
    main()
