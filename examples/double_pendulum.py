"""Double pendulum demo for symbolic derivation, simulation, and animation."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import sympy as sp

from lagrange_symb.animation import animate_double_pendulum
from lagrange_symb.dynamics import simulate_lagrangian
from lagrange_symb.interactive import pick_double_pendulum_initial_conditions
from lagrange_symb.symbolic import double_pendulum_symbolics


def double_pendulum_lagrangian(
    q: jnp.ndarray,
    qdot: jnp.ndarray,
    params: dict[str, float],
) -> jnp.ndarray:
    """Return the double-pendulum Lagrangian."""

    theta1, theta2 = q
    omega1, omega2 = qdot
    mass1 = params["mass1"]
    mass2 = params["mass2"]
    length1 = params["length1"]
    length2 = params["length2"]
    gravity = params["gravity"]

    kinetic = 0.5 * (mass1 + mass2) * (length1**2) * omega1**2
    kinetic += 0.5 * mass2 * (length2**2) * omega2**2
    kinetic += mass2 * length1 * length2 * omega1 * omega2 * jnp.cos(theta1 - theta2)

    potential = (mass1 + mass2) * gravity * length1 * (1.0 - jnp.cos(theta1))
    potential += mass2 * gravity * length2 * (1.0 - jnp.cos(theta2))
    return kinetic - potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theta1-0", type=float, default=2.7, help="Initial first angle in radians.")
    parser.add_argument("--theta2-0", type=float, default=-2.3, help="Initial second angle in radians.")
    parser.add_argument("--omega1-0", type=float, default=1.4, help="Initial first angular velocity in radians per second.")
    parser.add_argument("--omega2-0", type=float, default=3.8, help="Initial second angular velocity in radians per second.")
    parser.add_argument("--length1", type=float, default=1.0, help="First pendulum length in meters.")
    parser.add_argument("--length2", type=float, default=0.85, help="Second pendulum length in meters.")
    parser.add_argument("--mass1", type=float, default=1.0, help="First pendulum mass in kilograms.")
    parser.add_argument("--mass2", type=float, default=1.0, help="Second pendulum mass in kilograms.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravitational acceleration in meters per second squared.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step for integration.")
    parser.add_argument("--steps", type=int, default=900, help="Number of integration steps.")
    parser.add_argument(
        "--save",
        default="outputs/double_pendulum.gif",
        help="Path to save the resulting animation.",
    )
    parser.add_argument(
        "--pick-ic",
        action="store_true",
        help="Open a local interactive picker for the initial angles and angular velocities.",
    )
    parser.add_argument("--show", action="store_true", help="Display the animation window after rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbolic_system = double_pendulum_symbolics()
    equations = symbolic_system.equations()

    print("Derived Euler-Lagrange equations:")
    for index, equation in enumerate(equations, start=1):
        print(f"Equation {index}:")
        print(sp.pretty(equation))
        print()

    params = {
        "mass1": args.mass1,
        "mass2": args.mass2,
        "length1": args.length1,
        "length2": args.length2,
        "gravity": args.gravity,
    }

    theta1_0 = args.theta1_0
    theta2_0 = args.theta2_0
    omega1_0 = args.omega1_0
    omega2_0 = args.omega2_0
    if args.pick_ic:
        theta1_0, theta2_0, omega1_0, omega2_0 = pick_double_pendulum_initial_conditions(
            length1=args.length1,
            length2=args.length2,
            theta1_0=args.theta1_0,
            theta2_0=args.theta2_0,
            omega1_0=args.omega1_0,
            omega2_0=args.omega2_0,
        )

    result = simulate_lagrangian(
        double_pendulum_lagrangian,
        initial_q=jnp.array([theta1_0, theta2_0]),
        initial_qdot=jnp.array([omega1_0, omega2_0]),
        params=params,
        dt=args.dt,
        steps=args.steps,
    )

    animate_double_pendulum(
        result.times,
        result.q[:, 0],
        result.q[:, 1],
        result.qdot[:, 0],
        result.qdot[:, 1],
        length1=args.length1,
        length2=args.length2,
        save_path=args.save,
        show=args.show,
    )

    print(f"Animation saved to {args.save}")


if __name__ == "__main__":
    main()
