# lagrange-symb

An educational mechanics project for deriving Euler-Lagrange equations symbolically and simulating the resulting dynamics with JAX.

## What it does

This first version focuses on two pendulum systems:

- derives the Euler-Lagrange equation with `sympy`
- simulates simple and double pendula numerically from JAX-defined Lagrangians
- renders animations of their motion with `matplotlib`
- optionally opens a local setup window so you can drag the initial positions before starting the simulation

The symbolic layer is now reusable: you can define generalized coordinates plus symbolic
kinetic and potential energy terms, then derive equations from the resulting system object.

The long-term goal is to grow this into a playground for physics-side projects: symbolic derivations, constrained systems, richer visualizations, and more interesting mechanical models.

## Quickstart

```bash
uv sync
uv run python examples/simple_pendulum.py --save outputs/simple_pendulum.gif
uv run python examples/double_pendulum.py --save outputs/double_pendulum.gif
```

Those commands print the derived equations of motion and save GIFs under `outputs/`.

## Interactive Initial Conditions

For local exploratory runs, both examples accept `--pick-ic`:

```bash
uv run python examples/simple_pendulum.py --pick-ic --show
uv run python examples/double_pendulum.py --pick-ic --show
```

This opens a Matplotlib setup window before simulation:

- `simple_pendulum`: drag the bob to set `theta0`, then use the slider for `omega0`
- `double_pendulum`: drag either bob to set `theta1_0` and `theta2_0`, then use the sliders for `omega1_0` and `omega2_0`

Interactive picking requires a GUI Matplotlib backend. It will not work with `MPLBACKEND=Agg`.

## Symbolic API

```python
import sympy as sp

from lagrange_symb.symbolic import build_symbolic_system, generalized_coordinates

t = sp.symbols("t", real=True)
m, k = sp.symbols("m k", positive=True, real=True)
(q,) = generalized_coordinates("q", t)
qdot = sp.diff(q, t)

system = build_symbolic_system(
    name="mass-spring",
    time=t,
    coordinates=(q,),
    kinetic_energy=sp.Rational(1, 2) * m * qdot**2,
    potential_energy=sp.Rational(1, 2) * k * q**2,
    parameters=(m, k),
)

equations = system.equations()
print(equations[0])
```
