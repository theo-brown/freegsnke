"""Shared test fixtures."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jags import profiles, solver  # noqa: E402
from jags.grid import Grid  # noqa: E402

BOX = np.array([[0.35, -0.85], [1.55, -0.85], [1.55, 0.85], [0.35, 0.85]])
COILS = np.array(
    [[0.25, -1.05], [0.25, 1.05], [1.75, -1.05], [1.75, 1.05], [1.9, -0.4], [1.9, 0.4]]
)
CURRENTS = np.array([3.0e5, 3.0e5, 1.6e5, 1.6e5, -1.0e5, -1.0e5])
IP = 4.0e5


@pytest.fixture(scope="session")
def case():
    """A small, fast machine for exercising the solver.

    These coil currents do **not** confine the plasma -- the peak current sits
    against the limiter, so the vessel mask does the bounding. That is fine for
    testing machinery, but it means this fixture cannot support any claim about
    confined equilibria; those are checked against the MAST-U reference instead.

    Returns ``(grid, machine, psi_coil, profile, psi_p0)``.
    """
    g = Grid(0.15, 2.0, -1.2, 1.2, 33, 33, BOX)
    m = solver.build_machine(g, COILS)
    prof = profiles.compact(psi_edge=0.0, psi_scale=0.15, p0=8.0e3, fvac=0.5, f1=0.05)
    return (
        g,
        m,
        solver.coil_flux(m, CURRENTS),
        prof,
        solver.initial_guess(m, (0.9, 0.0), (0.45, 0.65), IP),
    )
