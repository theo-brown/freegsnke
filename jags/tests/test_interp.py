"""Bicubic interpolation: accuracy, C1 continuity, analytic derivatives."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags.grid import Grid  # noqa: E402
from jags.interp import make_interpolator  # noqa: E402

BOX = np.array([[0.3, -1.0], [1.8, -1.0], [1.8, 1.0], [0.3, 1.0]])


def field(R, Z):
    return np.sin(2.1 * R) * np.cos(1.7 * Z) + 0.3 * R**2 - 0.2 * Z**2


def field_grad(R, Z):
    return np.array(
        [
            2.1 * np.cos(2.1 * R) * np.cos(1.7 * Z) + 0.6 * R,
            -1.7 * np.sin(2.1 * R) * np.sin(1.7 * Z) - 0.4 * Z,
        ]
    )


@pytest.fixture(scope="module")
def setup():
    g = Grid(0.2, 2.0, -1.2, 1.2, 65, 65, BOX)
    psi = jnp.asarray(field(g.R, g.Z))
    return g, psi, make_interpolator(g)


def sample_points(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(0.5, 1.7, n), rng.uniform(-0.8, 0.8, n)], axis=-1
    )


def test_value_accuracy(setup):
    _, psi, (value, _, _) = setup
    for p in sample_points():
        assert abs(float(value(psi, jnp.asarray(p))) - field(*p)) < 1e-4


def test_gradient_accuracy(setup):
    _, psi, (_, gradient, _) = setup
    for p in sample_points():
        got = np.asarray(gradient(psi, jnp.asarray(p)))
        assert np.abs(got - field_grad(*p)).max() < 5e-3


def test_interpolant_is_c1_across_cell_boundaries(setup):
    """The gradient must not jump between cells.

    Bilinear interpolation fails this, and with it the inner Newton iteration
    on grad(psi) = 0 fails to converge.
    """
    g, psi, (_, gradient, _) = setup
    for i in (20, 33, 47):
        edge = g.R_1d[i]
        left = np.asarray(gradient(psi, jnp.array([edge - 1e-9, 0.13])))
        right = np.asarray(gradient(psi, jnp.array([edge + 1e-9, 0.13])))
        assert np.abs(left - right).max() < 1e-6

    for j in (20, 33, 47):
        edge = g.Z_1d[j]
        below = np.asarray(gradient(psi, jnp.array([1.05, edge - 1e-9])))
        above = np.asarray(gradient(psi, jnp.array([1.05, edge + 1e-9])))
        assert np.abs(below - above).max() < 1e-6


def test_hessian_is_symmetric(setup):
    _, psi, (_, _, hessian) = setup
    for p in sample_points(8, seed=3):
        H = np.asarray(hessian(psi, jnp.asarray(p)))
        assert np.abs(H - H.T).max() < 1e-12


def test_reproduces_grid_values_exactly(setup):
    """At a grid node the interpolant returns the stored value."""
    g, psi, (value, _, _) = setup
    for i, j in [(20, 20), (33, 40), (45, 25)]:
        got = float(value(psi, jnp.array([g.R_1d[i], g.Z_1d[j]])))
        assert got == pytest.approx(float(psi[i, j]), rel=1e-12)
