"""Critical points: location, and the implicit-function-theorem derivative.

These are diagnostics -- the solver does not use them (see ``solver.py``). They
are still differentiable, which is what lets quantities like the axis position
be differentiated with respect to the equilibrium.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags.critical import make_critical_fns, soft_max  # noqa: E402
from jags.grid import Grid  # noqa: E402

BOX = np.array([[0.4, -0.8], [1.6, -0.8], [1.6, 0.8], [0.4, 0.8]])


@pytest.fixture(scope="module")
def setup():
    g = Grid(0.2, 2.0, -1.0, 1.0, 65, 65, BOX)
    return g, make_critical_fns(g)


def peaked(g, R0=1.05, Z0=0.13, a=0.42, b=0.55):
    """A smooth field with a single interior maximum at a known location."""
    return jnp.asarray(
        np.exp(-(((g.R - R0) / a) ** 2 + ((g.Z - Z0) / b) ** 2))
    )


def test_axis_located_between_grid_points(setup):
    """The refined axis is far more accurate than the grid spacing.

    A bare argmax is limited to half a cell; the inner Newton on grad(psi) = 0
    should do much better than that.
    """
    g, critical = setup
    R0, Z0 = 1.05, 0.13
    c = critical(peaked(g, R0, Z0), use_xpoints=False)
    axis = np.asarray(c.axis)
    assert abs(axis[0] - R0) < 0.05 * g.dR
    assert abs(axis[1] - Z0) < 0.05 * g.dZ
    assert float(c.psi_axis) == pytest.approx(1.0, abs=1e-4)


def test_axis_gradient_matches_finite_differences(setup):
    """d(psi_axis)/d(psi) from the implicit function theorem.

    The inner Newton iteration runs under stop_gradient, so the derivative comes
    entirely from the single final step. If that step were wrong, the gradient
    would be silently incorrect while the value stayed right -- hence this test.
    """
    g, critical = setup
    psi = peaked(g)

    def psi_axis_of(p):
        return critical(p, use_xpoints=False).psi_axis

    grad = np.asarray(jax.grad(psi_axis_of)(psi))

    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.standard_normal(psi.shape)
        v /= np.linalg.norm(v)
        h = 1e-6
        fd = float(
            (psi_axis_of(psi + h * v) - psi_axis_of(psi - h * v)) / (2 * h)
        )
        ad = float((grad * v).sum())
        assert abs(fd - ad) < 1e-6 * max(1.0, abs(fd))


def test_axis_position_gradient_is_nonzero(setup):
    """Moving psi must move the axis.

    A hard argmax would give exactly zero here, which is the failure the IFT
    step exists to avoid.
    """
    g, critical = setup
    psi = peaked(g)
    grad = np.asarray(
        jax.grad(lambda p: critical(p, use_xpoints=False).axis[0])(psi)
    )
    assert np.abs(grad).max() > 1e-6


def test_soft_max_approaches_hard_max():
    vals = jnp.array([1.0, 2.5, 2.0, -3.0])
    valid = jnp.array([True, True, True, True])
    for beta, tol in [(1e2, 5e-2), (1e3, 5e-3), (1e4, 5e-4)]:
        assert float(soft_max(vals, valid, beta)) == pytest.approx(2.5, abs=tol)


def test_soft_max_ignores_invalid_entries():
    vals = jnp.array([1.0, 99.0, 2.0])
    valid = jnp.array([True, False, True])
    assert float(soft_max(vals, valid, 1e4)) == pytest.approx(2.0, abs=1e-3)


def test_boundary_flux_tracks_limiter_contact(setup):
    """With no X-point, psi_bndry is the largest flux on the limiter contour.

    This mirrors FreeGSNKE, which sets psi_bndry to the maximum of psi
    interpolated onto the limiter (``limiter_func.py:490-493``).
    """
    g, critical = setup
    psi = peaked(g, R0=1.0, Z0=0.0, a=0.9, b=1.2)  # broad, reaches the limiter
    c = critical(psi, beta_norm=4000.0, use_xpoints=False)

    pts = g.limiter_edge_points()
    from jags.interp import make_interpolator

    value, _, _ = make_interpolator(g)
    want = max(float(value(psi, jnp.asarray(p))) for p in pts)
    assert float(c.psi_bndry) == pytest.approx(want, rel=2e-3)


def test_core_mask_is_between_zero_and_one(setup):
    g, critical = setup
    c = critical(peaked(g), use_xpoints=False)
    mask = np.asarray(c.mask)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    # the mask must vanish outside the limiter
    assert np.abs(mask[~g.limiter_mask()]).max() == 0.0
