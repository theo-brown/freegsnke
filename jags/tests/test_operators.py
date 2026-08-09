"""Grid operators: the Delstar discretisation and the Green's functions."""

import numpy as np
import pytest

from jags import operators
from jags.grid import Grid

BOX = np.array([[0.35, -0.85], [1.55, -0.85], [1.55, 0.85], [0.35, 0.85]])


def make_grid(n=33):
    return Grid(0.2, 2.0, -1.0, 1.0, n, n, BOX)


def solovev_psi(R, Z, a=0.3, b=0.7, c=-0.2, d=0.1):
    """Quartic flux function whose Delstar image is known analytically.

    Delstar(R^4) = 12R^2 - 4R^2 = 8R^2, Delstar(Z^2) = 2, Delstar(R^2) = 0.
    This is the Solov'ev family: constant p' and FF' make the Grad-Shafranov
    equation linear with a polynomial right-hand side, and this is its solution.
    """
    return a * R**4 + b * Z**2 + c * R**2 + d


def solovev_delstar(R, Z, a=0.3, b=0.7):
    return 8 * a * R**2 + 2 * b


def test_delstar_exact_on_quartic():
    """The fourth-order stencils integrate the Solov'ev quartic exactly.

    The centred second-derivative stencil is exact to degree 5 and the centred
    first-derivative stencil to degree 4, and the one-sided variants used next
    to the boundary are exact to the same order, so the only error here is
    floating point.
    """
    g = make_grid(33)
    R, Z = g.R, g.Z
    psi = solovev_psi(R, Z)
    got = (operators.delstar_matrix(g) @ psi.ravel()).reshape(R.shape)
    want = solovev_delstar(R, Z)

    interior = np.zeros(R.shape, bool)
    interior[1:-1, 1:-1] = True
    err = np.abs(got - want)[interior].max() / np.abs(want).max()
    assert err < 1e-12, f"relative error {err:.3e}"

    # boundary rows are the identity, so A psi returns psi there
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        np.testing.assert_allclose(got[sl], psi[sl], rtol=1e-13)


def test_inverse_delstar_round_trip():
    g = make_grid(17)
    A = operators.delstar_matrix(g).toarray()
    Ainv = operators.inverse_delstar(g)
    assert np.abs(Ainv @ A - np.eye(g.N)).max() < 1e-8


def test_solovev_fixed_boundary_solve():
    """Recover the analytic Solov'ev solution from a fixed-boundary solve.

    Feeding the analytic source into the interior and the analytic flux onto the
    boundary must return the analytic solution, which exercises the assembled
    operator and its inverse together.
    """
    g = make_grid(33)
    R, Z = g.R, g.Z
    psi = solovev_psi(R, Z)

    rhs = solovev_delstar(R, Z).copy()
    for sl in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
        rhs[sl] = psi[sl]

    got = (operators.inverse_delstar(g) @ rhs.ravel()).reshape(psi.shape)
    err = np.abs(got - psi).max() / np.ptp(psi)
    assert err < 1e-10, f"relative error {err:.3e}"


def test_greens_satisfies_homogeneous_equation():
    """Delstar of a filament's flux vanishes away from the filament.

    An independent check on the Green's function that needs no reference
    implementation: G is by definition a homogeneous solution off-source.
    """
    g = make_grid(65)
    Rc, Zc = 1.9, 0.75  # filament placed away from the region tested
    psi = operators.greens(Rc, Zc, g.R, g.Z)
    got = (operators.delstar_matrix(g) @ psi.ravel()).reshape(psi.shape)

    # Test well away from both the filament and the boundary rows.
    far = (np.hypot(g.R - Rc, g.Z - Zc) > 0.5)
    far[:2, :] = far[-2:, :] = far[:, :2] = far[:, -2:] = False
    assert far.sum() > 100
    scale = np.abs(psi).max() / min(g.dR, g.dZ) ** 2
    assert np.abs(got)[far].max() / scale < 1e-6

    # and G is symmetric under exchange of source and observation point
    a, b = (1.1, 0.2), (1.7, -0.4)
    assert operators.greens(*a, *b) == pytest.approx(
        operators.greens(*b, *a), rel=1e-12
    )


def test_boundary_greens_matrix_shape_and_self_term():
    g = make_grid(17)
    G = operators.boundary_greens_matrix(g)
    bnd = g.boundary_indices
    assert G.shape == (len(bnd), g.N)
    # the self-interaction is removed to avoid the log singularity
    flat = bnd[:, 0] * g.nZ + bnd[:, 1]
    assert np.abs(G[np.arange(len(bnd)), flat]).max() == 0.0
