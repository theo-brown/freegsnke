"""Solver: the PDE itself, and that Newton is exact."""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import jacobian, operators, solver  # noqa: E402

from conftest import IP  # noqa: E402


@pytest.fixture(scope="module")
def solved(case):
    _, m, psi_coil, prof, psi_p0 = case
    return solver.solve(m, prof, psi_coil, psi_p0, Ip=IP, max_newton=20)


def test_grad_shafranov_equation_is_satisfied(case, solved):
    """Delstar psi_plasma = -mu0 R Jtor in the interior, and the free-boundary
    condition holds on the edge."""
    g, m, _, _, _ = case
    psi_p = np.asarray(solved.psi_p)
    got = (operators.delstar_matrix(g) @ psi_p.ravel()).reshape(psi_p.shape)
    want = -operators.MU0 * np.asarray(m.R) * np.asarray(solved.jtor)

    interior = np.zeros(psi_p.shape, bool)
    interior[1:-1, 1:-1] = True
    assert np.abs(got - want)[interior].max() / np.abs(want).max() < 1e-10

    bnd = np.asarray(m.bnd_flat)
    edge = np.asarray(m.G_bnd) @ np.asarray(solved.jtor).ravel()
    assert np.abs(psi_p.ravel()[bnd] - edge).max() / np.abs(edge).max() < 1e-10


def test_total_current_is_enforced(case, solved):
    g, _, _, _, _ = case
    assert float(jnp.sum(solved.jtor)) * g.dA == pytest.approx(IP, rel=1e-10)


def test_convergence_is_quadratic(solved):
    """Each Newton step should roughly square the residual.

    With a smooth residual and an exact Jacobian this is the expected rate, and
    it is the sharpest available evidence that both hold.
    """
    h = solved.residual_history
    assert solved.converged and len(h) >= 4
    for prev, nxt in zip(h[1:-1], h[2:]):
        # stop once the quadratic prediction is below machine precision
        assert nxt <= max(50.0 * prev**2, 1e-14), f"{nxt:.3e} vs {prev:.3e}^2"
    assert h[-1] < 1e-11


def test_matrix_free_matches_the_dense_jacobian(case, solved):
    """GMRES on the Jacobian's action must give the same step as forming it.

    ``dense_jacobian`` is the assumption-free reference: N tangent vectors, no
    structural claims. If the two disagree, the matrix-free step is not exact.
    """
    _, m, psi_coil, prof, _ = case
    residual, _ = solver.make_residual(m, prof, psi_coil, IP)
    psi = solved.psi_p.reshape(-1)
    F = residual(psi)

    dense = jnp.linalg.solve(jacobian.dense_jacobian(residual, psi), -F)
    free = jacobian.make_matrix_free_step(residual)(psi, F)
    assert np.linalg.norm(np.asarray(free - dense)) / np.linalg.norm(
        np.asarray(dense)
    ) < 1e-8


def test_no_plasma_is_a_fixed_point(case):
    """Zero current gives zero flux gives zero current.

    This is why the iteration must start from a state that already has a plasma,
    and it is worth pinning because a solver that silently returns the trivial
    solution looks like it converged.
    """
    _, m, psi_coil, prof, _ = case
    residual, _ = solver.make_residual(m, prof, psi_coil, None)
    below = jnp.zeros(m.grid.N) - 10.0  # psi below the profile edge everywhere
    assert float(jnp.linalg.norm(residual(below) - below)) < 1e-12


def test_solution_is_independent_of_initial_guess(case):
    _, m, psi_coil, prof, psi_p0 = case
    a = solver.solve(m, prof, psi_coil, psi_p0, Ip=IP, max_newton=25)
    b = solver.solve(
        m, prof, psi_coil,
        solver.initial_guess(m, (1.0, 0.1), (0.30, 0.50), IP),
        Ip=IP, max_newton=25,
    )
    assert a.converged and b.converged
    rel = np.linalg.norm(np.asarray(a.psi_p) - np.asarray(b.psi_p))
    assert rel / np.linalg.norm(np.asarray(a.psi_p)) < 1e-8
