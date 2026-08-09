"""Reachability: the fix for diverted equilibria.

``reach.make_reachability`` replaces psi by its running minimum along the ray
from the magnetic axis, so that flux lobes lying beyond a null get no current.
"""

import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import profiles, reach, solver  # noqa: E402
from jags.grid import Grid  # noqa: E402

from conftest import IP  # noqa: E402

_MASTU = pathlib.Path(__file__).parent.parent / "scripts" / "case_limited.npz"


def test_is_identity_on_a_monotone_peak(case):
    """Where psi decreases outward from the axis, m == psi.

    This is what makes the substitution free for a limited plasma: the ray
    minimum is attained at its endpoint, so the core is untouched.
    """
    g, _, _, _, _ = case
    R0, Z0 = 0.9, 0.05
    psi = jnp.asarray(np.exp(-(((g.R - R0) / 0.45) ** 2 + ((g.Z - Z0) / 0.55) ** 2)))
    m = np.asarray(reach.make_reachability(g, n_samples=32, beta_norm=2e5)(psi))

    # Inside the limiter and far enough from the axis for the ray to have length
    # -- outside the limiter no current is placed, and the bicubic stencil clamps
    # at the domain edge.
    where = g.limiter_mask() & (
        np.hypot(g.R - R0, g.Z - Z0) > 4 * max(g.dR, g.dZ)
    )
    assert np.abs(m - np.asarray(psi))[where].max() < 1e-4


def test_removes_a_disconnected_high_flux_lobe(case):
    """A blob of high flux unreachable from the axis is cut down.

    The lobe sits at the same flux level as the core but behind a trough, so no
    threshold on psi alone can separate them -- exactly the diverted-plasma
    failure this exists to fix.
    """
    g, _, _, _, _ = case
    core = np.exp(-(((g.R - 0.75) / 0.28) ** 2 + (g.Z / 0.35) ** 2))
    lobe = np.exp(-(((g.R - 1.45) / 0.10) ** 2 + ((g.Z - 0.6) / 0.10) ** 2))
    psi = jnp.asarray(core + 0.95 * lobe)

    m = np.asarray(reach.make_reachability(g, n_samples=48, beta_norm=2e5)(psi))
    psi_np = np.asarray(psi)

    assert m[lobe > 0.5 * lobe.max()].max() < 0.5 * psi_np[lobe > 0.5 * lobe.max()].max()
    core_cells = core > 0.5 * core.max()
    assert np.abs(m - psi_np)[core_cells].max() < 0.05 * psi_np[core_cells].max()


@pytest.mark.parametrize("mode", ["jacfwd", "matrix_free"])
def test_newton_stays_quadratic_with_reachability(case, mode):
    """Reachability must not cost the convergence rate.

    It makes ``dJtor/dpsi`` non-diagonal, so any Jacobian assuming a pointwise
    map degrades to linear. Both supported modes take the step from the exact
    Jacobian action or the full assembled matrix, and stay quadratic.
    """
    _, m, psi_coil, prof, psi0 = case
    rr = reach.make_reachability(m.grid, n_samples=32, beta_norm=2e5)
    r = solver.solve(
        m, prof, psi_coil, psi0, Ip=IP, max_newton=25,
        jacobian_mode=mode, reachability=rr,
    )
    assert r.converged
    assert len(r.residual_history) <= 8, "expected quadratic, not linear"
    for prev, nxt in zip(r.residual_history[1:-1], r.residual_history[2:]):
        assert nxt <= max(50.0 * prev**2, 1e-14)


@pytest.mark.skipif(
    not _MASTU.exists(), reason="run scripts/dump_freegsnke_case.py first"
)
def test_is_a_no_op_on_a_confined_limited_equilibrium():
    """On a real limited plasma, reachability must not change the answer.

    Deliberately not tested on the shared fixture, whose currents do not confine
    (see ``conftest.case``); the claim only means anything on an equilibrium
    bounded by its own flux.
    """
    d = np.load(_MASTU)
    g = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]),
        np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1),
    )
    mach = solver.build_machine(g, coil_RZ=None)
    prof = profiles.lao85(
        float(d["psi_axis"]), float(d["psi_bndry"]), d["alpha_full"],
        d["beta_full"], float(d["L"]), float(d["fvac"]), float(d["Raxis"]),
    )
    ref_j, Ip = np.asarray(d["jtor"]), float(d["Ip"])
    centre = (
        float((ref_j * g.R).sum() / ref_j.sum()),
        float((ref_j * g.Z).sum() / ref_j.sum()),
    )
    psi0 = solver.initial_guess(mach, centre, (0.45, 0.9), Ip)
    psi_coil = jnp.asarray(d["tokamak_psi"])

    kw = dict(Ip=Ip, max_newton=40)
    a = solver.solve(mach, prof, psi_coil, psi0, **kw)
    b = solver.solve(
        mach, prof, psi_coil, psi0,
        reachability=reach.make_reachability(g, n_samples=32, beta_norm=2e5),
        **kw,
    )
    assert a.converged and b.converged
    rel = np.linalg.norm(np.asarray(a.psi_p) - np.asarray(b.psi_p))
    assert rel / np.linalg.norm(np.asarray(a.psi_p)) < 5e-3
