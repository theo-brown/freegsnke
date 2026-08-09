"""The loose-coupling handoff, when TORAX is importable.

Skipped in the jags venv, which deliberately does not depend on TORAX. TORAX's
own venv runs jags, so::

    PYTHONPATH=. <torax>/.venv/bin/python -m pytest tests/test_torax_bridge.py

is how these actually execute. The full field-by-field comparison against
TORAX's eqdsk parser lives in ``scripts/couple_torax.py``; what is pinned here
is that the handoff is well formed, since a malformed bundle fails inside TORAX
with an error that says nothing about which jags quantity was wrong.
"""

import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

pytest.importorskip("torax", reason="TORAX not installed in this venv")

from jags import critical, reach, torax_bridge  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402

CASE = pathlib.Path(__file__).parent.parent / "scripts/case_diverted.npz"


@pytest.fixture(scope="module")
def geo():
    d = np.load(CASE)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    psi = jnp.asarray(d["psi"])
    pa, pb = float(d["psi_axis"]), float(d["psi_bndry"])
    averager = make_flux_surface_averager(grid)
    label = reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi)
    axis, _ = critical.make_axis_finder(grid)[0](psi)
    pn_ref, f_ref = np.asarray(d["psinorm"]), np.asarray(d["fpol"])

    def F_of_psi(levels):
        return np.interp(
            (np.asarray(levels) - pa) / (pb - pa), pn_ref, f_ref
        )

    inter = torax_bridge.build_intermediates(
        grid, averager, psi, F_of_psi, pa, pb,
        float(axis[0]), float(axis[1]), n_surfaces=60, label=label, n_rho=25,
    )
    return inter, torax_bridge.build_geometry(inter)


def test_the_axis_row_matches_toraxs_own_convention(geo):
    """Index 0 is the magnetic axis, which has no contour.

    TORAX hard-codes limiting values there rather than tracing; if jags used
    different ones, every profile would differ at the axis for a reason that
    has nothing to do with the equilibrium.
    """
    inter, _ = geo
    R_axis = float(inter.R_in[0])
    assert inter.R_out[0] == pytest.approx(R_axis)
    assert float(inter.psi[0]) == 0.0
    assert float(inter.Phi[0]) == 0.0
    for k in ("int_dl_over_Bp", "flux_surf_avg_grad_psi",
              "flux_surf_avg_grad_psi2", "flux_surf_avg_grad_psi2_over_R2",
              "Ip_profile", "vpr"):
        assert float(getattr(inter, k)[0]) == 0.0, k
    assert float(inter.flux_surf_avg_1_over_R[0]) == pytest.approx(1 / R_axis)
    assert float(inter.flux_surf_avg_1_over_R2[0]) == pytest.approx(1 / R_axis**2)


def test_torax_accepts_the_bundle_and_the_metrics_are_finite(geo):
    """build_standard_geometry runs its own sanity checks -- monotonic volume
    above all -- so getting a StandardGeometry back is itself the assertion."""
    _, g = geo
    for k in ("vpr", "spr", "g0", "g1", "g2", "g3", "g2g3_over_rhon", "F"):
        v = np.asarray(getattr(g, k))
        assert np.all(np.isfinite(v)), k
    assert np.all(np.diff(np.asarray(g.volume)) > 0)
    assert np.all(np.asarray(g.vpr)[1:] > 0)
    assert np.all(np.diff(np.asarray(g.rho_norm)) > 0)


def test_the_provider_is_constant_in_time(geo):
    """Loose coupling means the geometry does not respond to transport.

    Pinned so that a later move to a time-dependent or tightly coupled provider
    is a deliberate change rather than an accident.
    """
    _, g = geo
    provider = torax_bridge.geometry_provider(g)
    a, b = provider(0.0), provider(1.0)
    np.testing.assert_array_equal(np.asarray(a.vpr), np.asarray(b.vpr))
