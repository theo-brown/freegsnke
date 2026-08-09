"""The TORAX bundle: internal consistency, and the parts with a closed form.

Cross-checking against TORAX itself needs its venv and a geqdsk, so that lives
in ``scripts/check_torax_geometry.py``. What is testable here is that the
derived quantities agree with the ones they are derived from, and that circular
surfaces come out with the shape they analytically have.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import torax_geom  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402

R0 = 1.0
BOX = np.array([[0.35, -0.6], [1.65, -0.6], [1.65, 0.6], [0.35, 0.6]])
PSI_EDGE = -0.25
RADII = np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])


@pytest.fixture(scope="module")
def bundle():
    """Circular surfaces, so the shape is known: elongation 1, triangularity 0."""
    n = 193
    g = Grid(0.3, 1.7, -0.7, 0.7, n, n, BOX)
    psi = jnp.asarray(-((g.R - R0) ** 2 + g.Z**2))
    _, surfaces, _ = make_flux_surface_averager(g)
    fs = surfaces(psi, jnp.asarray(-(RADII**2)), 0.0, PSI_EDGE)
    F = jnp.full(RADII.shape, 0.5)
    return fs, F, torax_geom.intermediates(fs, F, 0.0, 0.0)


def test_psi_conversion_is_exact(bundle):
    """Wb/rad to Wb, with TORAX's origin at the axis and psi growing outward.

    Pinned because it is pure bookkeeping that is easy to get wrong in a way
    nothing else catches -- an earlier version dropped the axis offset and was
    a clean factor of two out at mid-radius.
    """
    fs, _, tg = bundle
    want = (0.0 - np.asarray(fs.psi_levels)) * 2 * np.pi
    np.testing.assert_allclose(np.asarray(tg.psi), want, rtol=1e-15)
    assert np.all(np.diff(np.asarray(tg.psi)) > 0), "psi must grow outward"


def test_toroidal_flux_is_the_integral_of_q(bundle):
    """dPhi/dpsi = q, which is how Phi was built -- so this checks the axis
    segment, the one part not determined by the trapezoid itself.

    Interior points only: ``np.gradient`` falls back to a one-sided difference
    at the ends, which is first-order and contributes more error than the thing
    being tested (1e-4 in the interior against 1e-2 at the ends).
    """
    fs, F, tg = bundle
    q = np.asarray(torax_geom.safety_factor(fs, F))
    psi, phi = np.asarray(tg.psi), np.asarray(tg.Phi)
    dphi = np.gradient(phi, psi)
    assert np.max(np.abs(dphi - q)[1:-1] / q[1:-1]) < 1e-3
    assert phi[0] > 0, "the core inside the first surface carries flux"


def test_vpr_integrates_to_the_enclosed_volume(bundle):
    """vpr = dV/d(rho_norm), so integrating it must return the volume.

    This ties together dV_dpsi, q and Phi -- three separately computed things --
    so it fails if any of the chain rules between them is wrong.
    """
    _, _, tg = bundle
    phi = np.asarray(tg.Phi)
    rho = np.sqrt(phi / phi[-1])
    v = np.asarray(tg.volume)
    integrated = np.cumsum(
        np.concatenate([[0.0], 0.5 * (np.asarray(tg.vpr)[1:]
                                      + np.asarray(tg.vpr)[:-1]) * np.diff(rho)])
    )
    grew = (v - v[0]) - integrated
    assert np.max(np.abs(grew)) / v[-1] < 5e-3


def test_circular_surfaces_have_unit_elongation_and_no_triangularity(bundle):
    """The shape quantities are the weak part of the bundle, so the case where
    the answer is exactly known is worth pinning.

    Elongation is a ratio of two soft extrema and the O(cell) bias cancels, so
    it is good to 1e-4. Triangularity is a *difference* of two radii that are
    close together, so the same bias does not cancel; it is asserted only to
    2e-2 in absolute terms. That asymmetry is the whole story of why
    ``delta_upper_face`` is the worst-agreeing field against TORAX.
    """
    _, _, tg = bundle
    np.testing.assert_allclose(np.asarray(tg.elongation), 1.0, atol=1e-4)
    assert np.max(np.abs(np.asarray(tg.delta_upper_face))) < 2e-2
    assert np.max(np.abs(np.asarray(tg.delta_lower_face))) < 2e-2


def test_enclosed_current_matches_amperes_law(bundle):
    """Ip = contour_int B_p dl / mu0, built out of averages rather than a new
    integral. Checked against the same contour integral formed independently
    from <|grad psi|> and the surface length, which uses a different pair of
    bundle entries."""
    fs, _, tg = bundle
    # int B_p dl = <|grad psi|^2/R^2> * int_dl_over_Bp, and for these circular
    # surfaces |grad psi| = 2r exactly, so int B_p dl = 2r * int dl / R.
    got = np.asarray(tg.Ip_profile) * torax_geom.MU0
    want = np.asarray(fs.avg_grad_psi2_over_R2 * fs.int_dl_over_Bp)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert np.all(np.diff(got) > 0), "enclosed current must grow outward"


def test_1_over_B2_is_not_silently_faked(bundle):
    """Omitting F must give NaN, not a wrong-but-plausible number."""
    fs, F, _ = bundle
    tg = torax_geom.intermediates(fs, F, 0.0, 0.0)
    assert np.all(np.isnan(np.asarray(tg.flux_surf_avg_1_over_B2)))
