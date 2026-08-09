"""Flux-surface averages: correctness against closed form, and differentiability.

Circular surfaces are the test case because every quantity is known exactly. For
surfaces of radius r centred at R0, with psi = -r^2 so |grad psi| = 2r:

    contour_int R dl / |grad psi| = pi R0          -> int_dl_over_Bp
    V = 2 pi^2 R0 r^2  (Pappus),  A = pi r^2
    <1/R>   = 1 / R0
    <1/R^2> = 1 / (R0 sqrt(R0^2 - r^2))
    <|grad psi|>   = 2r
    <|grad psi|^2> = 4 r^2
"""

import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags.fsa import make_flux_surface_averager, safety_factor  # noqa: E402
from jags.grid import Grid  # noqa: E402
from jags.reach import make_reachability  # noqa: E402

R0 = 1.0
BOX = np.array([[0.35, -0.6], [1.65, -0.6], [1.65, 0.6], [0.35, 0.6]])
PSI_SCALE = 0.25


def circular(n):
    """Grid and a psi whose contours are circles of radius r about (R0, 0)."""
    g = Grid(0.3, 1.7, -0.7, 0.7, n, n, BOX)
    psi = jnp.asarray(-((g.R - R0) ** 2 + g.Z**2))
    return g, psi


def exact(r):
    return dict(
        volume=2 * np.pi**2 * R0 * r**2,
        area=np.pi * r**2,
        int_dl_over_Bp=np.pi * R0,
        avg_1_over_R2=1 / (R0 * np.sqrt(R0**2 - r**2)),
        avg_grad_psi=2 * r,
        avg_grad_psi2=4 * r**2,
    )


# Resolved surfaces: r/a > ~0.3, where the grid represents the contour. The
# domain minor radius is ~0.5, so these are r/a of 0.3 to 0.8.
RESOLVED = np.array([0.15, 0.18, 0.25, 0.32, 0.40])


def test_average_of_unity_is_one(case):
    """<1> = 1 identically -- the kernel normalisation cancels in the ratio.

    Structural, so it holds at any eps and on any field, including the solver's
    own solution rather than a contrived one.
    """
    g, m, psi_coil, _, psi0 = case
    psi = psi0.reshape(g.nR, g.nZ) + psi_coil
    average, _, _ = make_flux_surface_averager(g, eps=0.02)
    levels = jnp.linspace(float(psi.min()) * 0.5, float(psi.max()) * 0.9, 6)
    got = average(psi, jnp.ones_like(psi), levels, float(jnp.ptp(psi)))
    np.testing.assert_allclose(np.asarray(got), 1.0, rtol=1e-12)


# Tolerances are per-quantity because they genuinely differ by two orders.
# <|grad psi|> is the slow one: the measure grows like r across the kernel band,
# biasing its centroid outward, which survives at O(eps^2 / r^2) for a quantity
# linear in r. Squared and inverse quantities are far less affected.
@pytest.mark.parametrize(
    "quantity,tol",
    [("avg_1_over_R2", 1e-4), ("avg_grad_psi2", 1e-3), ("avg_grad_psi", 1e-2)],
)
def test_ratios_match_closed_form(quantity, tol):
    """<X> against closed form on resolved surfaces.

    ``<1/R>`` is deliberately excluded: for circles centred at R0 *any*
    radially symmetric kernel returns 1/R0 by symmetry, so it would pass even if
    the weighting were wrong.
    """
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g, eps=0.01)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), PSI_SCALE)

    want = np.array([exact(r)[quantity] for r in RESOLVED])
    err = np.max(np.abs(np.asarray(getattr(fs, quantity)) - want) / want)
    assert err < tol, f"{quantity}: relative error {err:.2e}"


def test_enclosed_integrals_match_closed_form():
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g, eps=0.01)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), PSI_SCALE)

    for key in ("volume", "area"):
        want = np.array([exact(r)[key] for r in RESOLVED])
        got = np.asarray(getattr(fs, key))
        assert np.max(np.abs(got - want) / want) < 1e-4, key


def test_absolute_contour_integral_is_the_weakest_quantity():
    """``int_dl_over_Bp`` carries ~1%, an order worse than the ratios.

    Pinned rather than hidden: it is the one quantity whose kernel normalisation
    does not cancel, and a coupled transport solve would inherit this error in
    dV/dpsi. See the module docstring for why it cannot simply be tightened.
    """
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g, eps=0.01)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), PSI_SCALE)
    err = np.max(np.abs(np.asarray(fs.int_dl_over_Bp) - np.pi * R0) / (np.pi * R0))
    assert err < 2e-2
    assert err > 1e-4, "if this is now tight, the docstring is stale"


def test_near_axis_surfaces_are_unresolved_and_refining_does_not_help():
    """The innermost surfaces fail, and identically at both resolutions.

    A surface of radius 0.03 spans ~3 cells at 129x129. The failure is that the
    grid cannot represent the curve, not that the kernel is wrong -- which is
    why doubling the resolution changes nothing. Documented so nobody trusts
    the inner cells of a coupled transport grid.
    """
    errs = []
    for n in (65, 129):
        g, psi = circular(n)
        _, surfaces, _ = make_flux_surface_averager(g, eps=0.01)
        fs = surfaces(psi, jnp.asarray([-(0.03**2)]), PSI_SCALE)
        errs.append(
            abs(float(fs.volume[0]) - exact(0.03)["volume"]) / exact(0.03)["volume"]
        )

    assert errs[0] > 0.5, "expected the innermost surface to be badly wrong"
    assert abs(errs[0] - errs[1]) / errs[0] < 0.05, (
        f"refining changed the error {errs[0]:.3f} -> {errs[1]:.3f}, so this is "
        "grid resolution after all and the docstring is wrong"
    )


@pytest.mark.parametrize(
    "quantity", ["avg_1_over_R2", "avg_grad_psi2", "volume", "int_dl_over_Bp"]
)
def test_differentiable_with_respect_to_psi(quantity):
    """The whole point: exact gradients, which contour tracing cannot give.

    Checked against central differences on random perturbation directions, which
    is what a coupled Newton solve would actually exercise.
    """
    g, psi = circular(65)
    _, surfaces, _ = make_flux_surface_averager(g, eps=0.02)
    levels = jnp.asarray(-(np.array([0.18, 0.30]) ** 2))

    def scalar(p):
        return jnp.sum(getattr(surfaces(p, levels, PSI_SCALE), quantity))

    grad = np.asarray(jax.grad(scalar)(psi))
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).max() > 0, "gradient is identically zero"

    rng = np.random.default_rng(0)
    for _ in range(3):
        v = rng.standard_normal(psi.shape)
        v /= np.linalg.norm(v)
        h = 1e-6
        fd = float((scalar(psi + h * v) - scalar(psi - h * v)) / (2 * h))
        ad = float((grad * v).sum())
        assert abs(fd - ad) <= 1e-5 * max(1.0, abs(fd)), f"{fd:.6e} vs {ad:.6e}"


def test_diverted_level_sets_need_the_reachability():
    """On a real diverted equilibrium, a raw level set of psi is not a surface.

    Contours near the separatrix reappear in the divertor legs, and the kernel
    sums over those too, so q comes out far too large. ``reach.py`` is the fix,
    and the two ways of applying it must both help, with ``label`` -- which
    replaces the flux label outright rather than fading cells out -- ahead.

    Tolerances here are loose *because the reference grid is 65x65*: the
    committed case is the one the solver comparison uses, and at that resolution
    the co-area quadrature is barely resolved. The convergence study behind the
    numbers in ``fsa.py``'s docstring runs the same check at 129 and 193, where
    the median falls to 9.1e-3 and 2.0e-3. This test pins the *ordering*, which
    is resolution-independent, plus a ceiling.
    """
    d = np.load(pathlib.Path(__file__).parent.parent / "scripts/case_diverted.npz")
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    g = Grid(float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
             int(d["nR"]), int(d["nZ"]), limiter)

    psi = jnp.asarray(d["psi"])
    pa, pb = float(d["psi_axis"]), float(d["psi_bndry"])
    pn, ref = np.asarray(d["psinorm"]), np.asarray(d["q"])
    levels = jnp.asarray(pa + pn * (pb - pa))
    scale = abs(pa - pb)

    _, surfaces, _ = make_flux_surface_averager(g, eps=0.01)
    m = make_reachability(g, n_samples=64, beta_norm=2e5)(psi)

    def err(**kw):
        q = np.asarray(safety_factor(surfaces(psi, levels, scale, **kw),
                                     jnp.asarray(d["fpol"])))
        return float(np.median(np.abs(q - ref) / ref))

    raw = err()
    mask = err(mask=jax.nn.sigmoid((m - pb) / (0.02 * scale)))
    label = err(label=m)

    assert raw > 0.2, f"raw level set unexpectedly accurate ({raw:.2e})"
    assert label < mask < raw, f"{label:.2e} !< {mask:.2e} !< {raw:.2e}"
    assert label < 0.05


def test_mask_restricts_the_average(case):
    """A mask must exclude masked cells entirely, not merely downweight them."""
    g, psi = circular(65)
    average, _, _ = make_flux_surface_averager(g, eps=0.02)
    levels = jnp.asarray([-(0.25**2)])

    upper = jnp.asarray((g.Z > 0).astype(float))
    both = average(psi, jnp.asarray(g.Z), levels, PSI_SCALE)
    top = average(psi, jnp.asarray(g.Z), levels, PSI_SCALE, mask=upper)

    # <Z> vanishes by symmetry over the whole surface, but not over half of it.
    assert abs(float(both[0])) < 1e-12
    assert float(top[0]) > 0.1
