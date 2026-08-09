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
PSI_EDGE = -0.25  # plasma is {psi > psi_edge}; the axis is at psi = 0


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
    average, _, _ = make_flux_surface_averager(g)
    lo, hi = float(psi.min()) * 0.5, float(psi.max()) * 0.9
    levels = jnp.linspace(lo, hi, 6)
    got = average(psi, jnp.ones_like(psi), levels, hi, lo)
    np.testing.assert_allclose(np.asarray(got), 1.0, rtol=1e-12)


# Tolerances are per-quantity because they still differ by two orders, but all
# of them tightened by 2-3 orders when the width moved from a fixed fraction of
# the flux range to a fixed number of cells with a fourth-order kernel.
@pytest.mark.parametrize(
    "quantity,tol",
    [("avg_1_over_R2", 1e-6), ("avg_grad_psi2", 1e-5), ("avg_grad_psi", 1e-4)],
)
def test_ratios_match_closed_form(quantity, tol):
    """<X> against closed form on resolved surfaces.

    ``<1/R>`` is deliberately excluded: for circles centred at R0 *any*
    radially symmetric kernel returns 1/R0 by symmetry, so it would pass even if
    the weighting were wrong.
    """
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), 0.0, PSI_EDGE)

    want = np.array([exact(r)[quantity] for r in RESOLVED])
    err = np.max(np.abs(np.asarray(getattr(fs, quantity)) - want) / want)
    assert err < tol, f"{quantity}: relative error {err:.2e}"


def test_enclosed_integrals_match_closed_form():
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), 0.0, PSI_EDGE)

    for key in ("volume", "area"):
        want = np.array([exact(r)[key] for r in RESOLVED])
        got = np.asarray(getattr(fs, key))
        assert np.max(np.abs(got - want) / want) < 1e-6, key


def test_absolute_contour_integral_is_the_weakest_quantity():
    """``int_dl_over_Bp`` is still the weakest, but only by an order now.

    It is the one quantity whose kernel normalisation does not cancel, so it
    cannot benefit from the ratio the way every ``<X>`` does, and a coupled
    transport solve inherits it through ``dV/dpsi``. Both bounds are asserted:
    the lower one fires if the gap ever closes, which would mean the docstring's
    account of *why* it is weakest has gone stale.
    """
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g)
    fs = surfaces(psi, jnp.asarray(-(RESOLVED**2)), 0.0, PSI_EDGE)
    err = np.max(np.abs(np.asarray(fs.int_dl_over_Bp) - np.pi * R0) / (np.pi * R0))
    ratio = np.max(np.abs(np.asarray(fs.avg_1_over_R2)
                          - np.array([exact(r)["avg_1_over_R2"] for r in RESOLVED]))
                   / np.array([exact(r)["avg_1_over_R2"] for r in RESOLVED]))
    assert err < 1e-4
    assert err > 10 * ratio, "no longer weakest -- the docstring is stale"


def test_near_axis_surfaces_converge_with_resolution():
    """The innermost surfaces are the hard ones, but they are *not* a dead end.

    With a fixed-width kernel this test asserted the opposite -- the error at
    r = 0.03 was above 50% and identical at 65x65 and 129x129, so refining was
    genuinely useless. Setting the width in cells changed that: the same surface
    now converges, 3.9e-2 -> 1.7e-2 -> 3.2e-3 -> 2.9e-4 across 65/129/193/257.
    Kept as a convergence test rather than deleted, because the earlier claim
    was wrong and this is what disproves it.
    """
    errs = []
    for n in (65, 129, 193, 257):
        g, psi = circular(n)
        _, surfaces, _ = make_flux_surface_averager(g)
        fs = surfaces(psi, jnp.asarray([-(0.03**2)]), 0.0, PSI_EDGE)
        errs.append(
            abs(float(fs.volume[0]) - exact(0.03)["volume"]) / exact(0.03)["volume"]
        )

    assert all(b < a for a, b in zip(errs, errs[1:])), f"not monotone: {errs}"
    assert errs[-1] < 1e-3, f"257x257 should resolve r=0.03: {errs[-1]:.2e}"
    assert errs[0] / errs[-1] > 50, f"barely converging: {errs}"


def test_n_eff_predicts_whether_a_surface_can_be_trusted():
    """The diagnostic has to earn its place by actually tracking the error.

    ``n_eff`` is the participation ratio of the kernel weights -- how many cells
    carry the surface. It is the number a coupled transport solve would use to
    decide which inner surfaces to extrapolate instead of believe.
    """
    g, psi = circular(129)
    _, surfaces, _ = make_flux_surface_averager(g)
    radii = np.array([0.03, 0.05, 0.08, 0.12, 0.20, 0.30, 0.40])
    fs = surfaces(psi, jnp.asarray(-(radii**2)), 0.0, PSI_EDGE)

    n_eff = np.asarray(fs.n_eff)
    want = np.array([exact(r)["volume"] for r in radii])
    err = np.abs(np.asarray(fs.volume) - want) / want

    assert np.all(np.diff(n_eff) > 0), "n_eff should grow outward"
    assert err[n_eff > 80].max() < 1e-5, "well-sampled surfaces must be accurate"
    assert err[n_eff < 40].max() > 1e-3, "starved surfaces must look starved"


def test_a_level_on_the_plasma_edge_stays_finite():
    """The separatrix itself must not poison the whole bundle with NaN.

    The edge cap drives the width to zero for a level sitting exactly on
    ``psi_edge``, where the true contour integral diverges anyway. Without a
    floor that is a division by zero, and one bad level NaNs an entire coupled
    solve, so the floor is not cosmetic.
    """
    g, psi = circular(65)
    _, surfaces, _ = make_flux_surface_averager(g)
    levels = jnp.asarray([PSI_EDGE, -(0.25**2), -(0.4**2)])
    fs = surfaces(psi, levels, 0.0, PSI_EDGE)

    for name, v in zip(fs._fields, fs):
        assert np.all(np.isfinite(np.asarray(v))), name


@pytest.mark.parametrize(
    "quantity", ["avg_1_over_R2", "avg_grad_psi2", "volume", "int_dl_over_Bp"]
)
def test_differentiable_with_respect_to_psi(quantity):
    """The whole point: exact gradients, which contour tracing cannot give.

    Checked against central differences on random perturbation directions, which
    is what a coupled Newton solve would actually exercise.
    """
    g, psi = circular(65)
    _, surfaces, _ = make_flux_surface_averager(g)
    levels = jnp.asarray(-(np.array([0.18, 0.30]) ** 2))

    def scalar(p):
        return jnp.sum(getattr(surfaces(p, levels, 0.0, PSI_EDGE), quantity))

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

    _, surfaces, _ = make_flux_surface_averager(g)
    m = make_reachability(g, n_samples=64, beta_norm=2e5)(psi)

    def err(**kw):
        q = np.asarray(safety_factor(surfaces(psi, levels, pa, pb, **kw),
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
    average, _, _ = make_flux_surface_averager(g)
    levels = jnp.asarray([-(0.25**2)])

    upper = jnp.asarray((g.Z > 0).astype(float))
    both = average(psi, jnp.asarray(g.Z), levels, 0.0, PSI_EDGE)
    top = average(psi, jnp.asarray(g.Z), levels, 0.0, PSI_EDGE, mask=upper)

    # <Z> vanishes by symmetry over the whole surface, but not over half of it.
    assert abs(float(both[0])) < 1e-12
    assert float(top[0]) > 0.1
