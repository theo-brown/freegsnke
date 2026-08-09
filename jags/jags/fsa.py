"""Flux-surface averages without contour tracing.

A transport solver wants flux-surface-averaged geometry -- TORAX's
``StandardGeometryIntermediates`` asks for ``int_dl_over_Bp``,
``flux_surf_avg_1_over_R2``, ``flux_surf_avg_grad_psi2`` and friends
(``torax/_src/geometry/standard_geometry.py:183-198``). Computed the usual way
each is a contour integral: trace the closed curve psi = const, then integrate
along it. That is exactly the operation this package avoids, and putting it back
would make the coupled residual non-differentiable again.

The co-area formula removes the contour. For any quantity X,

    d/dpsi0  integral_{psi > psi0} X dV  =  -contour_integral X dl |grad psi|^-1 dV-weight

so with the volume element ``dV = 2 pi R dR dZ`` the flux-surface average

    <X> = contour_int X R dl / |grad psi|  /  contour_int R dl / |grad psi|

is just a ratio of derivatives of *volume* integrals. Differentiating a
smoothed indicator gives a smoothed delta, so each average reduces to a
kernel-weighted sum over the whole grid:

    <X> = sum_i w_i X_i / sum_i w_i,     w_i = 2 pi R_i delta_eps(psi_i - psi0) dA

No contour, no topology, smooth in psi by construction -- the same device as the
core mask in ``profiles.py`` and the ray minimum in ``reach.py``.

Three choices decide whether that is accurate, and all three matter:

**The kernel width is set in cells, not in flux.** A width fixed as a fraction
of the flux range is grid-blind: on the 65x65 MAST-U case a width of 1% of the
range spans 0.19 of a cell, so the quadrature samples almost nothing and
aliases. The width here is instead a multiple of ``<dpsi_cell>``, the flux
change across one cell *along* ``grad psi``,

    dpsi_cell = sqrt((dpsi/dR dR)^2 + (dpsi/dZ dZ)^2)

averaged over the surface itself, which handles anisotropic cells exactly. That
is circular -- the average needs a width -- so it is a two-line fixed point,
converged in two passes and independent of its seed to 1e-4 at 193x193. It is
differentiated through rather than frozen; see ``_width`` for why freezing it,
which is the more obvious choice, is the wrong one here.

**The kernel is fourth order.** A smoothed delta of width w smears neighbouring
surfaces together, an O(w^2) bias, which is what forces w small and puts it back
in conflict with the quadrature. A kernel whose second moment vanishes removes
that term, so the band can be a full cell wide with no smearing penalty. Using a
Gaussian rather than the logistic makes the construction exact and closed-form:
with ``phi``, ``Phi`` the normal density and CDF,

    surface  = (3 - u^2) phi(u) / 2w        enclosed = Phi(u) + u phi(u) / 2

and ``d(enclosed)/dpsi0 = -surface`` identically, so ``dV_dpsi`` stays the exact
derivative of ``volume``. ``order=6`` is available and is slightly better at the
optimum, but needs a wider band to realise its order and degrades faster when it
does not get one. Gaussian tails also decay far faster than logistic ones, which
matters in a diverted equilibrium where a fat tail reaches into the legs.

**The width is capped near the plasma edge.** Approaching the separatrix
``|grad psi| -> 0`` at the X-point, so a band of fixed *flux* width becomes
unboundedly wide in *space* and piles weight onto the X-point, where the
integrand 1/|grad psi| is largest. Capping the width at ``max_span`` times the
flux distance to the edge fixes it: at psi_N = 0.95 and 193x193 the error in q
drops from 2.6e-1 to 9.2e-3. The cap is deliberately **not** applied on the axis
side -- there is no pathology there, the level sets are just nested, and
clamping makes the innermost surfaces worse by starving them of cells (2.2e-3 ->
2.2e-1 at 129x129). It is a rail against a known singularity, not a knob: on
circular surfaces it never activates and changes nothing.

Accuracy, measured two ways. Against circular surfaces where every quantity is
known in closed form (``test_fsa.py``), worst relative error over all of
``volume``, ``area``, ``int_dl_over_Bp``, ``<1/R^2>``, ``<|grad psi|>`` and
``<|grad psi|^2>``, on surfaces of r/a from 0.3 to 0.8:

    grid       this scheme    fixed-width logistic
    65x65      4.6e-4         ~5e-2
    129x129    3.1e-5         ~1.2e-2
    193x193    6.8e-6         ~1.2e-2

Against a **real diverted equilibrium** -- FreeGSNKE's MAST-U case, checked
against FreeGS4E's ray-traced ``q`` and TORAX's ``contourpy``-based eqdsk parser
on the same psi (``scripts/check_fsa.py``) -- relative error in ``q``, with the
previous fixed-width logistic scheme in brackets:

    grid       median            psi_N <= 0.8      worst (at psi_N = 0.95)
    65x65      3.7e-3 (4.0e-2)   3.0e-2 (1.7e-1)   6.2e-2 (1.7e-1)
    129x129    1.2e-3 (9.1e-3)   5.1e-3 (2.4e-2)   4.2e-2 (1.0e-1)
    193x193    1.2e-3 (2.0e-3)   1.2e-3 (6.8e-3)   9.2e-3 (1.1e-1)

For scale, the two *tracing* codes disagree with each other by 2.2e-2, 1.1e-2
and 9.4e-3 at those resolutions. At 193x193 jags differs from TORAX by 8.8e-3,
which is *equal to* the TORAX-FreeGS4E spread of 9.4e-3: the comparison has hit
the floor set by the references, and nothing further can be concluded from them.
129x129 now beats what 193x193 achieved before.

What is left is resolution, and the bundle says so rather than hiding it.
``n_eff``, the participation ratio ``(sum w)^2 / sum w^2``, counts the cells
actually carrying a surface; ``n_cells`` is the width in cells, which drops
below ``width_cells`` exactly when the edge cap bites. On the circular case
``n_eff`` tracks the error closely enough to be used as a gate: above 80 every
quantity is inside 1e-5, below 40 none is better than 1e-3. A coupled transport
solve should extrapolate the surfaces that fail that test rather than believe
them.

One earlier claim in this file was **wrong** and is worth stating plainly: the
innermost surfaces were described as a hard limit of any grid method, on the
evidence that a surface of r = 0.03 was more than 50% wrong and *identically* so
at 65x65 and 129x129. That was a property of the fixed-width kernel, not of the
grid. With the width set in cells the same surface converges -- 3.9e-2, 1.7e-2,
3.2e-3, 2.9e-4 at 65/129/193/257 -- and on MAST-U the worst error moved off the
innermost surface entirely, onto the separatrix.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.scipy.stats import norm

from .grid import Grid

# Floor on the edge cap, as a fraction of the uncapped cell-based width. Only
# active for a level within max_span * this of psi_edge, i.e. essentially on the
# separatrix, where without it the width would be zero and every quantity NaN.
MIN_WIDTH_FRACTION = 0.25


class FluxSurfaces(NamedTuple):
    """Flux-surface geometry on a set of flux labels.

    Names follow TORAX's ``StandardGeometryIntermediates`` where they
    correspond, so the mapping across is direct. The last three are diagnostics:
    they say whether the rest can be believed.
    """

    psi_levels: jnp.ndarray  # (n,) flux labels the surfaces are taken at
    volume: jnp.ndarray  # (n,) enclosed volume [m^3]
    area: jnp.ndarray  # (n,) enclosed poloidal cross-section [m^2]
    dV_dpsi: jnp.ndarray  # (n,) dV/dpsi [m^3 / Wb per rad]
    int_dl_over_Bp: jnp.ndarray  # (n,) contour integral of dl / B_p [m/T]
    avg_1_over_R: jnp.ndarray  # (n,) <1/R>
    avg_1_over_R2: jnp.ndarray  # (n,) <1/R^2>
    avg_grad_psi: jnp.ndarray  # (n,) <|grad psi|>
    avg_grad_psi2: jnp.ndarray  # (n,) <|grad psi|^2>
    avg_grad_psi2_over_R2: jnp.ndarray  # (n,) <|grad psi|^2 / R^2>
    width: jnp.ndarray  # (n,) kernel width used [Wb/rad]
    n_cells: jnp.ndarray  # (n,) width in cells; < width_cells => edge cap bit
    n_eff: jnp.ndarray  # (n,) cells carrying the surface; small => unresolved


def _kernel(u, order):
    """``(enclosed, shape)`` with ``d(enclosed)/du == shape`` exactly.

    ``shape`` is a kernel of the stated order: symmetric, unit mass, and with
    every moment up to ``order - 1`` vanishing, so smearing across neighbouring
    surfaces cancels to that order. Orders above 2 take negative values in the
    tails, which is what makes the cancellation possible.
    """
    p = norm.pdf(u)
    if order == 2:
        return norm.cdf(u), p
    if order == 4:
        return norm.cdf(u) + 0.5 * u * p, 0.5 * (3.0 - u**2) * p
    if order == 6:
        return (
            norm.cdf(u) + p * (7.0 * u - u**3) / 8.0,
            p * (15.0 - 10.0 * u**2 + u**4) / 8.0,
        )
    raise ValueError(f"order must be 2, 4 or 6, got {order}")


def make_flux_surface_averager(
    grid: Grid,
    width_cells: float = 0.85,
    order: int = 4,
    max_span: float = 0.25,
    n_passes: int = 3,
):
    """Build flux-surface-average machinery for a grid.

    Parameters
    ----------
    width_cells : kernel width as a multiple of the flux change across one cell
        along ``grad psi``. The optimum is broad -- 0.7 to 1.2 changes the error
        by under 3x at 129x129 -- so this is a shape parameter, not a fit.
    order : 2, 4 or 6. See the module docstring; 4 is the default because 6 is
        less forgiving when the band is narrow.
    max_span : cap on the width as a fraction of the flux distance to
        ``psi_edge``. Set to ``None`` to disable. Applies on the edge side only.
    n_passes : iterations of the width fixed point. Two suffice; three makes the
        result seed-independent on coarse grids at the cost of one more kernel
        evaluation.

    Returns ``(average, surfaces, grad_psi)``:

    ``average(psi, X, psi_levels, psi_axis, psi_edge, mask=None, label=None)``
        <X> on each level, for an arbitrary field X on the grid.
    ``surfaces(psi, psi_levels, psi_axis, psi_edge, mask=None, label=None)``
        the ``FluxSurfaces`` bundle.
    ``grad_psi(psi)``
        ``(dpsi/dR, dpsi/dZ)``, exposed because callers usually need B_p too.

    ``psi_axis`` and ``psi_edge`` bracket the plasma. Only ``psi_edge`` is used
    numerically, by ``max_span``; ``psi_axis`` sets the seed width and is
    reported back through ``n_cells``.

    ``label`` is the field whose level sets define the surfaces, defaulting to
    ``psi`` itself. It exists for **diverted** equilibria, where a level set of
    psi is not one closed curve: contours near the separatrix reappear in the
    divertor legs, and the kernel would sum over those too. Passing
    ``label=reach.make_reachability(grid)(psi)`` restricts the surfaces to the
    core -- the reachability equals psi there and drops below the edge in every
    lobe, so the same device that fixes ``Jtor`` fixes the averages. The
    gradient stays that of the physical ``psi`` either way, so ``B_p`` is not
    contaminated by the softmin.
    """
    R = jnp.asarray(grid.R)
    dR, dZ, dA = float(grid.dR), float(grid.dZ), float(grid.dA)
    two_pi_R_dA = 2.0 * jnp.pi * R * dA

    def grad_psi(psi):
        """Fourth-order central differences, matching the Delstar operator.

        Second-order one-sided values are kept on the two outermost rings; the
        plasma never reaches them.
        """
        gR = jnp.gradient(psi, dR, axis=0)
        gZ = jnp.gradient(psi, dZ, axis=1)
        gR = gR.at[2:-2, :].set(
            (psi[:-4, :] - 8 * psi[1:-3, :] + 8 * psi[3:-1, :] - psi[4:, :])
            / (12 * dR)
        )
        gZ = gZ.at[:, 2:-2].set(
            (psi[:, :-4] - 8 * psi[:, 1:-3] + 8 * psi[:, 3:-1] - psi[:, 4:])
            / (12 * dZ)
        )
        return gR, gZ

    def _weights(label, psi_levels, width, mask):
        """Surface weights and enclosed-volume weights, shape (nR, nZ, n).

        The surface weight is the derivative of the enclosed indicator, i.e. a
        smoothed delta on the surface; the enclosed weight is the indicator
        itself. Deriving one from the other in closed form -- rather than
        calling ``jax.grad`` on the integral -- keeps this a single pass, and
        makes ``dV_dpsi`` the exact derivative of ``volume`` by construction.
        """
        enclosed, shape = _kernel((label[..., None] - psi_levels) / width, order)
        surface = shape / width
        if mask is not None:
            m = mask[..., None]
            enclosed, surface = enclosed * m, surface * m
        return surface, enclosed

    def _width(label, psi_levels, psi_axis, psi_edge, mask, dpsi_cell):
        """Fixed point on ``width = width_cells * <dpsi_cell>_surface``.

        Differentiated through, deliberately. Freezing the width with
        ``stop_gradient`` is tempting -- it is a quadrature parameter, not
        physics -- but it makes the returned gradient stop being the derivative
        of the returned value: measured 6e-4 relative, small but structural, and
        an inconsistent Jacobian is exactly what this package refuses elsewhere.
        Three unrolled passes cost three kernel evaluations forward and back.

        ``MIN_WIDTH_FRACTION`` keeps the edge cap from driving the width to zero
        for a level sitting *on* ``psi_edge``, where the true contour integral
        diverges and the unfloored expression returns NaN. The floor is far
        below where the cap does useful work, so it changes nothing elsewhere.
        """
        span = jnp.abs(psi_axis - psi_edge)
        width = jnp.full(jnp.shape(psi_levels), 0.01 * span)
        for _ in range(n_passes):
            surface, _ = _weights(label, psi_levels, width, mask)
            w = surface * two_pi_R_dA[..., None]
            cell = width_cells * jnp.sum(
                w * dpsi_cell[..., None], axis=(0, 1)
            ) / jnp.sum(w, axis=(0, 1))
            width = cell
            if max_span is not None:
                capped = max_span * jnp.abs(psi_levels - psi_edge)
                width = jnp.maximum(
                    jnp.minimum(cell, capped), MIN_WIDTH_FRACTION * cell
                )
        return width

    def _prepare(psi, psi_levels, psi_axis, psi_edge, mask, label):
        lab = psi if label is None else label
        gR, gZ = grad_psi(psi)
        dpsi_cell = jnp.sqrt((gR * dR) ** 2 + (gZ * dZ) ** 2)
        width = _width(lab, psi_levels, psi_axis, psi_edge, mask, dpsi_cell)
        surface, enclosed = _weights(lab, psi_levels, width, mask)
        return gR, gZ, dpsi_cell, width, surface, enclosed

    def average(psi, X, psi_levels, psi_axis, psi_edge, mask=None, label=None):
        """Flux-surface average of ``X`` on each level.

        Weighted by ``2 pi R dl / |grad psi|``, the standard volume-weighted
        convention (Wesson), so ``<1> == 1`` identically.
        """
        _, _, _, _, surface, _ = _prepare(
            psi, psi_levels, psi_axis, psi_edge, mask, label
        )
        w = surface * two_pi_R_dA[..., None]
        return jnp.sum(w * X[..., None], axis=(0, 1)) / jnp.sum(w, axis=(0, 1))

    def surfaces(psi, psi_levels, psi_axis, psi_edge, mask=None, label=None):
        gR, gZ, dpsi_cell, width, surface, enclosed = _prepare(
            psi, psi_levels, psi_axis, psi_edge, mask, label
        )
        w = surface * two_pi_R_dA[..., None]
        norm_ = jnp.sum(w, axis=(0, 1))

        def avg(X):
            return jnp.sum(w * X[..., None], axis=(0, 1)) / norm_

        grad2 = gR**2 + gZ**2
        grad = jnp.sqrt(jnp.maximum(grad2, 1e-300))

        # dV/dpsi is the same integral as the averaging norm, and
        # int dl/B_p = int R dl/|grad psi| = (dV/dpsi) / 2 pi.
        return FluxSurfaces(
            psi_levels=psi_levels,
            volume=jnp.sum(enclosed * two_pi_R_dA[..., None], axis=(0, 1)),
            area=jnp.sum(enclosed * dA, axis=(0, 1)),
            dV_dpsi=norm_,
            int_dl_over_Bp=norm_ / (2.0 * jnp.pi),
            avg_1_over_R=avg(1.0 / R),
            avg_1_over_R2=avg(1.0 / R**2),
            avg_grad_psi=avg(grad),
            avg_grad_psi2=avg(grad2),
            avg_grad_psi2_over_R2=avg(grad2 / R**2),
            width=width,
            n_cells=width / avg(dpsi_cell),
            n_eff=norm_**2 / jnp.sum(w**2, axis=(0, 1)),
        )

    return average, surfaces, grad_psi


def safety_factor(fs: FluxSurfaces, F):
    """q = (F / 2 pi) * contour_int dl / (R^2 B_p), from the bundle.

    Useful as an independent check: FreeGS4E computes ``q`` by tracing flux
    surfaces (``freegs4e/equilibrium.py:755``), so agreement tests the whole
    co-area construction against a contour-based implementation.
    """
    return F / (2.0 * jnp.pi) * fs.avg_1_over_R2 * fs.int_dl_over_Bp
