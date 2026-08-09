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

Accuracy, measured against circular surfaces where every quantity is known in
closed form (``test_fsa.py``), at eps = 0.01:

* **Ratios** -- every ``<X>`` -- benefit from the kernel normalisation
  cancelling. Outside r/a ~ 0.3, ``<1/R^2>`` and ``<|grad psi|^2>`` reach 1e-4
  or better. ``<|grad psi|>`` is the slow one, ~5e-3 at r/a = 0.3 falling to
  1e-4 by the edge: the measure grows like r across the band, biasing its
  centroid outward, and for a quantity linear in r that survives at
  O(eps^2 / r^2). Squared and inverse quantities are far less affected.
* **Absolute integrals** -- ``int_dl_over_Bp``, ``dV_dpsi`` -- carry ~1% at
  practical resolutions. The kernel does not sum to exactly one over a band
  only a fraction of a cell wide, and differentiating the enclosed volume
  amplifies its ~5e-4 error. There is no free lunch here: ``dV_dpsi`` *is* the
  derivative of ``volume``, so the two errors are the same error.

Two failure modes bound the useful range, and they pull in opposite directions:

* **Near the axis** the surfaces are simply not resolved -- at r = 0.03 in a
  domain of minor radius 0.5 a surface is ~3 cells across, and the error is the
  same at 65x65 and 129x129, so it is not fixed by refining. Contour tracing
  fares no better; nothing can average over a curve the grid cannot represent.
  Expect the innermost ~20% in minor radius to be unusable and extrapolate.
* **On outer surfaces of a coarse grid** the opposite happens: the band spans a
  fraction of a cell and the quadrature aliases. Widening ``eps`` fixes that but
  smears neighbouring surfaces, an O(eps) bias. eps ~ 0.01 is the compromise.

Making the width track ``|grad psi|`` -- a fixed number of *cells* everywhere --
does fix the aliasing (``int_dl_over_Bp`` error 9e-2 -> 9e-6 at 65x65) and the
co-area identity survives a slowly varying width. It is not used because the
same width also has to serve the enclosed indicator, where a spatially varying
smoothing is not a consistent volume, and because it divides by ``|grad psi|``,
which vanishes at the axis.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .grid import Grid


class FluxSurfaces(NamedTuple):
    """Flux-surface geometry on a set of flux labels.

    Names follow TORAX's ``StandardGeometryIntermediates`` where they
    correspond, so the mapping across is direct.
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


def make_flux_surface_averager(grid: Grid, eps: float = 0.01):
    """Build flux-surface-average machinery for a grid.

    Parameters
    ----------
    eps : width of the delta kernel, as a fraction of the flux range passed to
        the returned functions. Smaller resolves neighbouring surfaces better
        but samples fewer grid cells; see the module docstring.

    Returns ``(average, surfaces, grad_psi)``:

    ``average(psi, X, psi_levels, psi_scale, mask=None)``
        <X> on each level, for an arbitrary field X on the grid.
    ``surfaces(psi, psi_levels, psi_scale, mask=None)``
        the ``FluxSurfaces`` bundle.
    ``grad_psi(psi)``
        ``(dpsi/dR, dpsi/dZ)``, exposed because callers usually need B_p too.
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

    def _weights(psi, psi_levels, psi_scale, mask):
        """Surface weights w_i and enclosed-volume weights, shape (nR, nZ, n).

        The surface weight is the derivative of the enclosed indicator, i.e. a
        smoothed delta on the surface; the enclosed weight is the indicator
        itself. Deriving one from the other by hand -- rather than calling
        ``jax.grad`` on the integral -- keeps this a single pass, and the two
        are consistent by construction: ``d(enclosed)/d(psi0) = -surface``.
        """
        width = eps * psi_scale
        u = (psi[..., None] - psi_levels) / width
        enclosed = jax.nn.sigmoid(u)
        surface = enclosed * (1.0 - enclosed) / width  # d(sigmoid)/d(psi)
        if mask is not None:
            m = mask[..., None]
            enclosed, surface = enclosed * m, surface * m
        return surface, enclosed

    def average(psi, X, psi_levels, psi_scale, mask=None):
        """Flux-surface average of ``X`` on each level.

        Weighted by ``2 pi R dl / |grad psi|``, the standard volume-weighted
        convention (Wesson), so ``<1> == 1`` identically.
        """
        surface, _ = _weights(psi, psi_levels, psi_scale, mask)
        w = surface * two_pi_R_dA[..., None]
        return jnp.sum(w * X[..., None], axis=(0, 1)) / jnp.sum(w, axis=(0, 1))

    def surfaces(psi, psi_levels, psi_scale, mask=None):
        surface, enclosed = _weights(psi, psi_levels, psi_scale, mask)
        w = surface * two_pi_R_dA[..., None]
        norm = jnp.sum(w, axis=(0, 1))

        def avg(X):
            return jnp.sum(w * X[..., None], axis=(0, 1)) / norm

        gR, gZ = grad_psi(psi)
        grad2 = gR**2 + gZ**2
        grad = jnp.sqrt(jnp.maximum(grad2, 1e-300))

        # dV/dpsi is the same integral as the averaging norm, and
        # int dl/B_p = int R dl/|grad psi| = (dV/dpsi) / 2 pi.
        dV_dpsi = norm
        return FluxSurfaces(
            psi_levels=psi_levels,
            volume=jnp.sum(enclosed * two_pi_R_dA[..., None], axis=(0, 1)),
            area=jnp.sum(enclosed * dA, axis=(0, 1)),
            dV_dpsi=dV_dpsi,
            int_dl_over_Bp=dV_dpsi / (2.0 * jnp.pi),
            avg_1_over_R=avg(1.0 / R),
            avg_1_over_R2=avg(1.0 / R**2),
            avg_grad_psi=avg(grad),
            avg_grad_psi2=avg(grad2),
            avg_grad_psi2_over_R2=avg(grad2 / R**2),
        )

    return average, surfaces, grad_psi


def safety_factor(fs: FluxSurfaces, F):
    """q = (F / 2 pi) * contour_int dl / (R^2 B_p), from the bundle.

    Useful as an independent check: FreeGS4E computes ``q`` by tracing flux
    surfaces (``freegs4e/equilibrium.py:755``), so agreement tests the whole
    co-area construction against a contour-based implementation.
    """
    return F / (2.0 * jnp.pi) * fs.avg_1_over_R2 * fs.int_dl_over_Bp
