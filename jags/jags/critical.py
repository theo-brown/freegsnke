"""Smooth, differentiable replacement for FreeGSNKE's critical-point machinery.

FreeGSNKE determines the last closed flux surface with contour tracing,
``matplotlib.path.Path.contains_points`` and a bisection loop on the X-point
flux level (``freegsnke/jtor_update.py:136-269``). None of that is traceable by
JAX, and none of it is differentiable, so an exact Newton method cannot be built
on top of it.

This module reformulates the same three quantities -- magnetic axis, boundary
flux, core mask -- as smooth functions of psi:

* the axis is a stationary point of psi, located by grid search and inner
  Newton, then differentiated by the implicit function theorem;
* the boundary flux is a soft maximum over limiter contact points and X-point
  fluxes, which makes the limited/diverted switch smooth rather than a branch
  (FreeGSNKE takes a hard ``max``, see ``limiter_func.py:490-494``);
* the core mask is a sigmoid in psi rather than a point-in-polygon test.

These are diagnostics only. The solver does not use them -- see ``solver.py``
for why the residual needs no critical point at all.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .grid import Grid
from .interp import make_interpolator


class Critical(NamedTuple):
    """Quantities describing the plasma core, all differentiable in psi."""

    axis: jnp.ndarray  # (2,) magnetic axis (R, Z)
    psi_axis: jnp.ndarray  # flux on axis
    psi_bndry: jnp.ndarray  # flux on the last closed flux surface
    mask: jnp.ndarray  # (nR, nZ) smooth core mask in [0, 1]
    xpoints: jnp.ndarray  # (2, 2) refined saddle candidates (R, Z)
    xpoint_valid: jnp.ndarray  # (2,) bool, whether each candidate is a real saddle


def soft_max(values, valid, beta):
    """Smooth (log-sum-exp) maximum over ``values`` where ``valid``.

    ``beta`` has units of 1/psi. The result over-estimates the true maximum by
    at most ``log(n_valid) / beta``, so ``beta`` wants to be large.
    """
    neg_inf = jnp.array(-jnp.inf, dtype=values.dtype)
    shift = jax.lax.stop_gradient(jnp.max(jnp.where(valid, values, neg_inf)))
    weights = jnp.where(valid, jnp.exp(beta * (values - shift)), 0.0)
    return shift + jnp.log(jnp.sum(weights)) / beta


def make_axis_finder(grid: Grid, n_refine: int = 8):
    """Just the magnetic-axis part of ``make_critical_fns``.

    Split out because ``reach.py`` needs a differentiable axis inside the
    residual and must not pay for X-point search or limiter interpolation.
    Returns ``(find_axis, stationary_point, (value, gradient, hessian))``.
    """
    value, gradient, hessian = make_interpolator(grid)

    Rmin, Zmin = float(grid.Rmin), float(grid.Zmin)
    dR, dZ = float(grid.dR), float(grid.dZ)
    shape = (int(grid.nR), int(grid.nZ))

    interior = np.zeros(shape, dtype=bool)
    interior[2:-2, 2:-2] = True
    interior = jnp.asarray(interior & grid.limiter_mask())

    def stationary_point(psi, x_seed):
        """Refine a stationary point of psi and differentiate it by the IFT.

        The inner Newton iteration runs on a detached psi, so no gradient flows
        through the iteration itself. A single further Newton step, taken from
        the detached converged point but with live psi, reproduces exactly the
        implicit-function-theorem derivative

            dx*/dpsi = -H^-1 d(grad psi)/dpsi

        because the neglected term is proportional to grad psi, which vanishes
        at the solution.
        """
        psi_detached = jax.lax.stop_gradient(psi)

        def step(x, _):
            g = gradient(psi_detached, x)
            H = hessian(psi_detached, x) + 1e-12 * jnp.eye(2)
            return x - jnp.linalg.solve(H, g), None

        x_conv, _ = jax.lax.scan(step, x_seed, None, length=n_refine)
        x_conv = jax.lax.stop_gradient(x_conv)

        g = gradient(psi, x_conv)
        H = hessian(psi, x_conv)
        return x_conv - jnp.linalg.solve(H + 1e-12 * jnp.eye(2), g)

    def find_axis(psi):
        """Magnetic axis: the maximum of psi inside the limiter."""
        k = jnp.argmax(jnp.where(interior, psi, -jnp.inf))
        i, j = jnp.unravel_index(k, shape)
        seed = jnp.stack([Rmin + i * dR, Zmin + j * dZ])
        axis = stationary_point(psi, seed)
        return axis, value(psi, axis)

    return find_axis, stationary_point, (value, gradient, hessian)


def make_critical_fns(grid: Grid, n_refine: int = 8):
    """Build the differentiable critical-point diagnostics for a given grid.

    Returns a single function ``critical(psi, ...) -> Critical``.
    """
    find_axis, stationary_point, (value, _, hessian) = make_axis_finder(
        grid, n_refine
    )

    Rmin, Zmin = float(grid.Rmin), float(grid.Zmin)
    dR, dZ = float(grid.dR), float(grid.dZ)
    shape = (int(grid.nR), int(grid.nZ))

    mask_limiter = jnp.asarray(grid.limiter_mask())
    limiter_pts = jnp.asarray(grid.limiter_edge_points())

    # Grid points at least two cells inside the domain, where the bicubic
    # stencil is well defined and stationary points can safely be sought.
    interior = np.zeros(shape, dtype=bool)
    interior[2:-2, 2:-2] = True
    interior = jnp.asarray(interior & grid.limiter_mask())

    def index_to_rz(k):
        i, j = jnp.unravel_index(k, shape)
        return jnp.stack([Rmin + i * dR, Zmin + j * dZ])

    def find_xpoints(psi, axis):
        """Two saddle candidates, seeded above and below the magnetic axis.

        Seeds are grid minima of |grad psi|^2 in the half-plane above/below the
        axis; each is refined by the same Newton iteration, which converges to a
        stationary point of either type. Validity (a genuine saddle, inside the
        limiter, below the axis flux) is a discrete property, so it is evaluated
        under stop_gradient and used only to gate the soft maximum.
        """
        # |grad psi|^2 by central differences, for seeding only.
        gR = jnp.zeros_like(psi).at[1:-1, :].set(
            (psi[2:, :] - psi[:-2, :]) / (2 * dR)
        )
        gZ = jnp.zeros_like(psi).at[:, 1:-1].set(
            (psi[:, 2:] - psi[:, :-2]) / (2 * dZ)
        )
        gmag = gR**2 + gZ**2

        Zgrid = jnp.asarray(grid.Z)
        margin = 3 * dZ

        def seed_in(half):
            ok = interior & half
            return index_to_rz(jnp.argmin(jnp.where(ok, gmag, jnp.inf)))

        seeds = jnp.stack(
            [
                seed_in(Zgrid > axis[1] + margin),
                seed_in(Zgrid < axis[1] - margin),
            ]
        )
        xpts = jax.vmap(lambda s: stationary_point(psi, s))(seeds)

        def is_saddle(x):
            H = jax.lax.stop_gradient(hessian(psi, x))
            det_neg = jnp.linalg.det(H) < 0.0
            in_domain = (
                (x[0] > grid.Rmin + 2 * dR)
                & (x[0] < grid.Rmax - 2 * dR)
                & (x[1] > grid.Zmin + 2 * dZ)
                & (x[1] < grid.Zmax - 2 * dZ)
            )
            return det_neg & in_domain

        valid = jax.vmap(is_saddle)(xpts)
        return xpts, jax.lax.stop_gradient(valid)

    def boundary_flux(psi, axis, psi_axis, beta, use_xpoints=True, use_limiter=True):
        """Soft maximum of the limiter contact flux and the X-point fluxes.

        FreeGSNKE takes ``psi_bndry = max(psi_xpt, max(psi on limiter))`` and
        flags the configuration limited when the limiter wins
        (``limiter_func.py:490-494``). Replacing that hard max with a soft max
        makes the switch differentiable.

        One caveat, and it is the connectivity limitation in miniature.
        FreeGSNKE maximises only over limiter cells *adjacent to the plasma
        core* (``limiter_func.py:465-480``); this maximises over the whole
        contour. In a diverted configuration a point far down a divertor leg can
        carry a flux above the X-point value while being nowhere near the core,
        and no threshold on psi can tell the two apart -- that distinction is
        topological. On the MAST-U reference case this inflates psi_bndry by
        about 0.6% of the core flux depth, even though the X-point flux itself
        agrees with FreeGSNKE to 2e-6. Pass ``use_limiter=False`` for a
        configuration known to be diverted.
        """
        psi_lim = jax.vmap(lambda p: value(psi, p))(limiter_pts)
        cand = psi_lim
        valid = jnp.full(psi_lim.shape, bool(use_limiter))

        xpts, xvalid = find_xpoints(psi, axis)
        if use_xpoints:
            psi_x = jax.vmap(lambda p: value(psi, p))(xpts)
            # An X-point above the axis flux is spurious.
            xvalid = xvalid & (jax.lax.stop_gradient(psi_x) < psi_axis)
            cand = jnp.concatenate([cand, psi_x])
            valid = jnp.concatenate([valid, xvalid])

        return soft_max(cand, valid, beta), xpts, xvalid

    def critical(
        psi, beta_norm=2000.0, eps=0.02, use_xpoints=True, use_limiter=True
    ):
        """Full critical-point solve.

        Parameters
        ----------
        beta_norm : dimensionless sharpness of the soft maximum, in units of the
            core flux depth. Larger is sharper; the soft maximum overestimates
            the true maximum by at most ``log(n_candidates) / beta``, so this
            should be large enough that the bias is below the accuracy wanted.
        eps : width of the core-mask sigmoid, as a fraction of the core flux
            depth.
        use_xpoints, use_limiter : which candidates may set the boundary flux.
        """
        axis, psi_axis = find_axis(psi)

        # A detached flux scale keeps beta and eps dimensionless without
        # feeding the smoothing widths back into the derivative.
        scale = jax.lax.stop_gradient(
            jnp.maximum(psi_axis - jnp.min(psi), 1e-12)
        )
        psi_bndry, xpts, xvalid = boundary_flux(
            psi, axis, psi_axis, beta_norm / scale, use_xpoints, use_limiter
        )

        mask = jax.nn.sigmoid((psi - psi_bndry) / (eps * scale)) * mask_limiter
        return Critical(axis, psi_axis, psi_bndry, mask, xpts, xvalid)

    return critical
