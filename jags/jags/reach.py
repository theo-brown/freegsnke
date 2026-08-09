"""Reachability from the magnetic axis, as a smooth replacement for psi.

The pointwise formulation puts plasma current wherever ``psi > psi_edge`` inside
the vessel. For a limited plasma that is exactly right, because the boundary is
the wall. For a diverted plasma it is wrong: the level set also contains lobes
lying beyond a null, which are at high flux but are not part of the core, and no
fixed geometric mask can exclude them because the null's position depends on the
solution.

The fix here does not look for nulls at all. Define

    m(x) = min over t in [0, 1] of psi(axis + t (x - axis))

the running minimum of psi along the straight ray from the magnetic axis, and
give that to the profiles in place of psi.

* **Inside the core** psi decreases monotonically outward along any ray from the
  axis, so the minimum is attained at the endpoint and ``m == psi`` exactly. The
  core is untouched.
* **In a lobe** the ray must dip below ``psi_edge`` to get there, so
  ``m < psi_edge`` and the profile's existing compact support gives zero current.
  No extra mask and no extra smoothing parameter are introduced.

The only assumption is that the core is **star-shaped about the magnetic axis**.
That is much weaker than any statement about the number or position of X-points:
it holds for single and double null, and for Super-X, snowflake and X-divertor
geometries, all of which shape the divertor legs rather than the core. It is a
real assumption nonetheless, and a strongly indented boundary would violate it.

Cost is one to two residual evaluations. The structural price is that ``Jtor`` at
a point now depends on psi along a whole ray, so ``dJtor/dpsi`` is no longer
diagonal -- measured 3-5% off. Any Jacobian that assumes a pointwise map is
therefore only approximate here, which is why the solver takes its Newton step
from the Jacobian's exact action instead (``jacobian.make_matrix_free_step``).
The degeneracy is worst at the magnetic axis, where the ray collapses to a point
and every sample carries the same flux, so no amount of sharpening in ``beta``
removes it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .critical import make_axis_finder
from .grid import Grid


def make_reachability(grid: Grid, n_samples: int = 32, beta_norm: float = 2e5):
    """Build ``psi -> m``, both shaped (nR, nZ).

    Parameters
    ----------
    n_samples : points sampled along each ray. Controls how finely a thin
        barrier can be detected; the count of spuriously included cells falls
        as this rises.
    beta_norm : sharpness of the soft minimum, in units of the flux scale. The
        soft minimum underestimates the true minimum by about
        ``log(n_samples) / beta``, which biases m slightly low *inside* the core
        where it should be exactly psi, so this wants to be large. The bias
        scales as 1/beta: at 2e3 it is 2.9e-4 of the flux range, at 2e5 it is
        2.9e-6.
    """
    find_axis, _, (value, _, _) = make_axis_finder(grid)
    shape = (int(grid.nR), int(grid.nZ))

    # Endpoint of every ray: one per grid point.
    targets = jnp.stack(
        [jnp.asarray(grid.R).ravel(), jnp.asarray(grid.Z).ravel()], axis=-1
    )
    # t = 0 is the axis itself and carries no information, so start past it.
    ts = jnp.linspace(1.0 / n_samples, 1.0, n_samples)

    def reachability(psi):
        axis, psi_axis = find_axis(psi)

        # A detached flux scale keeps beta dimensionless without feeding the
        # smoothing width back into the derivative.
        scale = jax.lax.stop_gradient(
            jnp.maximum(psi_axis - jnp.min(psi), 1e-12)
        )
        beta = beta_norm / scale

        pts = axis + ts[None, :, None] * (targets - axis)[:, None, :]
        vals = jax.vmap(jax.vmap(lambda p: value(psi, p)))(pts)  # (N, n_samples)

        shift = jax.lax.stop_gradient(jnp.min(vals, axis=1))
        soft_min = shift - jnp.log(
            jnp.sum(jnp.exp(-beta * (vals - shift[:, None])), axis=1)
        ) / beta
        return soft_min.reshape(shape)

    return reachability
