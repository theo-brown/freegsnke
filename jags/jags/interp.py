"""Bicubic interpolation of psi on the grid.

Newton iteration on ``grad(psi) = 0`` needs a continuously differentiable
interpolant. ``jax.scipy.ndimage.map_coordinates(order=1)`` is bilinear and so
only C0 -- its gradient jumps across cell boundaries and the inner Newton will
not converge. Catmull-Rom bicubic is C1, with analytic gradient and Hessian
obtained for free from ``jax.grad`` / ``jax.hessian``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .grid import Grid


def _cubic_weights(t):
    """Catmull-Rom basis weights for samples at offsets -1, 0, 1, 2."""
    t2 = t * t
    t3 = t2 * t
    return jnp.stack(
        [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ]
    )


def make_interpolator(grid: Grid):
    """Build (value, gradient, hessian) evaluators closed over the grid geometry.

    Each returned function takes ``(psi, x)`` where ``psi`` is (nR, nZ) and
    ``x`` is ``(R, Z)``.
    """
    Rmin, Zmin = float(grid.Rmin), float(grid.Zmin)
    dR, dZ = float(grid.dR), float(grid.dZ)
    nR, nZ = int(grid.nR), int(grid.nZ)

    def value(psi, x):
        fi = (x[0] - Rmin) / dR
        fj = (x[1] - Zmin) / dZ

        # The stencil origin is a piecewise-constant function of position, so it
        # carries no derivative information; stop_gradient makes that explicit.
        # Catmull-Rom is C1 across cells, so the interpolant is still smooth.
        i0 = jnp.clip(jnp.floor(fi) - 1, 0, nR - 4)
        j0 = jnp.clip(jnp.floor(fj) - 1, 0, nZ - 4)
        i0 = jax.lax.stop_gradient(i0).astype(jnp.int32)
        j0 = jax.lax.stop_gradient(j0).astype(jnp.int32)

        patch = jax.lax.dynamic_slice(psi, (i0, j0), (4, 4))

        # Local coordinates relative to the second stencil point.
        wi = _cubic_weights(fi - (i0 + 1))
        wj = _cubic_weights(fj - (j0 + 1))
        return wi @ patch @ wj

    gradient = jax.grad(value, argnums=1)
    hessian = jax.jacfwd(gradient, argnums=1)

    return value, gradient, hessian
