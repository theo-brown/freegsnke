"""Exact Newton steps from the Jacobian's action, with no assembly.

Both routines here take only a residual function, so this module has no
intra-package dependencies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def dense_jacobian(residual_fn, psi, chunk=32):
    """Exact dense Jacobian, in column blocks to bound memory.

    ``jax.jacfwd`` vmaps over all N tangents at once. That is fine for a
    pointwise residual, but with ``reach.make_reachability`` every tangent
    carries an (N, n_samples) ray tensor and the batch needs 45 GB at 65x65.
    Chunking trades that for a loop.

    Each JVP against a basis vector gives one *column* of the Jacobian. This is
    the assumption-free reference the tests check ``make_matrix_free_step``
    against; it is too slow to use for real solves.
    """
    N = psi.size

    @jax.jit
    def block(idx):
        E = jax.nn.one_hot(idx, N, dtype=psi.dtype)
        return jax.vmap(lambda v: jax.jvp(residual_fn, (psi,), (v,))[1])(E).T

    return jnp.concatenate(
        [block(jnp.arange(s, min(s + chunk, N))) for s in range(0, N, chunk)],
        axis=1,
    )


def make_matrix_free_step(residual_fn, tol=1e-12, restart=40, maxiter=8):
    """Newton step by GMRES on the Jacobian's action -- nothing assembled.

    ``jax.jvp`` supplies ``J v`` exactly, so this is a true Newton step rather
    than a quasi-Newton one, and the (N, N) matrix is never formed. Both matter
    once ``reach.make_reachability`` is in play: it makes the map non-pointwise,
    and the dense assembly runs out of memory.

    GMRES suits this problem unusually well. Writing the residual as
    ``F = psi - A^-1 b`` makes ``J = I - A^-1 db/dpsi`` a compact perturbation of
    the identity: measured cond(J) = 3.5, with only ~7 eigenvalues further than
    0.1 from 1, both independent of grid size. The Krylov space needed is small
    and fixed, and no preconditioner is required. This is the structure
    FreeGSNKE exploits with its 16-direction Arnoldi basis; the difference is
    that the Jacobian action here is exact rather than finite differenced.
    """

    @jax.jit
    def step(psi, F):
        x, _ = jax.scipy.sparse.linalg.gmres(
            lambda v: jax.jvp(residual_fn, (psi,), (v,))[1],
            -F,
            tol=tol,
            atol=0.0,
            restart=restart,
            maxiter=maxiter,
            solve_method="batched",
        )
        return x

    return step
