"""Grid operators and Green's functions.

Everything in this module is geometry-only: none of it depends on psi, so it is
all precomputed once in NumPy/SciPy and frozen into constant JAX arrays. In
particular JAX never needs elliptic integrals.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import ellipe, ellipk

from .grid import Grid

MU0 = 4e-7 * np.pi

# Fourth-order finite-difference weights, copied from
# freegs4e/gradshafranov.py:240-275 so the discretisation matches exactly.
_CENTRED_1ST = [(-2, 1.0 / 12), (-1, -8.0 / 12), (1, 8.0 / 12), (2, -1.0 / 12)]
_OFFSET_1ST = [
    (-1, -3.0 / 12),
    (0, -10.0 / 12),
    (1, 18.0 / 12),
    (2, -6.0 / 12),
    (3, 1.0 / 12),
]
_CENTRED_2ND = [
    (-2, -1.0 / 12),
    (-1, 16.0 / 12),
    (0, -30.0 / 12),
    (1, 16.0 / 12),
    (2, -1.0 / 12),
]
_OFFSET_2ND = [
    (-1, 10.0 / 12),
    (0, -15.0 / 12),
    (1, -4.0 / 12),
    (2, 14.0 / 12),
    (3, -6.0 / 12),
    (4, 1.0 / 12),
]


def delstar_matrix(grid: Grid) -> sp.csr_matrix:
    """Fourth-order Grad-Shafranov elliptic operator with identity boundary rows.

    Delstar = d2/dR2 + d2/dZ2 - (1/R) d/dR

    Rows corresponding to boundary points are set to the identity, so the linear
    system ``A psi = b`` imposes ``psi = b`` there. That lets the free-boundary
    condition be applied by writing the Green's-function values into ``b``.

    This mirrors ``freegs4e.gradshafranov.GSsparse4thOrder``.
    """
    nR, nZ = grid.nR, grid.nZ
    dR, dZ = grid.dR, grid.dZ
    inv_dR2, inv_dZ2 = 1.0 / dR**2, 1.0 / dZ**2

    A = sp.lil_matrix((grid.N, grid.N))

    for i in range(1, nR - 1):
        R = grid.Rmin + dR * i
        for j in range(1, nZ - 1):
            row = i * nZ + j

            # d2/dZ2, one-sided next to the boundary
            if j == 1:
                for off, w in _OFFSET_2ND:
                    A[row, row + off] += w * inv_dZ2
            elif j == nZ - 2:
                for off, w in _OFFSET_2ND:
                    A[row, row - off] += w * inv_dZ2
            else:
                for off, w in _CENTRED_2ND:
                    A[row, row + off] += w * inv_dZ2

            # d2/dR2 - (1/R) d/dR
            if i == 1:
                for off, w in _OFFSET_2ND:
                    A[row, row + off * nZ] += w * inv_dR2
                for off, w in _OFFSET_1ST:
                    A[row, row + off * nZ] -= w / (R * dR)
            elif i == nR - 2:
                for off, w in _OFFSET_2ND:
                    A[row, row - off * nZ] += w * inv_dR2
                for off, w in _OFFSET_1ST:
                    A[row, row - off * nZ] += w / (R * dR)
            else:
                for off, w in _CENTRED_2ND:
                    A[row, row + off * nZ] += w * inv_dR2
                for off, w in _CENTRED_1ST:
                    A[row, row + off * nZ] -= w / (R * dR)

    for i in range(nR):
        for j in (0, nZ - 1):
            A[i * nZ + j, i * nZ + j] = 1.0
    for i in (0, nR - 1):
        for j in range(nZ):
            A[i * nZ + j, i * nZ + j] = 1.0

    return A.tocsr()


def inverse_delstar(grid: Grid) -> np.ndarray:
    """Dense inverse of the Delstar matrix, (N, N).

    The operator does not depend on psi, so this is computed once. Writing the
    residual as ``psi - A^-1 b(psi)`` (as FreeGSNKE does) makes the Newton
    Jacobian ``I - A^-1 db/dpsi``: a compact perturbation of the identity, which
    is far better conditioned than the raw operator form.
    """
    lu = spla.splu(delstar_matrix(grid).tocsc())
    return lu.solve(np.eye(grid.N))


def greens(Rc, Zc, R, Z):
    """Poloidal flux at (R, Z) from a unit-current filament at (Rc, Zc).

    Identical to ``freegs4e.gradshafranov.Greens``.
    """
    k2 = 4.0 * R * Rc / ((R + Rc) ** 2 + (Z - Zc) ** 2)
    k2 = np.clip(k2, 1e-10, 1.0 - 1e-10)
    k = np.sqrt(k2)
    return (
        (MU0 / (2.0 * np.pi))
        * np.sqrt(R * Rc)
        * ((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2))
        / k
    )


def boundary_greens_matrix(grid: Grid) -> np.ndarray:
    """(n_bnd, N) matrix mapping Jtor on the grid to psi on the domain edge.

    Includes the dR*dZ area element, so ``G_bnd @ jtor.ravel()`` is the plasma
    contribution to psi at each boundary point. The self-interaction term is
    zeroed to avoid the log singularity of the Green's function, matching
    ``freegsnke/GSstaticsolver.py:133-138``.
    """
    bnd = grid.boundary_indices
    R, Z = grid.R, grid.Z
    Rb = grid.R_1d[bnd[:, 0]]
    Zb = grid.Z_1d[bnd[:, 1]]

    G = greens(R[None, :, :], Z[None, :, :], Rb[:, None, None], Zb[:, None, None])
    G[np.arange(len(bnd)), bnd[:, 0], bnd[:, 1]] = 0.0
    return (G * grid.dA).reshape(len(bnd), grid.N)


def coil_greens_matrix(grid: Grid, coil_RZ: np.ndarray) -> np.ndarray:
    """(N, n_coil) matrix mapping coil currents to psi on the grid.

    ``coil_RZ`` is (n_coil, 2). Because psi_coil is linear in the currents and
    this matrix is a constant, gradients with respect to coil current flow
    through it without JAX ever needing elliptic integrals.
    """
    coil_RZ = np.atleast_2d(np.asarray(coil_RZ, dtype=float))
    R, Z = grid.R, grid.Z
    G = greens(
        coil_RZ[:, 0][:, None, None],
        coil_RZ[:, 1][:, None, None],
        R[None, :, :],
        Z[None, :, :],
    )
    return G.reshape(len(coil_RZ), grid.N).T
