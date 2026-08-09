"""Rectangular R-Z grid and limiter geometry.

Index convention matches FreeGS/FreeGS4E: arrays have shape ``(nR, nZ)`` with R
varying along axis 0 and Z along axis 1, i.e. ``R[i, j]``, ``Z[i, j]``.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class Grid:
    """A rectangular R-Z grid plus the limiter contour bounding the plasma."""

    Rmin: float
    Rmax: float
    Zmin: float
    Zmax: float
    nR: int
    nZ: int
    limiter: np.ndarray  # (n_points, 2) closed-ish polygon of (R, Z) vertices

    @property
    def dR(self) -> float:
        return (self.Rmax - self.Rmin) / (self.nR - 1)

    @property
    def dZ(self) -> float:
        return (self.Zmax - self.Zmin) / (self.nZ - 1)

    @property
    def R_1d(self) -> np.ndarray:
        return np.linspace(self.Rmin, self.Rmax, self.nR)

    @property
    def Z_1d(self) -> np.ndarray:
        return np.linspace(self.Zmin, self.Zmax, self.nZ)

    @property
    def R(self) -> np.ndarray:
        return np.broadcast_to(self.R_1d[:, None], (self.nR, self.nZ)).copy()

    @property
    def Z(self) -> np.ndarray:
        return np.broadcast_to(self.Z_1d[None, :], (self.nR, self.nZ)).copy()

    @property
    def N(self) -> int:
        return self.nR * self.nZ

    @property
    def dA(self) -> float:
        return self.dR * self.dZ

    @property
    def boundary_indices(self) -> np.ndarray:
        """(n_bnd, 2) array of (i, j) grid indices on the domain edge.

        Ordering matches ``freegsnke/GSstaticsolver.py:116-123`` so that boundary
        vectors can be compared against FreeGSNKE directly.
        """
        nR, nZ = self.nR, self.nZ
        idx = (
            [(i, 0) for i in range(nR)]
            + [(i, nZ - 1) for i in range(nR)]
            + [(0, j) for j in range(1, nZ - 1)]
            + [(nR - 1, j) for j in range(1, nZ - 1)]
        )
        return np.array(idx, dtype=int)

    def limiter_mask(self) -> np.ndarray:
        """Boolean (nR, nZ) mask, True for grid points inside the limiter."""
        pts = np.stack([self.R.ravel(), self.Z.ravel()], axis=-1)
        return _points_in_polygon(pts, self.limiter).reshape(self.nR, self.nZ)

    def limiter_edge_points(self, n_per_segment: int = 4) -> np.ndarray:
        """Points sampled along the limiter contour, where the plasma may touch.

        These are the candidate contact points for a limited plasma: FreeGSNKE
        interpolates psi onto the limiter polygon rather than onto grid points
        (``limiter_func.core_mask_limiter``), and we do the same so that the
        boundary flux does not jump as the plasma moves between cells.
        """
        verts = np.asarray(self.limiter, dtype=float)
        if not np.allclose(verts[0], verts[-1]):
            verts = np.concatenate([verts, verts[:1]], axis=0)
        segs = []
        for a, b in zip(verts[:-1], verts[1:]):
            t = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)[:, None]
            segs.append(a[None, :] * (1 - t) + b[None, :] * t)
        return np.concatenate(segs, axis=0)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting test.

    Parameters
    ----------
    points : (n, 2) array of query points.
    polygon : (m, 2) array of polygon vertices (closed automatically).
    """
    poly = np.asarray(polygon, dtype=float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.concatenate([poly, poly[:1]], axis=0)

    x, y = points[:, 0], points[:, 1]
    x1, y1 = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:, 0], poly[1:, 1]

    # Does the horizontal ray from (x, y) to +inf cross each edge?
    straddles = (y1[None, :] > y[:, None]) != (y2[None, :] > y[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1[None, :] + (y[:, None] - y1[None, :]) * (
            (x2 - x1)[None, :] / (y2 - y1)[None, :]
        )
    crossings = straddles & (x[:, None] < x_cross)
    return (crossings.sum(axis=1) % 2) == 1
