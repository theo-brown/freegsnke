"""
Multi-droplet plasma profiles.

Copyright 2026 FreeGSNKE contributors.

SPDX-License-Identifier: LGPL-3.0-or-later

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.

Overview
--------
The standard profile classes describe a single plasma: one magnetic axis
(O-point), one last closed flux surface and one total current. This module
provides :class:`Droplets`, a composite profile object describing several
disjoint plasma "droplets", each enclosed by its own separatrix (e.g. an upper
and a lower droplet, as in merging-compression start-up or in up/down
"droplet" configurations).

Each droplet is described by an ordinary FreeGSNKE profile object (e.g.
:class:`freegsnke.jtor_update.ConstrainPaxisIp`) carrying its own current
``Ip`` and profile shape, together with a *search region*: a boolean grid mask
in which that droplet's magnetic axis is looked for. At each evaluation of
``Jtor`` and for every droplet independently:

1. the droplet's O-point is the strongest (current-sign aware) O-point found
   inside ``limiter & search_region``;
2. its separatrix X-point is the X-point closest in flux to that O-point that
   is reachable along a monotonic line of sight (the same criterion used for
   single plasmas in :func:`freegsnke.critical.find_critical`);
3. its core mask is obtained by flood-filling from its own O-point
   (:func:`freegsnke.critical.inside_mask`) and is then checked against the
   limiter (:meth:`freegsnke.limiter_func.Limiter_handler.core_mask_limiter`);
   if no valid X-point is found, a contour-based search that excludes the other
   droplets' axes is used instead;
4. the droplet's current density is evaluated by its own profile's
   ``Jtor_part2`` using its own axis flux, boundary flux and mask, so that each
   droplet carries exactly its own ``Ip``.

The total current density is the sum over droplets. The object exposes the
interface used by the static solvers (``Jtor``, ``jtor``, ``opt``, ``xpt``,
``psi_bndry``, ``flag_limiter``, ``diverted_core_mask``,
``limiter_core_mask``, ``copy``), so it can be passed to
``NKGSsolver.forward_solve`` and ``NKGSsolver.solve`` in place of a
single-plasma profile.

Limitations
-----------
* Static (forward and inverse) solves only. The time-dependent solvers assume
  a single plasma current and are not supported.
* Sub-grid refinement of ``jtor`` (``refine_jtor``) is not supported.
* Equilibrium-level diagnostics that assume a single axis and boundary (q
  profile, betap, ``eq.psi_axis``/``eq.psi_bndry``, separatrix tracing, ...)
  refer to the *primary* droplet (index 0) only. Per-droplet quantities are
  available via :meth:`Droplets.droplet_currents`, :attr:`Droplets.droplets`
  and the ``opt``/``xpt``/``psi_bndry`` attributes of each sub-profile.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import maximum_filter
from skimage import measure

from . import critical


class Droplets:
    """
    Composite profile describing several disjoint plasma droplets.

    Parameters
    ----------
    eq : FreeGSNKE Equilibrium object
        Equilibrium defining the computational grid and limiter.
    profiles : list
        One FreeGSNKE profile object per droplet (e.g. ``ConstrainPaxisIp``,
        ``ConstrainBetapIp``, ``Lao85``), each built with the same ``eq`` and
        carrying the current ``Ip`` of that droplet. Profiles must use the
        unrefined ``jtor`` (``refine_jtor=False``).
    search_regions : list of np.ndarray
        One boolean array of shape ``(eq.nx, eq.ny)`` per droplet, marking where
        that droplet's magnetic axis should be looked for. Regions must not
        overlap. The droplet's core is *not* clipped to its search region.

    Attributes
    ----------
    droplets : list
        The per-droplet profile objects. After a call to :meth:`Jtor` each holds
        its own ``opt``, ``xpt``, ``psi_axis``, ``psi_bndry``, ``jtor``,
        ``limiter_core_mask`` and ``flag_limiter``.
    jtor : np.ndarray
        Total toroidal current density (sum over droplets).
    opt : np.ndarray
        O-points of the droplets, one row ``[R, Z, psi]`` per droplet, in droplet
        order (row 0 is the primary droplet).
    xpt : np.ndarray
        Separatrix X-points of the diverted droplets, one row per droplet that
        has one, in droplet order.
    psi_bndry : float
        Boundary flux of the primary droplet (see the module limitations).
    """

    def __init__(self, eq: object, profiles: list, search_regions: list) -> None:
        if len(profiles) < 1:
            raise ValueError("At least one droplet profile is required.")
        if len(profiles) != len(search_regions):
            raise ValueError("Provide exactly one search region per droplet.")

        self.droplets = list(profiles)
        self.search_regions = []
        for region in search_regions:
            region = np.asarray(region, dtype=bool)
            if region.shape != np.shape(eq.R):
                raise ValueError(
                    "Each search region must have the shape of the grid "
                    f"{np.shape(eq.R)}, got {region.shape}."
                )
            self.search_regions.append(region)
        overlap = np.sum(np.asarray(self.search_regions, dtype=int), axis=0) > 1
        if np.any(overlap):
            raise ValueError("Droplet search regions must not overlap.")

        for p in self.droplets:
            if getattr(p, "_refine_jtor", False):
                raise NotImplementedError(
                    "Sub-grid jtor refinement is not supported for droplets."
                )

        self.limiter_handler = eq.limiter_handler
        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = self.limiter_handler.mask_outside_limiter
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out
        self.eqRidx = self.limiter_handler.eqRidx
        self.eqZidx = self.limiter_handler.eqZidx
        self.dR_dZ = self.limiter_handler.dR_dZ
        self.R0Z0 = self.limiter_handler.R0Z0
        self.nx, self.ny = np.shape(eq.R)
        self.dRdZ = (eq.R_1D[1] - eq.R_1D[0]) * (eq.Z_1D[1] - eq.Z_1D[0])

        self.Ip_logic = True
        self.has_relevant_xpoint = False

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def n_droplets(self) -> int:
        """Number of droplets."""
        return len(self.droplets)

    @property
    def Ip(self) -> float:
        """Total plasma current (sum of the droplets' currents) [A]."""
        return float(np.sum([p.Ip for p in self.droplets]))

    def fvac(self) -> float:
        """Vacuum toroidal field parameter R*Bt, taken from the primary droplet."""
        return self.droplets[0].fvac()

    def pprime(self, pn):
        """pprime of the primary droplet (see module limitations)."""
        return self.droplets[0].pprime(pn)

    def ffprime(self, pn):
        """ffprime of the primary droplet (see module limitations)."""
        return self.droplets[0].ffprime(pn)

    def pressure(self, pn):
        """Pressure of the primary droplet (see module limitations)."""
        return self.droplets[0].pressure(pn)

    def fpol(self, pn):
        """Poloidal current function of the primary droplet (see limitations)."""
        return self.droplets[0].fpol(pn)

    def droplet_currents(self) -> np.ndarray:
        """
        Plasma current actually carried by each droplet in the last evaluation.

        Returns
        -------
        np.ndarray
            Current of each droplet [A], in droplet order.
        """
        return np.array([np.sum(p.jtor) * self.dRdZ for p in self.droplets])

    # ------------------------------------------------------------------
    # Critical points and core masks of a single droplet
    # ------------------------------------------------------------------
    def _droplet_opoint(
        self, k: int, opoints: np.ndarray, psi: np.ndarray, sign: float
    ) -> np.ndarray:
        """Select droplet k's magnetic axis among the candidate O-points."""
        allowed = self.mask_inside_limiter * self.search_regions[k]
        if len(opoints):
            # nearest grid node
            iR = np.argmin(np.abs(self.R_1D[None, :] - opoints[:, :1]), axis=1)
            iZ = np.argmin(np.abs(self.Z_1D[None, :] - opoints[:, 1:2]), axis=1)
            iR = np.clip(iR, 1, self.nx - 2)
            iZ = np.clip(iZ, 1, self.ny - 2)
            inside = allowed[iR, iZ]
            # an axis must be a maximum of sign*psi (negative curvature)
            curvature = (
                psi[iR + 1, iZ]
                + psi[iR - 1, iZ]
                + psi[iR, iZ + 1]
                + psi[iR, iZ - 1]
                - 4 * psi[iR, iZ]
            )
            is_max = sign * curvature < 0
            candidates = opoints[inside * is_max]
            if len(candidates):
                return candidates[np.argmax(sign * candidates[:, 2])]

        # fall back to the grid maximum within the region
        if not np.any(allowed):
            raise ValueError(f"Search region of droplet {k} is outside the limiter.")
        masked = np.where(allowed, sign * psi, -np.inf)
        i, j = np.unravel_index(np.argmax(masked), masked.shape)
        return np.array([self.R_1D[i], self.Z_1D[j], psi[i, j]])

    def _droplet_xpoints(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        psi: np.ndarray,
        opt: np.ndarray,
        xpoints: np.ndarray,
        sign: float,
    ) -> np.ndarray:
        """
        X-points relevant to the droplet with axis ``opt``, ordered so that the
        first row is the droplet's separatrix X-point.
        """
        if len(xpoints) == 0:
            return np.zeros((0, 3))
        # X-points with the correct flux ordering with respect to the axis
        xpt = xpoints[sign * (xpoints[:, 2] - opt[2]) < 0]
        # closest in flux first
        xpt = xpt[np.argsort(np.abs(xpt[:, 2] - opt[2]))]
        # discard X-points not on a monotonic line of sight from the axis
        while len(xpt) and not critical.discard_xpoints_f(R, Z, psi, opt, xpt[0]):
            xpt = xpt[1:]
        return xpt

    def _contour_core(
        self,
        k: int,
        psi: np.ndarray,
        opt: np.ndarray,
        other_opts: list,
        sign: float,
        rel_tolerance: float = 1e-6,
        starting_dx: float = 0.05,
    ) -> tuple[np.ndarray, float]:
        """
        Contour-based fallback for droplet k: find the lowest flux level whose
        closed contour around the droplet's axis encloses neither another
        droplet's axis nor any flux maximum outside the limiter or on the edge
        of the domain.
        """
        psi_map = sign * np.copy(psi)
        min_psi = np.amin(psi_map)
        psi_map[:, 0] = psi_map[0, :] = psi_map[-1, :] = psi_map[:, -1] = min_psi
        del_psi = np.amax(psi_map) - min_psi
        psi_map = (psi_map - min_psi) / del_psi

        idx_axis = np.array(
            [
                np.argmin(np.abs(self.R_1D - opt[0])),
                np.argmin(np.abs(self.Z_1D - opt[1])),
            ],
            dtype=float,
        )
        axis_level = psi_map[int(idx_axis[0]), int(idx_axis[1])]

        maxima = maximum_filter(psi_map, size=3) == psi_map
        excluded = maxima * (self.mask_outside_limiter > 0)
        excluded[1, :] = excluded[:, 1] = excluded[-2, :] = excluded[:, -2] = True
        excluded_idx = [
            np.array([self.eqRidx[excluded], self.eqZidx[excluded]]).T.astype(float)
        ]
        for o in other_opts:
            excluded_idx.append(
                np.array(
                    [
                        [
                            np.argmin(np.abs(self.R_1D - o[0])),
                            np.argmin(np.abs(self.Z_1D - o[1])),
                        ]
                    ],
                    dtype=float,
                )
            )
        excluded_idx = np.concatenate(excluded_idx, axis=0)

        def enclosing_path(level):
            regions = measure.find_contours(psi_map, level)
            order = np.argsort(
                [np.linalg.norm(np.mean(r, axis=0) - idx_axis) for r in regions]
            )
            for i in order:
                path = Path(regions[i])
                if path.contains_point(idx_axis):
                    return path
            return None

        increment = -starting_dx
        level = axis_level + increment
        desired_check_larger = True
        best_path, best_level = None, None
        n_iter = 0
        while (abs(increment) > rel_tolerance or not desired_check_larger) and (
            n_iter < 200
        ):
            n_iter += 1
            path = enclosing_path(level)
            if path is None:
                check_larger = True
            else:
                check_larger = np.any(path.contains_points(excluded_idx))
                if not check_larger:
                    best_path, best_level = path, level
            if check_larger == desired_check_larger:
                desired_check_larger = not desired_check_larger
                increment *= -0.5
            level += increment

        if best_path is None:
            raise ValueError(f"Could not identify a closed core for droplet {k}.")
        core = best_path.contains_points(self.limiter_handler.idx_grid_points)
        core = core.reshape((self.nx, self.ny))
        psi_bndry = sign * (best_level * del_psi + min_psi)
        return core, psi_bndry

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    def Jtor(
        self,
        R: np.ndarray,
        Z: np.ndarray,
        psi: np.ndarray,
        psi_bndry: float | None = None,
    ) -> np.ndarray:
        """
        Total toroidal current density of all droplets.

        Parameters
        ----------
        R, Z : np.ndarray
            Grid coordinates, shape (nx, ny).
        psi : np.ndarray
            Total poloidal flux [Webers/2pi], shape (nx, ny).
        psi_bndry : None
            Prescribing a boundary flux is not supported for droplets.

        Returns
        -------
        np.ndarray
            Total toroidal current density [A/m^2].
        """
        if psi_bndry is not None:
            raise NotImplementedError(
                "Prescribing psi_bndry is not supported for droplets: each "
                "droplet determines its own boundary."
            )
        self.R_1D = R[:, 0]
        self.Z_1D = Z[0, :]

        opoints, xpoints = critical.scan_for_crit(R, Z, psi)
        opoints = np.asarray(opoints).reshape(-1, 3)
        xpoints = np.asarray(xpoints).reshape(-1, 3)

        signs = [float(np.sign(p.Ip)) or 1.0 for p in self.droplets]
        axes = [
            self._droplet_opoint(k, opoints, psi, signs[k])
            for k in range(self.n_droplets)
        ]

        claimed = np.zeros_like(psi, dtype=bool)
        jtor = np.zeros_like(psi)
        all_xpt = []
        diverted_masks = []
        flags = []
        for k, p in enumerate(self.droplets):
            sign = signs[k]
            opt = axes[k]
            xpt = self._droplet_xpoints(R, Z, psi, opt, xpoints, sign)

            diverted_core = None
            if len(xpt):
                d_bndry = xpt[0, 2]
                diverted_core = critical.inside_mask(
                    R,
                    Z,
                    psi,
                    opt[np.newaxis],
                    xpt,
                    self.mask_outside_limiter,
                    d_bndry,
                ).astype(bool)
                # reject flood fills that leak into another droplet's axis
                for j, o in enumerate(axes):
                    if j != k:
                        i_o = np.argmin(np.abs(self.R_1D - o[0]))
                        j_o = np.argmin(np.abs(self.Z_1D - o[1]))
                        if diverted_core[i_o, j_o]:
                            diverted_core = None
                            break
            if diverted_core is None or not np.any(diverted_core):
                diverted_core, d_bndry = self._contour_core(
                    k, psi, opt, [axes[j] for j in range(len(axes)) if j != k], sign
                )
                xpt = np.zeros((0, 3))

            core_in = diverted_core * self.mask_inside_limiter * np.logical_not(claimed)
            psi_b, core, flag = self.limiter_handler.core_mask_limiter(
                psi, d_bndry, core_in, self.limiter_mask_out, sign
            )
            if np.sum(core * self.mask_inside_limiter) == 0:
                core = core_in.astype(float)
                psi_b = d_bndry
            core = core.astype(bool) * self.mask_inside_limiter
            if not np.any(core):
                raise ValueError(f"Droplet {k} has an empty core.")
            claimed |= core

            jtor_k = p.Jtor_part2(R, Z, psi, opt[2], psi_b, core)

            # store per-droplet state on the sub-profile
            p.opt = opt[np.newaxis]
            p.xpt = xpt
            p.psi_axis = opt[2]
            p.psi_bndry = psi_b
            p.diverted_core_mask = diverted_core
            p.limiter_core_mask = core
            p.flag_limiter = bool(flag)
            p.has_relevant_xpoint = len(xpt) > 0
            p.jtor = jtor_k
            p.inputs = [opt[2], psi_b, core]

            jtor += jtor_k
            if len(xpt):
                all_xpt.append(xpt[0])
            diverted_masks.append(diverted_core)
            flags.append(bool(flag))

        primary = self.droplets[0]
        self.jtor = jtor
        self.opt = np.array(axes)
        self.xpt = np.array(all_xpt).reshape(-1, 3)
        self.psi_axis = primary.psi_axis
        self.psi_bndry = primary.psi_bndry
        self.flag_limiter = primary.flag_limiter
        self.has_relevant_xpoint = primary.has_relevant_xpoint
        self.diverted_core_mask = np.any(diverted_masks, axis=0)
        self.limiter_core_mask = claimed
        return jtor

    def copy(self) -> Droplets:
        """
        Copy of the composite profile; each droplet profile is copied with its
        own ``copy`` method and the per-droplet state is carried over.
        """
        obj = type(self).__new__(type(self))
        obj.__dict__.update(self.__dict__)
        obj.droplets = []
        for p in self.droplets:
            q = p.copy()
            for attr in ("opt", "xpt", "psi_axis", "psi_bndry", "jtor"):
                if hasattr(p, attr):
                    setattr(q, attr, np.copy(getattr(p, attr)))
            for attr in (
                "diverted_core_mask",
                "limiter_core_mask",
                "flag_limiter",
                "has_relevant_xpoint",
            ):
                if hasattr(p, attr):
                    setattr(q, attr, getattr(p, attr))
            obj.droplets.append(q)
        obj.search_regions = [np.copy(r) for r in self.search_regions]
        for attr in (
            "jtor",
            "opt",
            "xpt",
            "diverted_core_mask",
            "limiter_core_mask",
        ):
            if hasattr(self, attr):
                setattr(obj, attr, np.copy(getattr(self, attr)))
        return obj


class DoubleDroplet(Droplets):
    """
    Two droplets, one above and one below a horizontal line ``Z = Z_split``.

    Parameters
    ----------
    eq : FreeGSNKE Equilibrium object
        Equilibrium defining the computational grid and limiter.
    upper : profile object
        Profile (with its own ``Ip``) of the droplet above ``Z_split``. This is
        the primary droplet.
    lower : profile object
        Profile (with its own ``Ip``) of the droplet below ``Z_split``.
    Z_split : float, optional
        Height [m] separating the search regions of the two magnetic axes.

    Examples
    --------
    >>> from freegsnke.jtor_update import ConstrainPaxisIp
    >>> upper = ConstrainPaxisIp(eq, 2e3, 1.5e5, 0.5, alpha_m=1.8, alpha_n=1.2)
    >>> lower = ConstrainPaxisIp(eq, 2e3, 1.5e5, 0.5, alpha_m=1.8, alpha_n=1.2)
    >>> profiles = DoubleDroplet(eq, upper, lower, Z_split=0.0)
    >>> NK.forward_solve(eq, profiles, 1e-8)
    """

    def __init__(
        self, eq: object, upper: object, lower: object, Z_split: float = 0.0
    ) -> None:
        self.Z_split = Z_split
        super().__init__(eq, [upper, lower], [eq.Z > Z_split, eq.Z <= Z_split])
