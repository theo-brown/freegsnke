"""
Enables FreeGSNKE-simulated equilibrium data to be read/written to/from IMAS
IDS (via the IMAS-Python package), including netCDF serialisation.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

from datetime import date

import contourpy
import imas
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RectBivariateSpline

import freegsnke

# The IMAS equilibrium IDS is written in the data dictionary's COCOS 17
# convention: poloidal flux in Wb (i.e. including the 2*pi factor) whereas
# FreeGSNKE (like FreeGS) works in Wb/rad. Derivatives with respect to the
# poloidal flux (dp/dpsi, FF') are therefore rescaled by this factor on
# write/read.
_PSI_IDS_OVER_PSI_FREEGSNKE = 2 * np.pi

# number of points used when integrating p' and FF' to obtain p and F
_N_PROFILE_INTEGRATION_POINTS = 4001

# number of innermost flux surfaces used to extrapolate q to the magnetic axis
_N_SURFACES_FOR_AXIS_EXTRAPOLATION = 4

_IDS_FACTORY = None


def _ids_factory():
    """
    Returns a module-level cached `imas.IDSFactory` (parsing the data
    dictionary is expensive and only needs doing once per session).
    """
    global _IDS_FACTORY
    if _IDS_FACTORY is None:
        _IDS_FACTORY = imas.IDSFactory()
    return _IDS_FACTORY


def _integrated_profiles(profiles, psi_n, psi_axis, psi_bndry):
    """
    Evaluates the pressure and toroidal field function profiles by integrating
    the (normalised) p' and FF' profiles from the plasma boundary inwards:

        p(psi_n) = (psi_axis - psi_bndry) * integral_{psi_n}^{1} p'(x) dx,
        F(psi_n)^2 = fvac^2 + 2 (psi_axis - psi_bndry) * integral_{psi_n}^{1} FF'(x) dx,

    which is the same definition used by `freegs4e.jtor.Profile.pressure` and
    `freegs4e.jtor.Profile.fpol`, evaluated with a single vectorised
    cumulative trapezoidal integration on a fine grid rather than one adaptive
    quadrature per point.

    Parameters
    ----------
    profiles : freegsnke.jtor_update profile object
        Profile object (with `pprime`, `ffprime` and `fvac` methods).
    psi_n : np.array
        Normalised flux values at which to evaluate the profiles.
    psi_axis : float
        Poloidal flux on the magnetic axis [Wb/rad].
    psi_bndry : float
        Poloidal flux on the plasma boundary [Wb/rad].

    Returns
    -------
    pressure : np.array
        Pressure at each value of `psi_n` [Pa].
    fpol : np.array
        F = R*Btor at each value of `psi_n` [T m].
    """

    fine = np.linspace(0.0, 1.0, _N_PROFILE_INTEGRATION_POINTS)
    dpsi = psi_axis - psi_bndry

    pprime_fine = np.asarray(profiles.pprime(fine), dtype=float)
    ffprime_fine = np.asarray(profiles.ffprime(fine), dtype=float)

    # integral from psi_n to 1 = total integral - cumulative integral from 0
    int_pprime = cumulative_trapezoid(pprime_fine, fine, initial=0.0)
    int_pprime = int_pprime[-1] - int_pprime
    int_ffprime = cumulative_trapezoid(ffprime_fine, fine, initial=0.0)
    int_ffprime = int_ffprime[-1] - int_ffprime

    psi_n = np.clip(np.asarray(psi_n, dtype=float), 0.0, 1.0)
    pressure = np.interp(psi_n, fine, int_pprime * dpsi)
    fpol = np.interp(
        psi_n, fine, np.sqrt(2.0 * int_ffprime * dpsi + profiles.fvac() ** 2)
    )
    return pressure, fpol


def _flux_surface_geometry(eq, psi_n, fpol_1d, fields_2d=None):
    """
    Traces each requested normalised-flux surface once (via `contourpy`, the
    same technique used by `eq.flux_averaged_function`) and returns the
    geometric quantities and flux-surface averages needed to build several
    `profiles_1d` IDS fields, without re-tracing the surfaces for each one.

    Parameters
    ----------
    eq : freegsnke.equilibrium_update.Equilibrium
        Solved equilibrium object.
    psi_n : np.array
        Normalised flux values to trace (each strictly between 0 and 1).
    fpol_1d : np.array
        F = R*Btor at each value in `psi_n` (e.g. `profiles.fpol(psi_n)`).
        F is constant on a flux surface, so this gives Btor = F/R at every
        traced point without any extra profile evaluation.
    fields_2d : dict, optional
        Mapping from a name to a 2D field interpolator (callable as
        `f(R, Z, grid=False)`, e.g. a `RectBivariateSpline`). The flux
        surface average of each field is returned under `avg_<name>`.

    Returns
    -------
    dict of np.array
        `r_inboard`, `r_outboard` : major radius of the surface at the
            magnetic-axis height, on the inboard/outboard side [m].
        `volume` : volume enclosed by the surface [m^3], from the exact
            line-integral form of Pappus's theorem, V = |∮ pi R^2 dZ|.
        `elongation`, `triangularity_upper`, `triangularity_lower` : standard
            Miller-style shape parameters, using the surface's own
            (Rmin, Rmax, Zmin, Zmax) extent - the same formulas as
            freegs4e's `geometricElongation`/`triangularity_upper/lower`,
            generalised from the LCFS to an arbitrary internal surface.
        `q` : safety factor, q = (F / 2 pi) * ∮ dl / (R^2 Bp) (with the flux
            in Wb/rad, i.e. dPhi/dpsi_total).
        `avg_inv_R`, `avg_inv_R2` : flux-surface averages <1/R>, <1/R^2>.
        `avg_R_Bp`, `avg_R2_Bp2`, `avg_Bp2` : flux-surface averages
            <R*Bp>, <R^2*Bp^2>, <Bp^2>, from which gm2/gm3/gm7 are built
            (see `write_equilibrium_to_ids`).
        `avg_B2`, `avg_inv_B2` : flux-surface averages <B^2>, <1/B^2> of the
            *total* field (poloidal + toroidal), for gm5/gm4.
        `avg_<name>` : flux-surface average of each field in `fields_2d`.
    """

    if fields_2d is None:
        fields_2d = {}

    # normalised total flux on the grid (from the cached plasma + coil fluxes)
    psi_n_2d = (eq.psi() - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
    masked_psi = np.ma.array(psi_n_2d, mask=eq.mask_outside_limiter)
    mag_r, mag_z = eq.magneticAxis()[0:2]

    # total-psi spline, reused for cheap Br/Bz on each surface (see the
    # equivalent optimisation in eq.flux_averaged_function)
    psi_total_func = RectBivariateSpline(eq.R_1D, eq.Z_1D, eq.psi())

    cont_gen = contourpy.contour_generator(
        x=eq.R, y=eq.Z, z=masked_psi, line_type=contourpy.LineType.Separate
    )

    n = len(psi_n)
    result = {
        key: np.full(n, np.nan)
        for key in [
            "r_inboard",
            "r_outboard",
            "volume",
            "q",
            "elongation",
            "triangularity_upper",
            "triangularity_lower",
            "avg_inv_R",
            "avg_inv_R2",
            "avg_R_Bp",
            "avg_R2_Bp2",
            "avg_Bp2",
            "avg_B2",
            "avg_inv_B2",
        ]
        + [f"avg_{name}" for name in fields_2d]
    }

    # loop over each poloidal flux surface
    for i, val in enumerate(psi_n):

        # get the coords
        lines = [line for line in cont_gen.lines(val) if line.shape[0] > 0]
        distances = [
            np.min(np.linalg.norm(line - [mag_r, mag_z], axis=1)) for line in lines
        ]
        flux_surface = lines[np.argmin(distances)]
        Rc, Zc = flux_surface[:, 0], flux_surface[:, 1]

        # arc length and Bp along the surface
        dl = np.sqrt(np.diff(Rc) ** 2 + np.diff(Zc) ** 2)
        l_cum = np.concatenate(([0], np.cumsum(dl)))
        Br = -psi_total_func(Rc, Zc, dy=1, grid=False) / Rc
        Bz = psi_total_func(Rc, Zc, dx=1, grid=False) / Rc
        Bp2 = Br**2 + Bz**2
        Bp_inv = 1 / np.sqrt(Bp2)
        norm = trapezoid(Bp_inv, l_cum)  # = integral of dl/Bp

        def flux_average(values, Bp_inv=Bp_inv, l_cum=l_cum, norm=norm):
            return trapezoid(values * Bp_inv, l_cum) / norm

        # q = dPhi/dpsi_total = (F / 2 pi) ∮ dl / (R^2 Bp) (psi here in Wb/rad)
        result["q"][i] = fpol_1d[i] * trapezoid(Bp_inv / Rc**2, l_cum) / (2 * np.pi)
        result["avg_inv_R"][i] = flux_average(1 / Rc)
        result["avg_inv_R2"][i] = flux_average(1 / Rc**2)
        result["avg_R_Bp"][i] = flux_average(Rc * np.sqrt(Bp2))
        result["avg_R2_Bp2"][i] = flux_average(Rc**2 * Bp2)
        result["avg_Bp2"][i] = flux_average(Bp2)

        # total field: F is constant on the surface, so Btor = F/R here
        Btor = fpol_1d[i] / Rc
        B2 = Bp2 + Btor**2
        result["avg_B2"][i] = flux_average(B2)
        result["avg_inv_B2"][i] = flux_average(1 / B2)

        # any additional 2D fields
        for name, field in fields_2d.items():
            result[f"avg_{name}"][i] = flux_average(field(Rc, Zc, grid=False))

        # Miller-style shape parameters
        Rmax_i, Rmin_i = np.max(Rc), np.min(Rc)
        Zmax_i, Zmin_i = np.max(Zc), np.min(Zc)
        R_geo = 0.5 * (Rmax_i + Rmin_i)
        a_minor = 0.5 * (Rmax_i - Rmin_i)
        result["elongation"][i] = (Zmax_i - Zmin_i) / (Rmax_i - Rmin_i)
        result["triangularity_upper"][i] = (R_geo - Rc[np.argmax(Zc)]) / a_minor
        result["triangularity_lower"][i] = (R_geo - Rc[np.argmin(Zc)]) / a_minor

        # r_inboard/r_outboard: where the surface crosses the magnetic-axis
        # height (the midplane), on either side of the axis
        dz = Zc - mag_z
        crossing_indices = np.where(np.diff(np.sign(dz)) != 0)[0]
        crossings = []
        for k in crossing_indices:
            f = (mag_z - Zc[k]) / (Zc[k + 1] - Zc[k])
            crossings.append(Rc[k] + f * (Rc[k + 1] - Rc[k]))
        crossings = np.array(crossings)
        inboard = crossings[crossings < mag_r]
        outboard = crossings[crossings > mag_r]
        if inboard.size:
            result["r_inboard"][i] = np.min(inboard)
        if outboard.size:
            result["r_outboard"][i] = np.max(outboard)

        # enclosed volume, exact for the piecewise-linear traced surface:
        # Green's theorem with Q = pi*R^2 gives ∫∫ 2*pi*R dR dZ = ∮ pi*R^2 dZ
        result["volume"][i] = np.abs(trapezoid(np.pi * Rc**2, Zc))

    return result


def write_equilibrium_to_ids(
    eq,
    profiles,
    psi_n=None,
    time=0.0,
):
    """
    Populates an IMAS `equilibrium` IDS (single time slice) with quantities taken
    from a solved FreeGSNKE equilibrium.

    The IDS follows the IMAS data dictionary convention (COCOS 17): the poloidal
    flux is stored in Wb (FreeGSNKE's Wb/rad multiplied by 2*pi) and the
    derivatives `dpressure_dpsi` and `f_df_dpsi` are with respect to that flux.

    Parameters
    ----------
    eq : freegsnke.equilibrium_update.Equilibrium
        Solved equilibrium object.
    profiles : freegsnke.jtor_update profile object
        The profile object used to solve for `eq` (e.g. a `ConstrainPaxisIp` or
        `GeneralPprimeFFprime` instance).
    psi_n : np.array, optional
        Normalised poloidal flux values of the flux surfaces on which the 1D
        profiles are written. Values must lie strictly inside (0, 1) since flux
        surfaces cannot be traced on the axis or (for diverted plasmas) on the
        separatrix. Defaults to `eq.nx` points clipped to [0.01, 0.99]. The
        magnetic axis (psi_norm = 0) and near-axis points between the axis and
        the innermost surface are always prepended to the 1D profiles, with
        their values prescribed from the leading-order near-axis behaviour of
        each quantity.
    time : float, optional
        Time [s] assigned to the IDS time slice (default 0.0).

    Returns
    -------
    ids_out : imas.ids_toplevel.IDSToplevel
        The populated `equilibrium` IDS.
    """

    # initialise an empty equilibrium IDS
    ids_out = _ids_factory().equilibrium()

    # high-level ids properties
    ids_out.ids_properties.name = "FreeGSNKE-generated equilibrium IDS"
    ids_out.ids_properties.homogeneous_time = 1
    ids_out.ids_properties.creation_date = date.today().strftime("%d-%m-%Y")

    # code properties
    ids_out.code.name = freegsnke.__name__
    ids_out.code.description = (
        "A Python-based free-boundary evolutive Grad-Shafranov equilibrium solver."
    )
    ids_out.code.version = freegsnke.__version__
    ids_out.code.repository = "https://github.com/FusionComputingLab/freegsnke"

    # vacuum toroidal field properties (rcentr taken as centre of limiter geometry)
    rcentr = 0.5 * (np.min(eq.tokamak.limiter.R) + np.max(eq.tokamak.limiter.R))
    ids_out.vacuum_toroidal_field.r0 = rcentr
    ids_out.vacuum_toroidal_field.b0 = np.array([profiles.fvac() / rcentr])

    ids_out.time = np.array([float(time)])
    ids_out.time_slice.resize(1)
    time_slice = ids_out.time_slice[0]
    time_slice.time = float(time)

    # boundary quantities
    time_slice.boundary.type = 1 - profiles.flag_limiter  # 0 = limited, 1 = diverted
    time_slice.boundary.psi_norm = 1.0
    time_slice.boundary.psi = _PSI_IDS_OVER_PSI_FREEGSNKE * eq.psi_bndry
    time_slice.boundary.minor_radius = eq.minorRadius()
    boundary = eq.separatrix(ntheta=360)
    time_slice.boundary.outline.r = boundary[:, 0]
    time_slice.boundary.outline.z = boundary[:, 1]

    # global quantities
    time_slice.global_quantities.ip = eq.plasmaCurrent()
    time_slice.global_quantities.psi_axis = _PSI_IDS_OVER_PSI_FREEGSNKE * eq.psi_axis
    time_slice.global_quantities.psi_boundary = (
        _PSI_IDS_OVER_PSI_FREEGSNKE * eq.psi_bndry
    )
    mag_r, mag_z = eq.magneticAxis()[0:2]
    time_slice.global_quantities.magnetic_axis.r = mag_r
    time_slice.global_quantities.magnetic_axis.z = mag_z

    # 1D profile quantities. psi_n is clipped away from the exact axis/
    # boundary values (0, 1): q (and everything derived from it below) is
    # singular there, and eq.psiN_1D(N) would otherwise include them exactly.
    # Matches the default clip range already used by eq.flux_averaged_function.
    if psi_n is None:
        N = eq.nx
        psi_n = np.clip(eq.psiN_1D(N), 0.01, 0.99)
    psi_n_surfaces = np.asarray(psi_n, dtype=float).reshape(-1)
    if (
        np.any(psi_n_surfaces <= 0.0)
        or np.any(psi_n_surfaces >= 1.0)
        or np.any(np.diff(psi_n_surfaces) <= 0)
    ):
        raise ValueError(
            "psi_n must be strictly increasing and lie strictly inside (0, 1)."
        )
    # The 1D profiles are written on the magnetic axis (psi_n = 0, where the
    # values are prescribed analytically since no flux surface can be traced),
    # on near-axis fill points (see below) and on the requested flux surfaces.
    # F (constant on each surface) is needed for the flux surface tracing.
    _, fpol_surfaces = _integrated_profiles(
        profiles, psi_n_surfaces, eq.psi_axis, eq.psi_bndry
    )

    # All flux-surface quantities (including the safety factor and the
    # flux-averaged toroidal current density) derive from one pass of
    # flux-surface tracing, with the same normalisation of the flux as used
    # for psi_n here. (Note that `eq.q` normalises the flux to the X-point
    # whenever one exists in the domain, which is not the plasma boundary for a
    # limiter-bound plasma.)
    jtor_interp = RectBivariateSpline(eq.R_1D, eq.Z_1D, profiles.jtor)
    geom = _flux_surface_geometry(
        eq, psi_n_surfaces, fpol_surfaces, fields_2d={"jtor": jtor_interp}
    )
    mag_r, mag_z = eq.magneticAxis()[0:2]
    jtor_axis = float(jtor_interp(mag_r, mag_z, grid=False))

    # safety factor on the flux surfaces, extrapolated to the axis with a
    # low-order polynomial in psi_n (q is smooth in psi near the axis)
    q_surfaces = geom["q"]
    n_fit = min(_N_SURFACES_FOR_AXIS_EXTRAPOLATION, len(psi_n_surfaces))
    q_axis_fit = np.polyfit(
        psi_n_surfaces[:n_fit], q_surfaces[:n_fit], deg=min(2, n_fit - 1)
    )
    q_axis = np.polyval(q_axis_fit, 0.0)

    # Flux surfaces very close to the axis cannot be traced reliably on the
    # computational grid, which would leave a gap in the profiles between the
    # axis and the innermost traced surface (sqrt(psi_n_min) in the normalised
    # radius). The gap is filled with the leading-order near-axis behaviour of
    # each quantity: with s = rho/rho_1 (rho_1 the innermost surface, rho the
    # near-axis flux surface radius, so psi_n = s^2 psi_n_1 and s ~ sqrt(psi_n)),
    # even quantities vary as a + b s^2, Bp ~ s, |grad(rho)| ~ 1, enclosed
    # volume ~ s^2 and the midplane radii ~ s. The fill spacing in s matches
    # that of the innermost traced surfaces.
    psi_n_1 = psi_n_surfaces[0]
    n_fill = 1
    if len(psi_n_surfaces) > 1:
        ds_surfaces = np.sqrt(psi_n_surfaces[1]) - np.sqrt(psi_n_1)
        n_fill = max(1, int(round(np.sqrt(psi_n_1) / ds_surfaces)))
    s_fill = np.arange(1, n_fill) / n_fill
    psi_n_fill = s_fill**2 * psi_n_1

    def assemble(axis_value, fill_values, surface_values):
        return np.concatenate(
            (
                [axis_value],
                np.asarray(fill_values, dtype=float).reshape(-1),
                np.asarray(surface_values, dtype=float).reshape(-1),
            )
        )

    def even(axis_value, key):
        # a + b s^2 between the axis and the innermost surface
        return assemble(
            axis_value, axis_value + s_fill**2 * (geom[key][0] - axis_value), geom[key]
        )

    psi_n = np.concatenate(([0.0], psi_n_fill, psi_n_surfaces))
    psi_actual = eq.psi_axis + psi_n * (eq.psi_bndry - eq.psi_axis)
    time_slice.profiles_1d.psi = _PSI_IDS_OVER_PSI_FREEGSNKE * psi_actual
    time_slice.profiles_1d.psi_norm = psi_n
    pressure_1d, fpol_1d = _integrated_profiles(
        profiles, psi_n, eq.psi_axis, eq.psi_bndry
    )
    time_slice.profiles_1d.pressure = pressure_1d
    time_slice.profiles_1d.f = fpol_1d
    btor_axis = fpol_1d[0] / mag_r
    time_slice.profiles_1d.dpressure_dpsi = (
        np.asarray(profiles.pprime(psi_n)).reshape(-1) / _PSI_IDS_OVER_PSI_FREEGSNKE
    )
    time_slice.profiles_1d.f_df_dpsi = (
        np.asarray(profiles.ffprime(psi_n)).reshape(-1) / _PSI_IDS_OVER_PSI_FREEGSNKE
    )
    q_1d = np.concatenate(([q_axis], np.polyval(q_axis_fit, psi_n_fill), q_surfaces))
    time_slice.profiles_1d.q = q_1d
    time_slice.profiles_1d.j_phi = even(jtor_axis, "avg_jtor")

    # toroidal flux from dphi/dpsi = q (psi here in Wb, hence the 2*pi)
    psi_mag = psi_n * abs(eq.psi_bndry - eq.psi_axis)
    phi_1d = cumulative_trapezoid(2 * np.pi * q_1d, psi_mag, initial=0.0)

    b0 = ids_out.vacuum_toroidal_field.b0[0]
    rho_tor = np.sqrt(phi_1d / (np.pi * b0))
    rho_tor_norm = np.sqrt(phi_1d / phi_1d[-1])
    # d(rho_tor)/d(psi_mag) off axis (singular on axis)
    drho_dpsi_mag = q_1d[1:] / (b0 * rho_tor[1:])
    avg_Bp2 = assemble(0.0, s_fill**2 * geom["avg_Bp2"][0], geom["avg_Bp2"])
    avg_R2_Bp2 = assemble(0.0, s_fill**2 * geom["avg_R2_Bp2"][0], geom["avg_R2_Bp2"])
    avg_R_Bp = assemble(0.0, s_fill * geom["avg_R_Bp"][0], geom["avg_R_Bp"])
    # On axis: <1/R^n> -> 1/R_axis^n, |grad(rho_tor)| -> 1 and B -> Btor(axis).
    gm1 = even(1 / mag_r**2, "avg_inv_R2")  # <1/R^2>
    gm2 = np.concatenate(
        ([1 / mag_r**2], drho_dpsi_mag**2 * avg_Bp2[1:])
    )  # <|grad(rho_tor)|^2/R^2>
    gm3 = np.concatenate(
        ([1.0], drho_dpsi_mag**2 * avg_R2_Bp2[1:])
    )  # <|grad(rho_tor)|^2>
    gm4 = even(1 / btor_axis**2, "avg_inv_B2")  # <1/B^2>
    gm5 = even(btor_axis**2, "avg_B2")  # <B^2>
    gm7 = np.concatenate(([1.0], drho_dpsi_mag * avg_R_Bp[1:]))  # <|grad(rho_tor)|>
    gm9 = even(1 / mag_r, "avg_inv_R")  # <1/R>

    time_slice.profiles_1d.gm1 = gm1
    time_slice.profiles_1d.gm2 = gm2
    time_slice.profiles_1d.gm3 = gm3
    time_slice.profiles_1d.gm4 = gm4
    time_slice.profiles_1d.gm5 = gm5
    time_slice.profiles_1d.gm7 = gm7
    time_slice.profiles_1d.gm9 = gm9

    time_slice.profiles_1d.phi = phi_1d
    time_slice.profiles_1d.rho_tor = rho_tor
    time_slice.profiles_1d.rho_tor_norm = rho_tor_norm
    time_slice.profiles_1d.r_inboard = assemble(
        mag_r, mag_r + s_fill * (geom["r_inboard"][0] - mag_r), geom["r_inboard"]
    )
    time_slice.profiles_1d.r_outboard = assemble(
        mag_r, mag_r + s_fill * (geom["r_outboard"][0] - mag_r), geom["r_outboard"]
    )
    time_slice.profiles_1d.volume = assemble(
        0.0, s_fill**2 * geom["volume"][0], geom["volume"]
    )
    for key in ["elongation", "triangularity_upper", "triangularity_lower"]:
        setattr(
            time_slice.profiles_1d,
            key,
            assemble(geom[key][0], np.full(len(s_fill), geom[key][0]), geom[key]),
        )

    # 2D fields (total, plasma, and tokamak flux - jtor also stored)
    tokamak_psi = eq.tokamak.getPsitokamak(eq._vgreen)
    two_d_fields = [
        (0, "total", _PSI_IDS_OVER_PSI_FREEGSNKE * eq.psi(), profiles.jtor),
        (4, "plasma", _PSI_IDS_OVER_PSI_FREEGSNKE * eq.plasma_psi, None),
        (1, "vacuum", _PSI_IDS_OVER_PSI_FREEGSNKE * tokamak_psi, None),
    ]

    time_slice.profiles_2d.resize(len(two_d_fields))
    for i, (type_index, type_name, psi_2d, j_phi_2d) in enumerate(two_d_fields):
        profiles_2d = time_slice.profiles_2d[i]
        profiles_2d.type.index = type_index
        profiles_2d.type.name = type_name
        profiles_2d.grid_type.name = "rectangular"
        profiles_2d.grid_type.index = 1
        profiles_2d.grid_type.description = "Rectangular grid with dims (R, Z)."
        profiles_2d.grid.dim1 = eq.R_1D
        profiles_2d.grid.dim2 = eq.Z_1D
        profiles_2d.psi = psi_2d
        if j_phi_2d is not None:
            profiles_2d.j_phi = j_phi_2d

    return ids_out


def save_equilibrium_ids(ids, path):
    """
    Writes an `equilibrium` IDS to a netCDF file.

    Parameters
    ----------
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS to save (e.g. as returned by `write_equilibrium_to_ids`).
    path : str
        Destination netCDF file path (should end in `.nc`).
    """

    with imas.DBEntry(path, "w") as db_entry:
        db_entry.put(ids)


def load_equilibrium_ids(path):
    """
    Reads an `equilibrium` IDS back from a netCDF file.

    Parameters
    ----------
    path : str
        Path to a netCDF file previously written by `save_equilibrium_ids`.

    Returns
    -------
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS loaded from file.
    """

    with imas.DBEntry(path, "r") as db_entry:
        return db_entry.get("equilibrium")


def edge_taper(psi_n, width):
    """
    Smooth factor going from 1 to 0 over the outermost `width` of normalised
    flux (a cubic smoothstep in (1 - psi_n) / width), used to bring profiles
    that do not vanish at the separatrix smoothly to zero there.

    Parameters
    ----------
    psi_n : np.array
        Normalised poloidal flux values.
    width : float
        Width of the layer in normalised flux; 0 disables the taper.

    Returns
    -------
    np.array
        The taper factor at each value of `psi_n`.
    """
    psi_n = np.asarray(psi_n, dtype=float)
    if width <= 0.0:
        return np.ones_like(psi_n)
    s = np.clip((1.0 - psi_n) / width, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _strictly_increasing(psi_n):
    """
    Returns a copy of `psi_n` with any non-increasing steps removed (a strictly
    increasing grid is required by the spline interpolators in the profile
    classes). Repeated or decreasing values, which can arise from round-off
    near the axis/boundary, are nudged by a tiny amount.
    """
    psi_n = np.array(psi_n, dtype=float)
    for i in range(1, len(psi_n)):
        if psi_n[i] <= psi_n[i - 1]:
            psi_n[i] = np.nextafter(psi_n[i - 1], np.inf)
    return psi_n


def read_profiles_from_equilibrium_ids(ids, slice_index=0, edge_taper_width=0.0):
    """
    Extracts the p' and FF' profiles (and the associated scalars) from an IMAS
    `equilibrium` IDS in FreeGSNKE's conventions, ready to be used with a
    `GeneralPprimeFFprime` profile object.

    The IDS is assumed to follow the IMAS data dictionary convention (COCOS 17,
    psi in Wb) as written by `write_equilibrium_to_ids` or by other codes such as
    TORAX. FreeGSNKE uses psi in Wb/rad with the flux decreasing from the
    magnetic axis to the boundary for a positive plasma current. If the IDS flux
    increases outwards (which in COCOS 17 corresponds to a negative plasma
    current) the sign of the derivatives is flipped so that the returned profiles
    describe the same plasma with a positive current in FreeGSNKE's convention.

    Parameters
    ----------
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS.
    slice_index : int
        Index of the time slice to read.
    edge_taper_width : float
        If positive, p' and FF' are brought smoothly to zero over the
        outermost `edge_taper_width` of normalised flux (see `edge_taper`).
        Profiles from transport codes are generally finite at the separatrix,
        which makes the current density jump across the last closed flux
        surface; FreeGSNKE's static solver cannot resolve such a jump on its
        grid and may stall, whereas its own profile models vanish there. The
        total plasma current is unaffected when the profile object
        renormalises to `Ip`.

    Returns
    -------
    dict
        `psi_n` : strictly increasing normalised poloidal flux grid.
        `pprime` : dp/dpsi at `psi_n` [Pa/(Wb/rad)].
        `ffprime` : F dF/dpsi at `psi_n` [T^2 m^2/(Wb/rad)].
        `Ip` : plasma current magnitude [A].
        `fvac` : vacuum toroidal field function magnitude |R*Btor| [T m].
        `psi_axis`, `psi_bndry` : poloidal flux on axis/boundary [Wb/rad].
        `time` : time of the slice [s].
    """

    time_slice = ids.time_slice[slice_index]
    profiles_1d = time_slice.profiles_1d

    psi = np.asarray(profiles_1d.psi, dtype=float)
    if time_slice.global_quantities.psi_axis.has_value:
        psi_axis = float(time_slice.global_quantities.psi_axis)
    else:
        psi_axis = psi[0]
    if time_slice.global_quantities.psi_boundary.has_value:
        psi_bndry = float(time_slice.global_quantities.psi_boundary)
    else:
        psi_bndry = psi[-1]

    if profiles_1d.psi_norm.has_value:
        psi_n = np.asarray(profiles_1d.psi_norm, dtype=float)
    else:
        psi_n = (psi - psi_axis) / (psi_bndry - psi_axis)

    pprime = np.asarray(profiles_1d.dpressure_dpsi, dtype=float)
    ffprime = np.asarray(profiles_1d.f_df_dpsi, dtype=float)
    if pprime.size == 0 or ffprime.size == 0:
        raise ValueError(
            "The equilibrium IDS must contain profiles_1d.dpressure_dpsi and "
            "profiles_1d.f_df_dpsi."
        )
    if not (
        np.all(np.isfinite(psi_n))
        and np.all(np.isfinite(pprime))
        and np.all(np.isfinite(ffprime))
    ):
        raise ValueError(
            "The equilibrium IDS profiles (psi_norm, dpressure_dpsi, f_df_dpsi) "
            "contain non-finite values."
        )

    # COCOS 17 -> FreeGSNKE (Wb -> Wb/rad, and flux decreasing outwards)
    sign = -1.0 if psi_bndry > psi_axis else 1.0
    scale = sign * _PSI_IDS_OVER_PSI_FREEGSNKE

    Ip = abs(float(time_slice.global_quantities.ip))
    b0 = np.asarray(ids.vacuum_toroidal_field.b0, dtype=float)
    b0 = b0[slice_index] if b0.size > 1 else b0[0]
    fvac = abs(b0 * float(ids.vacuum_toroidal_field.r0))

    taper = edge_taper(psi_n, edge_taper_width)

    return {
        "psi_n": _strictly_increasing(psi_n),
        "pprime": scale * pprime * taper,
        "ffprime": scale * ffprime * taper,
        "Ip": Ip,
        "fvac": fvac,
        "psi_axis": sign * psi_axis / _PSI_IDS_OVER_PSI_FREEGSNKE,
        "psi_bndry": sign * psi_bndry / _PSI_IDS_OVER_PSI_FREEGSNKE,
        "time": (
            float(time_slice.time)
            if time_slice.time.has_value
            else float(np.asarray(ids.time, dtype=float)[slice_index])
        ),
    }


def profiles_from_equilibrium_ids(
    eq,
    ids,
    slice_index=0,
    Ip=None,
    fvac=None,
    Raxis=1.0,
    Ip_logic=True,
    interpolator="univariate_spline",
    edge_taper_width=0.0,
):
    """
    Builds a `GeneralPprimeFFprime` profile object from the p' and FF' profiles
    stored in an IMAS `equilibrium` IDS (see `read_profiles_from_equilibrium_ids`
    for the conventions used).

    Parameters
    ----------
    eq : freegsnke.equilibrium_update.Equilibrium
        Equilibrium object defining the grid and limiter.
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS.
    slice_index : int
        Index of the time slice to read.
    Ip : float, optional
        Plasma current [A]. Defaults to the magnitude of the IDS value.
    fvac : float, optional
        Vacuum field function R*Btor [T m]. Defaults to the magnitude of the IDS
        value (b0*r0).
    Raxis : float
        Radial scaling parameter passed to `GeneralPprimeFFprime`.
    Ip_logic : bool
        If True, the current density is renormalised to match `Ip` exactly.
    interpolator : str
        Interpolator passed to `GeneralPprimeFFprime`.
    edge_taper_width : float
        See `read_profiles_from_equilibrium_ids`.

    Returns
    -------
    freegsnke.jtor_update.GeneralPprimeFFprime
        The profile object.
    """

    # imported here to avoid a circular import at module load time
    from .jtor_update import GeneralPprimeFFprime

    data = read_profiles_from_equilibrium_ids(
        ids, slice_index=slice_index, edge_taper_width=edge_taper_width
    )
    return GeneralPprimeFFprime(
        eq=eq,
        Ip=data["Ip"] if Ip is None else Ip,
        fvac=data["fvac"] if fvac is None else fvac,
        psi_n=data["psi_n"],
        pprime_data=data["pprime"],
        ffprime_data=data["ffprime"],
        Raxis=Raxis,
        Ip_logic=Ip_logic,
        interpolator=interpolator,
    )


def update_profiles_from_equilibrium_ids(
    profiles, ids, slice_index=0, Ip=None, edge_taper_width=0.0
):
    """
    Updates an existing `GeneralPprimeFFprime` profile object in place with the
    p' and FF' profiles stored in an IMAS `equilibrium` IDS (see
    `read_profiles_from_equilibrium_ids` for the conventions used). This avoids
    rebuilding the grid-dependent state of the profile object when the profiles
    are exchanged repeatedly, e.g. when coupling to a transport code.

    Parameters
    ----------
    profiles : freegsnke.jtor_update.GeneralPprimeFFprime
        The profile object to update.
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS.
    slice_index : int
        Index of the time slice to read.
    Ip : float, optional
        Plasma current [A]. Defaults to the magnitude of the IDS value.
    edge_taper_width : float
        See `read_profiles_from_equilibrium_ids`.

    Returns
    -------
    freegsnke.jtor_update.GeneralPprimeFFprime
        The (same) updated profile object.
    """

    data = read_profiles_from_equilibrium_ids(
        ids, slice_index=slice_index, edge_taper_width=edge_taper_width
    )
    profiles.psi_n = data["psi_n"]
    profiles.pprime_data = data["pprime"]
    profiles.ffprime_data = data["ffprime"]
    profiles.p_data = None
    profiles.f_data = None
    profiles.Ip = data["Ip"] if Ip is None else Ip
    profiles.initialize_profile()
    return profiles
