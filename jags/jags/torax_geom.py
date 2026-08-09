"""Assemble TORAX's ``StandardGeometryIntermediates`` from a jags equilibrium.

TORAX consumes geometry as a bundle of 1D profiles against normalised toroidal
flux (``torax/_src/geometry/standard_geometry.py:166-224``). Its EQDSK path
builds that bundle by tracing contours -- ``contourpy`` for the curve, a bicubic
spline for ``grad psi``, ``np.gradient`` along the traced points for ``dl``
(``torax/_src/geometry/eqdsk.py:330-430``). This module builds the same bundle
from ``fsa.FluxSurfaces``, so the whole thing is a differentiable function of
the psi grid and can sit inside a coupled residual.

Almost everything is a rearrangement of quantities the co-area formula already
gives. Only three additions are needed:

* ``Ip_profile``, the current enclosed by a surface. Ampere's law gives
  ``mu0 Ip = contour_int B_p dl``, and ``B_p = |grad psi| / R``, so with the
  co-area weight ``R dl / |grad psi|`` that is
  ``<|grad psi|^2 / R^2> * int_dl_over_Bp / mu0`` -- no new integral.
* ``Phi``, the enclosed toroidal flux, from ``dPhi/dpsi = q`` by cumulative
  trapezoid, exactly as TORAX does it. This is the one quantity built by
  quadrature *across* surfaces rather than on them, so it inherits the level
  spacing as well as the per-surface error.
* ``<B^2>`` and ``<1/B^2>``, which need ``F`` as well as the geometry:
  ``B^2 = (|grad psi|^2 + F^2) / R^2``. ``<B^2>`` follows from averages already
  in the bundle; ``<1/B^2>`` does not, so it takes a second pass and the
  averager is required rather than just the bundle.

``F`` itself is *not* derived from the geometry. In a coupled solve it is the
profile function the equilibrium was posed with, known in closed form, so it
enters exactly rather than being read back off a contour.

Conventions: jags carries FreeGS4E's psi in Wb/rad, TORAX COCOS 11 in Wb, so
``psi_torax = 2 pi psi_jags`` and every quantity naming ``grad psi`` picks up
``2 pi`` (squared where it appears squared). ``B_p`` itself is convention-free,
so ``int_dl_over_Bp``, the ``<1/R^n>`` and ``<B^2>`` are unaffected. The
conversion happens here, at the boundary, so the rest of jags never sees it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .fsa import FluxSurfaces

MU0 = 4.0e-7 * jnp.pi
TWO_PI = 2.0 * jnp.pi


class ToraxIntermediates(NamedTuple):
    """The subset of ``StandardGeometryIntermediates`` that is geometry.

    Field names and units match TORAX exactly (COCOS 11, psi in Wb) so this can
    be passed straight across. Omitted from TORAX's dataclass, deliberately:

    * ``geometry_type``, ``Ip_from_parameters``, ``face_centers``,
      ``hires_factor`` -- configuration, not geometry.
    * ``connection_length_target``, ``connection_length_divertor``,
      ``angle_of_incidence_target``, ``R_OMP``, ``R_target``, ``B_pol_OMP``,
      ``diverted`` -- scrape-off-layer quantities. TORAX's own EQDSK path sets
      every one of them to ``None``, so there is nothing to match.
    """

    # scalars
    R_major: jnp.ndarray  # (R_out + R_in)/2 at the LCFS [m]
    a_minor: jnp.ndarray  # (R_out - R_in)/2 at the LCFS [m]
    B_0: jnp.ndarray  # vacuum toroidal field at R_major [T]
    z_magnetic_axis: jnp.ndarray  # [m]
    # profiles
    psi: jnp.ndarray  # poloidal flux [Wb]
    Ip_profile: jnp.ndarray  # enclosed plasma current [A]
    Phi: jnp.ndarray  # enclosed toroidal flux [Wb]
    R_in: jnp.ndarray  # [m]
    R_out: jnp.ndarray  # [m]
    F: jnp.ndarray  # R B_phi [m T]
    int_dl_over_Bp: jnp.ndarray  # [m/T]
    flux_surf_avg_1_over_R: jnp.ndarray  # [1/m]
    flux_surf_avg_1_over_R2: jnp.ndarray  # [1/m^2]
    flux_surf_avg_grad_psi: jnp.ndarray  # [m T]
    flux_surf_avg_grad_psi2: jnp.ndarray  # [m^2 T^2]
    flux_surf_avg_grad_psi2_over_R2: jnp.ndarray  # [T^2]
    flux_surf_avg_B2: jnp.ndarray  # [T^2]
    flux_surf_avg_1_over_B2: jnp.ndarray  # [1/T^2]
    delta_upper_face: jnp.ndarray  # upper triangularity
    delta_lower_face: jnp.ndarray  # lower triangularity
    elongation: jnp.ndarray
    vpr: jnp.ndarray  # dV/d(rho_norm) [m^3]
    # jags diagnostics, no TORAX counterpart
    n_eff: jnp.ndarray  # cells carrying each surface
    volume: jnp.ndarray  # [m^3]


def safety_factor(fs: FluxSurfaces, F):
    """q = (F / 2 pi) * contour_int dl / (R^2 B_p). Convention-free."""
    return F / TWO_PI * fs.avg_1_over_R2 * fs.int_dl_over_Bp


def intermediates(
    fs: FluxSurfaces,
    F,
    psi_axis,
    z_magnetic_axis,
    avg_1_over_B2=None,
) -> ToraxIntermediates:
    """Map a ``FluxSurfaces`` bundle plus ``F(psi)`` into TORAX's names.

    Parameters
    ----------
    fs : surfaces on levels ordered from the axis outwards.
    F : ``R B_phi`` on the same levels [m T].
    psi_axis : flux on the magnetic axis [Wb/rad]. TORAX puts psi = 0 there and
        has it grow outwards, so this fixes both the origin and the sign.
    z_magnetic_axis : height of the magnetic axis [m], from ``critical.py``.
    avg_1_over_B2 : ``<1/B^2>`` if it has been computed; it needs a second pass
        through the averager because ``1/B^2`` mixes ``F`` with the geometry.
        Left ``None`` it is filled with NaN rather than a plausible-looking
        wrong value.
    """
    a_minor = 0.5 * (fs.R_out[-1] - fs.R_in[-1])
    R_major = 0.5 * (fs.R_out[-1] + fs.R_in[-1])

    # Local shape, per surface, in TORAX's definitions.
    a_local = 0.5 * (fs.R_out - fs.R_in)
    R_local = 0.5 * (fs.R_out + fs.R_in)
    elongation = (fs.Z_upper - fs.Z_lower) / (2.0 * a_local)

    q = safety_factor(fs, F)
    # dPhi/dpsi = q, integrated outwards from the axis by trapezoid, as in
    # TORAX. The axis carries no surface, so q there is linearly extrapolated
    # from the innermost two levels; omitting that segment entirely leaves Phi
    # short by the whole core and rho_norm wrong everywhere.
    psi_t = (psi_axis - fs.psi_levels) * TWO_PI
    dq = (q[1] - q[0]) / (psi_t[1] - psi_t[0])
    q_axis = q[0] - dq * psi_t[0]
    q_full = jnp.concatenate([q_axis[None], q])
    psi_full = jnp.concatenate([jnp.zeros((1,)), psi_t])
    Phi = jnp.cumsum(
        jnp.concatenate([
            jnp.zeros((1,)),
            0.5 * (q_full[1:] + q_full[:-1]) * jnp.diff(psi_full),
        ])
    )[1:]
    rho_norm = jnp.sqrt(Phi / Phi[-1])

    # vpr = dV/d(rho_norm) = (dV/dpsi) / (drho_norm/dpsi), and
    # drho_norm/dpsi = q / (2 rho_norm Phi[-1]) by the definition above.
    # Written this way rather than by differencing V, so it stays exact.
    drho_dpsi = jnp.where(rho_norm > 0, q / (2.0 * rho_norm * Phi[-1]), jnp.inf)
    vpr = fs.dV_dpsi / TWO_PI / drho_dpsi

    return ToraxIntermediates(
        R_major=R_major,
        a_minor=a_minor,
        B_0=F[-1] / R_major,
        z_magnetic_axis=jnp.asarray(z_magnetic_axis),
        psi=psi_t,
        Ip_profile=fs.avg_grad_psi2_over_R2 * fs.int_dl_over_Bp / MU0,
        Phi=Phi,
        R_in=fs.R_in,
        R_out=fs.R_out,
        F=F,
        int_dl_over_Bp=fs.int_dl_over_Bp,
        flux_surf_avg_1_over_R=fs.avg_1_over_R,
        flux_surf_avg_1_over_R2=fs.avg_1_over_R2,
        flux_surf_avg_grad_psi=fs.avg_grad_psi * TWO_PI,
        flux_surf_avg_grad_psi2=fs.avg_grad_psi2 * TWO_PI**2,
        flux_surf_avg_grad_psi2_over_R2=fs.avg_grad_psi2_over_R2 * TWO_PI**2,
        flux_surf_avg_B2=fs.avg_grad_psi2_over_R2 + F**2 * fs.avg_1_over_R2,
        flux_surf_avg_1_over_B2=(
            jnp.full(F.shape, jnp.nan) if avg_1_over_B2 is None else avg_1_over_B2
        ),
        delta_upper_face=(R_local - fs.R_at_Z_upper) / a_local,
        delta_lower_face=(R_local - fs.R_at_Z_lower) / a_local,
        elongation=elongation,
        vpr=vpr,
        n_eff=fs.n_eff,
        volume=fs.volume,
    )


def avg_1_over_B2(average, psi, psi_levels, psi_axis, psi_edge, grad_psi, F,
                  R, **kw):
    """``<1/B^2>``, which needs ``F`` alongside the geometry.

    ``B^2 = (|grad psi|^2 + F^2) / R^2`` with psi in Wb/rad, so the average is
    over a field that differs level by level -- hence a separate call rather
    than a member of the bundle.
    """
    gR, gZ = grad_psi(psi)
    grad2 = gR**2 + gZ**2
    out = []
    for i in range(F.shape[0]):
        field = R**2 / (grad2 + F[i] ** 2)
        out.append(
            average(psi, field, psi_levels[i : i + 1], psi_axis, psi_edge, **kw)[0]
        )
    return jnp.stack(out)
