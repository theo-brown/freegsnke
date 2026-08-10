"""Loose coupling: hand a jags equilibrium to TORAX as its geometry.

Loose coupling means the two codes take turns. jags solves Grad-Shafranov, its
geometry is frozen, TORAX evolves transport on it. Nothing is solved jointly, so
this is exactly what TORAX already supports for CHEASE, FBT and EQDSK inputs --
the only difference is that the geometry arrives as arrays in memory rather than
through a file, and that it came from a differentiable pipeline, which matters
only if you later want to tighten the coupling.

The handoff point is ``StandardGeometryIntermediates``. TORAX's
``build_standard_geometry`` turns that into the ``StandardGeometry`` its
transport equations actually read: the metric coefficients ``g0`` = <|grad V|>,
``g1`` = <(grad V)^2>, ``g2`` = <(grad V)^2 / R^2>, ``g3`` = <1/R^2>, plus
``vpr``, ``spr`` and the rho grid. Producing the intermediates is therefore the
whole job; ``torax_geom.py`` does the physics and this module does the plumbing.

Two pieces of plumbing are not physics and are easy to get wrong:

* **The axis row.** Every TORAX profile starts at the magnetic axis, which has
  no flux surface, so TORAX hard-codes limiting values there
  (``torax/_src/geometry/eqdsk.py:435-447``): zero for anything proportional to
  a contour length or ``|grad psi|``, ``1/R_axis^n`` for the inverse-radius
  averages, and the shape of the innermost resolved surface for elongation and
  triangularity. ``axis_row`` reproduces exactly those, so the difference
  between the two paths is never the convention at index 0.
* **Ordering and normalisation.** TORAX wants profiles from the axis outwards
  against normalised toroidal flux, and asserts internally that volume
  increases monotonically. jags produces surfaces on whatever psi levels it is
  given, so the levels are built here rather than left to the caller.

``import torax`` happens inside the functions, so the rest of jags stays
importable without it -- they are separate virtualenvs in the cross-check setup,
though TORAX's own venv happily runs jags too.
"""

from __future__ import annotations

import numpy as np

from . import torax_geom

# TORAX's own default for the outermost surface. A diverted separatrix has a
# divergent contour integral at the X-point, so both codes stop short of it.
LAST_SURFACE_FACTOR = 0.95


def axis_row(R_axis, F_axis):
    """Limiting values on the magnetic axis, matching TORAX's EQDSK path."""
    Btor_axis = F_axis / R_axis
    return dict(
        psi=0.0,
        Ip_profile=0.0,
        Phi=0.0,
        R_in=R_axis,
        R_out=R_axis,
        F=F_axis,
        int_dl_over_Bp=0.0,
        flux_surf_avg_1_over_R=1.0 / R_axis,
        flux_surf_avg_1_over_R2=1.0 / R_axis**2,
        flux_surf_avg_grad_psi=0.0,
        flux_surf_avg_grad_psi2=0.0,
        flux_surf_avg_grad_psi2_over_R2=0.0,
        flux_surf_avg_B2=Btor_axis**2,
        flux_surf_avg_1_over_B2=1.0 / Btor_axis**2,
        # shape is taken from the innermost resolved surface, as TORAX does
        delta_upper_face=None,
        delta_lower_face=None,
        elongation=None,
        vpr=0.0,
    )


def flux_levels(psi_axis, psi_edge, n_surfaces, last_surface_factor=LAST_SURFACE_FACTOR):
    """``n_surfaces - 1`` levels from just inside the axis to near the edge.

    Uniform in psi, as TORAX's EQDSK path is. The axis itself is excluded: it
    carries no surface, and ``axis_row`` supplies its values instead.
    """
    span = (psi_edge - psi_axis) * last_surface_factor
    return psi_axis + np.linspace(0.0, span, n_surfaces)[1:]


def build_intermediates(
    grid,
    averager,
    psi,
    F_of_psi,
    psi_axis,
    psi_edge,
    R_axis,
    Z_axis,
    n_surfaces=60,
    label=None,
    n_rho=25,
    hires_factor=4,
    last_surface_factor=LAST_SURFACE_FACTOR,
    Ip_from_parameters=False,
):
    """A TORAX ``StandardGeometryIntermediates`` from a jags psi field.

    Parameters
    ----------
    averager : the ``(average, surfaces, grad_psi)`` triple from
        ``fsa.make_flux_surface_averager``.
    F_of_psi : called on the flux levels. In a coupled solve this is the profile
        function the equilibrium was posed with, so ``F`` is exact rather than
        interpolated off a contour -- one of the few places this route is
        strictly better than reading a geqdsk.
    label : the reachability field, for a diverted equilibrium. Without it the
        outer surfaces pick up the divertor legs; see ``fsa.py``.
    Ip_from_parameters : whether TORAX should rescale psi so the plasma current
        matches the one in its own config rather than the one in this geometry.
        Left False so the handoff is faithful by default; set it to match the
        eqdsk path when the two are being compared.
    """
    import jax.numpy as jnp
    from torax._src.geometry import geometry, standard_geometry

    average, surfaces, grad_psi = averager
    levels = flux_levels(psi_axis, psi_edge, n_surfaces, last_surface_factor)
    levels_j = jnp.asarray(levels)
    F = jnp.asarray(np.asarray(F_of_psi(levels), dtype=float))

    fs = surfaces(psi, levels_j, psi_axis, psi_edge, label=label)
    one_over_B2 = torax_geom.avg_1_over_B2(
        average, psi, levels_j, psi_axis, psi_edge, grad_psi, F,
        jnp.asarray(grid.R), label=label,
    )
    ours = torax_geom.intermediates(fs, F, psi_axis, Z_axis, one_over_B2)

    axis = axis_row(R_axis, float(np.asarray(F_of_psi(np.array([psi_axis])))[0]))

    def stack(name):
        arr = np.asarray(getattr(ours, name), dtype=float)
        head = axis[name]
        if head is None:  # shape quantities: copy the innermost surface
            head = arr[0]
        return np.concatenate([[head], arr])

    fields = {k: stack(k) for k in axis}
    # Phi must start at zero on the axis and grow; torax_geom already integrates
    # from the axis, so shifting is not needed, only the prepended zero.
    return standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.EQDSK,
        Ip_from_parameters=Ip_from_parameters,
        R_major=np.asarray(ours.R_major, dtype=float),
        a_minor=np.asarray(ours.a_minor, dtype=float),
        B_0=np.asarray(ours.B_0, dtype=float),
        z_magnetic_axis=np.asarray(Z_axis, dtype=float),
        face_centers=np.linspace(0.0, 1.0, n_rho + 1),
        hires_factor=hires_factor,
        diverted=None,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
        **fields,
    )


def build_geometry(intermediates):
    """``StandardGeometryIntermediates`` -> the ``StandardGeometry`` TORAX runs on."""
    from torax._src.geometry import standard_geometry

    return standard_geometry.build_standard_geometry(intermediates)


def geometry_provider(geo):
    """A ``GeometryProvider``: the frozen geometry, for every time step.

    This *is* the loose coupling -- the geometry does not respond to what
    transport does to the pressure. A sequence of jags solves at different times
    would slot in here as a time-dependent provider instead.
    """
    from torax._src.geometry import geometry_provider as gp

    return gp.ConstantGeometryProvider(geo)
