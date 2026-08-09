"""Run TORAX's own eqdsk parser on a jags case and dump its flux-surface averages.

TORAX computes exactly the quantity list in ``jags.fsa.FluxSurfaces``, but by
contour tracing: ``contourpy`` for the curve, a bicubic spline for ``grad psi``,
and ``FSA(G) = int G dl/Bp / int dl/Bp`` (``torax/_src/geometry/eqdsk.py:330``).
That makes it the sharpest available reference -- an independent implementation
of the same definitions, sharing nothing with jags but the psi grid.

Runs in the **TORAX** venv, and writes an .npz that ``check_fsa.py`` reads back
in the jags venv.

Two conventions have to be tracked across the boundary:

* TORAX works in COCOS 11, where psi is in Wb and ``B_p = |grad psi| / (2 pi R)``.
  jags keeps FreeGS4E's Wb/rad, so ``psi_torax = 2 pi psi_jags``. The two 2 pi's
  cancel in ``B_p``, which is physical either way, so ``int_dl_over_Bp`` and the
  ``<1/R^n>`` averages compare directly. ``<|grad psi|>`` and
  ``<|grad psi|^2 ...>`` do **not**: they carry an explicit 2 pi and (2 pi)^2.
* TORAX flips the sign of psi so that it increases outward and puts the axis at
  zero, so ``psi_jags = psi_axis - psi_torax / (2 pi)``.

``q_rebuilt`` is emitted so the jags side can confirm all of that against
FreeGS4E's independently traced ``q``. Since B_p is convention-free,
``q = F <1/R^2> int_dl_over_Bp / 2 pi`` is the same expression on both sides,
and any factor left over in the conversion would show up immediately.

The COCOS defaults to 7: FreeGS4E writes psi in Wb/rad and, with Ip > 0, psi
*decreasing* outward, which is sigma_Bp = -1. The ``eqdsk`` library refuses
any other identification for these files, so this is not a free choice.

Usage:
    /home/user/torax/.venv/bin/python scripts/check_fsa_torax.py in.geqdsk out.npz [cocos]
"""

import os
import sys

import numpy as np

N_SURFACES = 60
# The outermost contour of a diverted equilibrium runs into the X-point, where
# the integrals diverge; TORAX's own remedy is to stop just short of it. On
# these MAST-U cases 0.99 is not short enough -- contourpy returns an open or
# doubled curve and TORAX's own monotonic-volume check rejects it -- so the
# reference has to stop at 0.95. This is the cost of tracing: there is no
# setting at which the traced reference reaches the separatrix here.
LAST_SURFACE_FACTOR = 0.95

FIELDS = [
    "psi", "F", "int_dl_over_Bp",
    "flux_surf_avg_1_over_R", "flux_surf_avg_1_over_R2",
    "flux_surf_avg_grad_psi", "flux_surf_avg_grad_psi2",
    "flux_surf_avg_grad_psi2_over_R2",
    "flux_surf_avg_B2", "flux_surf_avg_1_over_B2",
    "R_in", "R_out", "elongation", "Ip_profile", "Phi", "vpr",
    "delta_upper_face", "delta_lower_face",
    "R_major", "a_minor", "B_0", "z_magnetic_axis",
]


def main(gfile, out_path, cocos=7, last_surface_factor=LAST_SURFACE_FACTOR):
    from torax._src.geometry import eqdsk as torax_eqdsk

    intermediates = torax_eqdsk._construct_intermediates_from_eqdsk(
        geometry_directory=os.path.dirname(os.path.abspath(gfile)),
        geometry_file=os.path.basename(gfile),
        eqdsk_object=None,
        hires_factor=4,
        Ip_from_parameters=False,
        face_centers=np.linspace(0.0, 1.0, 26),
        n_surfaces=N_SURFACES,
        last_surface_factor=last_surface_factor,
        cocos=cocos,
    )

    g = {k: np.asarray(getattr(intermediates, k)) for k in FIELDS}
    g["q_rebuilt"] = (
        g["F"] * g["flux_surf_avg_1_over_R2"] * g["int_dl_over_Bp"] / (2 * np.pi)
    )
    g["n_surfaces"] = np.asarray(N_SURFACES)
    g["last_surface_factor"] = np.asarray(last_surface_factor)

    np.savez_compressed(out_path, **g)
    print(f"wrote {out_path}  ({N_SURFACES} surfaces from {gfile})")
    print(f"  psi range   : {g['psi'][0]:.6f} .. {g['psi'][-1]:.6f} Wb")
    print(f"  q_rebuilt   : {np.array2string(g['q_rebuilt'][1::12], precision=4)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 7,
         float(sys.argv[4]) if len(sys.argv) > 4 else LAST_SURFACE_FACTOR)
