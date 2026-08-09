"""Solve a reference equilibrium in FreeGSNKE and dump it for cross-checking.

Runs in the FreeGSNKE virtualenv (numpy<2), not the jags one. Writes a .npz
that ``compare_freegsnke.py`` reads back in the JAX environment.

The vacuum flux ``tokamak_psi`` is dumped directly rather than the coil
geometry. FreeGSNKE's machine uses multi-filament and shaped coils, and
reproducing its Green's functions exactly would test the wrong thing: the point
of the comparison is the *plasma* solve, so the vacuum field is taken as given
and identical on both sides.

Usage:
    <freegsnke-venv>/bin/python scripts/dump_freegsnke_case.py out.npz [limited|diverted] [n]

The two coil-current sets ship with FreeGSNKE as
``examples/data/simple_{limited,diverted}_currents_PaxisIp.pk``. Note that those
names describe the configuration they produce with the ConstrainPaxisIp profile
they were generated for; with a different profile the resulting shape, and hence
whether the plasma is limited or diverted, may differ.
"""

import pathlib
import pickle
import sys

import numpy as np

# This project lives at <freegsnke>/jags, so the repo root is two levels up.
FREEGSNKE = pathlib.Path(__file__).resolve().parents[2]
MACHINE = f"{FREEGSNKE}/machine_configs/MAST-U"

# Profile: Lao85 polynomial pprime/ffprime. Chosen because it exists in both
# codes and has an analytic antiderivative, so p(psi) and F(psi) can be written
# in closed form on the jags side (see profiles.lao85).
IP = 6.2e5
FVAC = 0.5
ALPHA = [2.0, -1.0]
BETA = [1.0, -0.5]
RAXIS = 0.9


def main(out_path, case="limited", n=65):
    from freegsnke import build_machine, equilibrium_update, GSstaticsolver
    from freegsnke.jtor_update import Lao85

    tokamak = build_machine.tokamak(
        active_coils_path=f"{MACHINE}/MAST-U_like_active_coils.pickle",
        passive_coils_path=f"{MACHINE}/MAST-U_like_passive_coils.pickle",
        limiter_path=f"{MACHINE}/MAST-U_like_limiter.pickle",
        wall_path=f"{MACHINE}/MAST-U_like_wall.pickle",
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1, Rmax=2.0,
        Zmin=-2.2, Zmax=2.2,
        nx=n, ny=n,
        psi=None,
    )

    with open(f"{FREEGSNKE}/examples/data/simple_{case}_currents_PaxisIp.pk", "rb") as f:
        currents = pickle.load(f)
    for name, value in currents.items():
        eq.tokamak.set_coil_current(coil_label=name, current_value=value)

    profiles = Lao85(
        eq=eq, Ip=IP, fvac=FVAC, alpha=ALPHA, beta=BETA,
        alpha_logic=True, beta_logic=True, Raxis=RAXIS, Ip_logic=True,
    )

    solver = GSstaticsolver.NKGSsolver(eq)
    solver.solve(
        eq=eq, profiles=profiles, constrain=None,
        target_relative_tolerance=1e-9, verbose=False,
    )

    tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
    limiter = eq.tokamak.limiter

    # Contour-traced flux-surface quantities, the reference for jags/fsa.py.
    # FreeGS4E's q is (1/2pi) * contour_int F dl / (R^2 B_p), evaluated on 128
    # points traced by ray-casting from the O-point (freegs4e/critical.py:1066).
    # fpol is dumped alongside so the jags side uses an identical F and the
    # comparison isolates the geometry.
    # find_safety normalises by the X-point flux and raises without one, so a
    # genuinely limited equilibrium yields no q profile.
    psinorm = np.linspace(0.05, 0.95, 19)
    try:
        q = np.asarray(eq.q(psinorm))
    except ValueError as exc:
        print(f"  no q profile: {exc}")
        q = np.full_like(psinorm, np.nan)
    fpol = np.asarray(eq.fpol(psinorm))

    np.savez_compressed(
        out_path,
        Rmin=eq.R[0, 0], Rmax=eq.R[-1, 0],
        Zmin=eq.Z[0, 0], Zmax=eq.Z[0, -1],
        nR=eq.R.shape[0], nZ=eq.R.shape[1],
        limiter_R=np.asarray(limiter.R), limiter_Z=np.asarray(limiter.Z),
        tokamak_psi=tokamak_psi,
        plasma_psi=eq.plasma_psi,
        psi=eq.psi(),
        jtor=profiles.jtor,
        psi_axis=eq.psi_axis,
        psi_bndry=eq.psi_bndry,
        flag_limiter=eq.flag_limiter,
        opt=eq.opt, xpt=eq.xpt,
        # profile definition, including the coefficients Lao85 appends internally
        alpha_full=profiles.alpha, beta_full=profiles.beta,
        L=profiles.L, Ip=IP, fvac=FVAC, Raxis=RAXIS,
        # flux-surface reference
        psinorm=psinorm, q=q, fpol=fpol,
        plasma_volume=eq.plasmaVolume(),
    )

    # A geqdsk of the same equilibrium, so TORAX's own eqdsk parser can be run
    # on it (see check_fsa_torax.py). FreeGS4E writes psi in Wb/rad with the
    # axis shifted to zero, which is COCOS 1; TORAX converts and then enforces
    # its own sign convention.
    from freegs4e import geqdsk

    # FreeGSNKE's Lao85 returns pressure with a leading singleton axis, which
    # the fixed-width g-file writer cannot format. Everything else is already
    # 1D, so flatten just this one on the way out.
    eq.pressure = lambda pn, _p=eq.pressure: np.ravel(_p(pn))

    gfile = str(out_path).replace(".npz", ".geqdsk")
    with open(gfile, "w") as fh:
        geqdsk.write(eq, fh, label=case[:8])
    print(f"wrote {gfile}")

    total_ip = profiles.jtor.sum() * (eq.R[1, 0] - eq.R[0, 0]) * (eq.Z[0, 1] - eq.Z[0, 0])
    print(f"wrote {out_path}  (case: {case})")
    print(f"  limiter configuration : {bool(eq.flag_limiter)}")
    print(f"  psi_axis  = {eq.psi_axis:.6f}")
    print(f"  psi_bndry = {eq.psi_bndry:.6f}")
    print(f"  Ip        = {total_ip:.6e}  (target {IP:.6e})")
    print(f"  alpha_full = {profiles.alpha}")
    print(f"  beta_full  = {profiles.beta}")
    print(f"  L          = {profiles.L:.6e}")
    print(f"  volume     = {eq.plasmaVolume():.6f} m^3")
    print(f"  q(0.05..0.95) = {np.array2string(q, precision=4)}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "freegsnke_case.npz",
        sys.argv[2] if len(sys.argv) > 2 else "limited",
        int(sys.argv[3]) if len(sys.argv) > 3 else 65,
    )
