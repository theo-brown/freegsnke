"""Inverse-solve the ITER hybrid shape in FreeGSNKE, then forward-solve it.

Stage 1 of the ITER comparison. Runs in the FreeGSNKE venv.

The shape target comes from TORAX's own ``iterhybrid_cocos11.eqdsk`` via
``iter_target.py``: 24 isoflux points spread evenly in poloidal angle around the
separatrix, plus the X-point and the magnetic axis as nulls. ``Ip`` and
``fvac = R B_phi`` are taken from the same file, so the plasma being asked for is
the one TORAX runs on.

The profile is ``ConstrainBetapIp``, as in FreeGSNKE's own ITER example.
``Lao85`` -- used for the MAST-U cross-check -- is unusable here: its
``FF' ~ Raxis``, ``p' ~ 1/Raxis`` scaling makes the FF' term dominate at ITER's
major radius, and the inverse solve collapses the plasma to 8 m^3. jags matches
it with ``profiles.topeol``, which is the same two-term form written as p(psi)
and F(psi), so stage 2 compares the solvers rather than two different current
profiles.

Two settings differ from FreeGSNKE's example and both matter:

* ``Raxis`` is left at its default of 1.0. Despite the name it is a *shape*
  parameter -- it sets the relative weight of the ``Beta0 R / Raxis`` and
  ``(1 - Beta0) Raxis / R`` terms -- so putting ITER's actual 6.2 there
  rebalances the current profile by a factor of 38 and collapses the solve.
* ``l2_reg`` is 1e-8, not the example's 1e-14, because at 1e-14 nothing in this
  environment converges. Pass ``--sweep`` to see it.

The target arrives already shifted into machine coordinates by
``iter_target.py``. This matters more than anything else here. The CHEASE file
is plasma-centred -- its axis is at Z = -0.0000 and its "limiter" is the grid
bounding box -- while the FreeGSNKE ITER machine has a real wall running from
Z = -4.574 to +4.720. Used unshifted, the requested separatrix has its X-point
down inside the divertor cassette and its lower outboard flank *outside* the
limiter, which is not a boundary any plasma can have. The solve then builds
something in the upper half of the vessel instead. Every "wrong shape" result
before that shift was this, and no amount of retuning ``l2_reg`` addressed it.

The solve is mildly **nondeterministic** at 129^2: when successive residuals come
out collinear, ``GSstaticsolver`` restarts the Krylov space along a direction
built from ``np.random.random()`` (``freegsnke/GSstaticsolver.py:557-570``). Six
seeds gave the same 672 m^3 answer there and one earlier run collapsed to
5.6 m^3, so the collapse is rare rather than typical. At 65^2 the restart never
triggers and all six seeds are identical. The RNG is seeded and the seed
recorded either way, because a case that cannot be reproduced is not a reference.

The inverse solve finds coil currents. Those currents are then used for an
ordinary forward solve, dumped in the same format as ``dump_freegsnke_case.py``
so jags can be pointed at it unchanged.

The forward solve switches the profile to ``Fiesta_Topeol``, holding ``Beta0``
fixed at whatever the inverse solve converged to. This is not cosmetic.
``ConstrainBetapIp`` re-solves for ``Beta0`` at every forward iteration to hold
betap at the constrained Ip, and that inner solve is unstable here: it drives
``Beta0`` negative, which sends ``F^2 = fvac^2 + 2 mu0 L (1 - Beta0) Raxis I``
below zero and collapses the plasma to a few cells. With ``Beta0`` frozen the
same currents converge in 7 Newton steps to 7.2e-10 and land 2.3e-4 away from
the inverse solution, as they should -- the inverse solution *is* the forward
solution. Fixing ``Beta0`` also makes stage 2 a fair test, since jags' ``topeol``
is the same fixed two-term form with no inner constraint solve of its own.

Usage:
    <freegsnke-venv>/bin/python scripts/iter_inverse.py [target.npz] [out.npz] [--sweep]
"""

import pathlib
import sys

import numpy as np
from scipy.interpolate import RectBivariateSpline

FREEGSNKE = pathlib.Path(__file__).resolve().parents[2]
MACHINE = f"{FREEGSNKE}/machine_configs/ITER"

# Grid from FreeGSNKE's own ITER example. The target separatrix spans
# R 4.21-8.19, Z -3.91 to 3.56, so it sits inside with room for the legs.
# NGRID must be 2**n + 1 for FreeGSNKE, so 65, 129 or 257. 65 is the default
# because the jags forward solve in stage 2 stores a dense N x N inverse: 0.13 GB
# at 65^2 but 2.28 GB at 129^2, and JAX captures that as a lowering constant and
# copies it, which took 15.2 GB and killed the machine.
RMIN, RMAX, ZMIN, ZMAX = 3.2, 8.8, -5.0, 5.0
NGRID = 65

# Topeol shape parameters, from FreeGSNKE's ITER example. alpha_n must stay an
# integer for jags' topeol to have a closed-form antiderivative.
ALPHA_M, ALPHA_N, BETAP, PROFILE_RAXIS = 2.0, 1, 0.15, 1.0
L2_REG = 1e-8
SWEEP = (1e-9, 1e-8, 1e-7)

# Seeds tried in order; the first usable equilibrium is kept.
SEEDS = (0, 1, 2, 3, 4, 5)

# Two different questions, kept apart on purpose.
#
# Usable: is this a converged, diverted, ITER-scale equilibrium that a forward
# solve can reproduce? That is all stages 2 and 3 need, and it gates the run --
# a collapsed 5 m^3 plasma with gigaamp currents is simply wrong.
MIN_VOLUME, MAX_CURRENT = 300.0, 3e7
# Faithful: does it match the CHEASE separatrix this target came from? Reported
# and stored, never fatal, because with this machine description it is not
# achievable -- see the module docstring.
MAX_MEAN_ERR, MAX_SD = 0.05, 0.05


def main(target_path="scripts/iter_target.npz", out_path="scripts/case_iter.npz",
         sweep=False, ngrid=None):
    global NGRID
    NGRID = ngrid or NGRID
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update
    from freegsnke.inverse import Inverse_optimizer
    from freegsnke.jtor_update import ConstrainBetapIp, Fiesta_Topeol

    t = np.load(target_path)
    Ip, fvac = float(t["Ip"]), float(t["fvac"])
    Rb, Zb = np.asarray(t["lcfs_R"]), np.asarray(t["lcfs_Z"])

    def attempt(l2, seed):
        np.random.seed(seed)
        eq = equilibrium_update.Equilibrium(
            tokamak=build_machine.tokamak(
                active_coils_path=f"{MACHINE}/ITER_active_coils.pickle",
                passive_coils_path=f"{MACHINE}/ITER_passive_coils.pickle",
                limiter_path=f"{MACHINE}/ITER_limiter.pickle",
                wall_path=f"{MACHINE}/ITER_wall.pickle",
            ),
            Rmin=RMIN, Rmax=RMAX, Zmin=ZMIN, Zmax=ZMAX,
            nx=NGRID, ny=NGRID, psi=None,
        )
        profiles = ConstrainBetapIp(
            eq=eq, betap=BETAP, Ip=Ip, fvac=fvac,
            alpha_m=ALPHA_M, alpha_n=ALPHA_N, Raxis=PROFILE_RAXIS,
        )
        solver = GSstaticsolver.NKGSsolver(eq)
        constrain = Inverse_optimizer(
            null_points=[[float(t["Rx"]), float(t["R_axis"])],
                         [float(t["Zx"]), float(t["Z_axis"])]],
            isoflux_set=np.array([[list(np.asarray(t["isoflux_R"])),
                                   list(np.asarray(t["isoflux_Z"]))]]),
        )
        solver.inverse_solve(
            eq=eq, profiles=profiles, constrain=constrain,
            target_relative_tolerance=1e-4, target_relative_psit_update=1e-3,
            verbose=False, l2_reg=l2,
        )
        # How well the achieved boundary matches the target: normalised flux at
        # the target separatrix points, which should all be 1.
        #
        # Sampled from eq.psi() through a fresh spline rather than eq.psi_func.
        # psi_func lags the solve -- FreeGSNKE prints "Discrepancy between
        # psi_func and plasma_psi detected" and re-sets it -- and reading it here
        # gave a mean of 0.07 where the converged field gives 1.16, which is the
        # difference between rejecting a good solve and accepting a bad one.
        sp = RectBivariateSpline(
            np.linspace(RMIN, RMAX, NGRID), np.linspace(ZMIN, ZMAX, NGRID), eq.psi()
        )
        pn = (sp(Rb, Zb, grid=False) - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
        act = {k: v for k, v in eq.tokamak.getCurrents().items()
               if not str(k).startswith(("I", "O"))}
        xpt = np.asarray(eq.xpt)
        return eq, profiles, solver, float(np.std(pn)), float(np.mean(pn)), \
            max(abs(v) for v in act.values()), \
            (float(xpt[0][1]) if len(xpt) else float("nan"))

    def describe(e, sd, mean, maxI, xz):
        return (f"vol={e.plasmaVolume():>8.1f} m3  "
                f"diverted={str(not bool(e.flag_limiter)):<5}  "
                f"max|I_active|={maxI:.2e} A  X-pt Z={xz:>6.2f}  "
                f"psiN at target LCFS: mean {mean:.3f} sd {sd:.3f}")

    if sweep:
        print("l2_reg sweep (diagnostic only -- L2_REG is what gets used):")
        for l2 in SWEEP:
            e, _, _, sd, mean, maxI, xz = attempt(l2, SEEDS[0])
            print(f"  l2={l2:.0e}  {describe(e, sd, mean, maxI, xz)}")

    print(f"inverse solve, l2_reg = {L2_REG:.0e}, trying seeds until one gives a "
          f"usable equilibrium ...")
    best = None
    for seed in SEEDS:
        eq, profiles, solver, sd, mean, maxI, xz = attempt(L2_REG, seed)
        usable = (not bool(eq.flag_limiter) and eq.plasmaVolume() > MIN_VOLUME
                  and maxI < MAX_CURRENT)
        # ITER is a lower single null, so a positive X-point Z is by itself
        # proof the achieved shape is not the target's.
        faithful = abs(mean - 1.0) < MAX_MEAN_ERR and sd < MAX_SD and xz < 0.0
        print(f"  seed={seed}  {describe(eq, sd, mean, maxI, xz)}  "
              f"{'usable' if usable else 'UNUSABLE'}"
              f"{', matches target' if faithful else ''}")
        if usable or best is None:
            best = (seed, eq, profiles, solver, faithful, usable)
        if usable:
            break
    else:
        print("  no seed produced a usable equilibrium")
    seed, eq, profiles, solver, faithful, usable = best
    print(f"  using seed {seed}")

    if not usable:
        # Writing to the usual name would let stages 2 and 3 run on a collapsed
        # plasma and report differences that mean nothing.
        out_path = str(out_path).replace(".npz", "_rejected.npz")
        print(f"  writing to {out_path} instead, so nothing downstream picks it up")
    elif not faithful:
        print("  NOTE: this is a converged, diverted, ITER-scale equilibrium of "
              "the FreeGSNKE ITER\n        machine, but it is NOT the CHEASE "
              "separatrix -- see the module docstring.\n        Stages 2 and 3 "
              "compare solvers on it; they do not validate the shape.")

    currents = eq.tokamak.getCurrents()
    print("  active coil currents [A]:")
    for k, v in currents.items():
        if not str(k).startswith(("I", "O")):
            print(f"    {k:<10} {v:>14.4e}")

    # Forward solve with those currents, so the equilibrium being compared is
    # one both codes can reproduce from the same vacuum flux. Beta0 is frozen at
    # the inverse solve's value -- see the module docstring for why re-solving it
    # here destroys the plasma.
    Beta0 = float(profiles.Beta0)
    print(f"forward solve, Fiesta_Topeol with Beta0 = {Beta0:.6f} ...")
    np.random.seed(seed)  # the forward solver draws from the same RNG
    profiles = Fiesta_Topeol(
        eq=eq, Beta0=Beta0, Ip=Ip, fvac=fvac,
        alpha_m=ALPHA_M, alpha_n=ALPHA_N, Raxis=PROFILE_RAXIS,
    )
    solver.solve(eq=eq, profiles=profiles, constrain=None,
                 target_relative_tolerance=1e-9, verbose=False)

    tokamak_psi = eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen)
    limiter = eq.tokamak.limiter
    psinorm = np.linspace(0.05, 0.95, 19)
    try:
        q = np.asarray(eq.q(psinorm))
    except ValueError as exc:
        print(f"  no q profile: {exc}")
        q = np.full_like(psinorm, np.nan)

    from freegs4e import geqdsk

    eq.pressure = lambda pn, _p=eq.pressure: np.ravel(_p(pn))
    gfile = str(out_path).replace(".npz", ".geqdsk")
    with open(gfile, "w") as fh:
        geqdsk.write(eq, fh, label="iter")

    np.savez_compressed(
        out_path,
        Rmin=eq.R[0, 0], Rmax=eq.R[-1, 0], Zmin=eq.Z[0, 0], Zmax=eq.Z[0, -1],
        nR=eq.R.shape[0], nZ=eq.R.shape[1],
        limiter_R=np.asarray(limiter.R), limiter_Z=np.asarray(limiter.Z),
        tokamak_psi=tokamak_psi, plasma_psi=eq.plasma_psi, psi=eq.psi(),
        jtor=profiles.jtor, psi_axis=eq.psi_axis, psi_bndry=eq.psi_bndry,
        flag_limiter=eq.flag_limiter, opt=eq.opt, xpt=eq.xpt,
        L=profiles.L, Beta0=profiles.Beta0, Ip=Ip, fvac=fvac,
        alpha_m=ALPHA_M, alpha_n=ALPHA_N, profile_Raxis=PROFILE_RAXIS,
        betap=BETAP, l2_reg=L2_REG, seed=seed,
        matches_chease_target=faithful,
        psinorm=psinorm, q=q, fpol=np.asarray(eq.fpol(psinorm)),
        plasma_volume=eq.plasmaVolume(),
        coil_names=np.array(list(currents), dtype=object),
        coil_currents=np.array([currents[k] for k in currents]),
        target_lcfs_R=t["lcfs_R"], target_lcfs_Z=t["lcfs_Z"],
        isoflux_R=t["isoflux_R"], isoflux_Z=t["isoflux_Z"],
    )
    dA = (eq.R[1, 0] - eq.R[0, 0]) * (eq.Z[0, 1] - eq.Z[0, 0])
    print(f"wrote {out_path} and {gfile}")
    print(f"  limiter configuration : {bool(eq.flag_limiter)}")
    print(f"  psi_axis  = {eq.psi_axis:.6f}   psi_bndry = {eq.psi_bndry:.6f}")
    print(f"  Ip        = {profiles.jtor.sum() * dA:.6e}  (target {Ip:.6e})")
    print(f"  volume    = {eq.plasmaVolume():.4f} m^3")
    print(f"  q(0.05..0.95) = {np.array2string(q, precision=3)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else "scripts/iter_target.npz",
        args[1] if len(args) > 1 else "scripts/case_iter.npz",
        sweep="--sweep" in sys.argv,
        ngrid=next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--ngrid=")),
                   None),
    )
