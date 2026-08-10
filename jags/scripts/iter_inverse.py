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
* ``l2_reg`` is 1e-8, not the example's 1e-14. Unregularised, the optimiser
  finds coil currents of order a gigaamp and a plasma of a few cubic metres;
  at 1e-8 the peak active current is a few MA and the plasma is a diverted
  ~670 m^3. Pass ``--sweep`` to print the neighbouring decades rather than take
  that on trust; the value is fixed rather than auto-selected, because a score
  built from boundary flux alone prefers 1e-9, which has a visibly worse
  boundary and larger currents.

The inverse solve is **not deterministic**. When successive residuals come out
collinear, ``GSstaticsolver`` restarts the Krylov space along a direction built
from ``np.random.random()`` (``freegsnke/GSstaticsolver.py:557-570``), so the
same settings reach a diverted ~670 m^3 plasma on one run and collapse to ~5 m^3
on the next. That is not something to average over: a collapsed result is simply
wrong. So the RNG is seeded, each seed is checked against the target boundary,
and the first one that actually achieves the requested shape is kept. The
accepted seed is recorded in the .npz, which is what makes the case
reproducible.

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

FREEGSNKE = pathlib.Path(__file__).resolve().parents[2]
MACHINE = f"{FREEGSNKE}/machine_configs/ITER"

# Grid from FreeGSNKE's own ITER example. The target separatrix spans
# R 4.21-8.19, Z -3.91 to 3.56, so it sits inside with room for the legs.
RMIN, RMAX, ZMIN, ZMAX, NGRID = 3.2, 8.8, -5.0, 5.0, 129

# Topeol shape parameters, from FreeGSNKE's ITER example. alpha_n must stay an
# integer for jags' topeol to have a closed-form antiderivative.
ALPHA_M, ALPHA_N, BETAP, PROFILE_RAXIS = 2.0, 1, 0.15, 1.0
L2_REG = 1e-8
SWEEP = (1e-9, 1e-8, 1e-7)

# Seeds tried in order; the first that achieves the target boundary is kept.
SEEDS = (0, 1, 2, 3, 4, 5)
# Acceptance: normalised flux at the 24 target separatrix points should be 1
# everywhere. A collapsed solve misses by a factor of a few, so this is a wide
# gate that only has to separate "solved the right problem" from "did not".
MAX_MEAN_ERR, MAX_SD = 0.15, 0.15


def main(target_path="scripts/iter_target.npz", out_path="scripts/case_iter.npz",
         sweep=False):
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
        pn = (eq.psi_func(Rb, Zb, grid=False) - eq.psi_axis) / (
            eq.psi_bndry - eq.psi_axis
        )
        act = {k: v for k, v in eq.tokamak.getCurrents().items()
               if not str(k).startswith(("I", "O"))}
        return eq, profiles, solver, float(np.std(pn)), float(np.mean(pn)), \
            max(abs(v) for v in act.values())

    def describe(e, sd, mean, maxI):
        return (f"vol={e.plasmaVolume():>8.1f} m3  "
                f"diverted={str(not bool(e.flag_limiter)):<5}  "
                f"max|I_active|={maxI:.2e} A  "
                f"psiN at target LCFS: mean {mean:.3f} sd {sd:.3f}")

    if sweep:
        print("l2_reg sweep (diagnostic only -- L2_REG is what gets used):")
        for l2 in SWEEP:
            e, _, _, sd, mean, maxI = attempt(l2, SEEDS[0])
            print(f"  l2={l2:.0e}  {describe(e, sd, mean, maxI)}")

    print(f"inverse solve, l2_reg = {L2_REG:.0e}, trying seeds until the target "
          f"boundary is hit ...")
    best = None
    for seed in SEEDS:
        eq, profiles, solver, sd, mean, maxI = attempt(L2_REG, seed)
        ok = (not bool(eq.flag_limiter) and abs(mean - 1.0) < MAX_MEAN_ERR
              and sd < MAX_SD)
        print(f"  seed={seed}  {describe(eq, sd, mean, maxI)}  "
              f"{'ACCEPTED' if ok else 'rejected'}")
        if ok:
            best = (0.0, seed, eq, profiles, solver)
            break
        score = abs(mean - 1.0) + sd
        if best is None or score < best[0]:
            best = (score, seed, eq, profiles, solver)
    else:
        print(f"  no seed met the acceptance gate; keeping the closest "
              f"(seed {best[1]}), which is NOT a usable ITER equilibrium")
    accepted, seed, eq, profiles, solver = best[0] == 0.0, *best[1:]
    print(f"  using seed {seed}")
    if not accepted:
        # Writing to the usual name would let stages 2 and 3 run on a collapsed
        # plasma and report differences that mean nothing.
        out_path = str(out_path).replace(".npz", "_rejected.npz")
        print(f"  writing to {out_path} instead, so nothing downstream picks it up")

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
    )
