"""Cross-check jags against the FreeGSNKE reference equilibria.

Reads the .npz files written by ``dump_freegsnke_case.py`` and re-solves the same
problems: same grid, same limiter, same vacuum flux, same Lao85 profile. jags is
given FreeGSNKE's own converged ``psi_bndry`` as ``psi_edge``, so both codes are
solving the same problem and any discrepancy is model, not setup.

Prints the metrics and writes a comparison figure.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/compare.py [out.png] [--no-reach]
"""

import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jags import critical, profiles, reach, solver  # noqa: E402
from jags.grid import Grid  # noqa: E402
from jags.interp import make_interpolator  # noqa: E402

CASES = ["limited", "diverted"]
HERE = pathlib.Path(__file__).resolve().parent


def run(case, use_reach=True):
    d = np.load(HERE / f"case_{case}.npz")
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    machine = solver.build_machine(grid, coil_RZ=None)
    profile = profiles.lao85(
        psi_axis=float(d["psi_axis"]), psi_bndry=float(d["psi_bndry"]),
        alpha=d["alpha_full"], beta=d["beta_full"],
        L=float(d["L"]), fvac=float(d["fvac"]), Raxis=float(d["Raxis"]),
    )

    ref_j, Ip = np.asarray(d["jtor"]), float(d["Ip"])
    # Start from a blob at the reference current centroid, not the reference
    # solution: converging from a cold start is the point.
    centre = (
        float((ref_j * grid.R).sum() / ref_j.sum()),
        float((ref_j * grid.Z).sum() / ref_j.sum()),
    )
    res = solver.solve(
        machine, profile, jnp.asarray(d["tokamak_psi"]),
        solver.initial_guess(machine, centre, (0.45, 0.9), Ip),
        Ip=Ip, max_newton=40,
        reachability=(
            reach.make_reachability(grid, n_samples=64, beta_norm=2e5)
            if use_reach
            else None
        ),
    )

    value, _, _ = make_interpolator(grid)
    c = critical.make_critical_fns(grid)(
        res.psi, beta_norm=20000.0, use_limiter=False
    )
    return dict(
        case=case, d=d, grid=grid, res=res, limiter=limiter,
        psi=np.asarray(res.psi), ref_psi=np.asarray(d["psi"]),
        jtor=np.asarray(res.jtor), ref_jtor=ref_j, Ip=Ip,
        axis=np.asarray(c.axis), xpts=np.asarray(c.xpts if hasattr(c, "xpts") else c.xpoints),
        psi_x=float(value(res.psi, c.xpoints[0])),
    )


def report(r):
    g, d = r["grid"], r["d"]
    dA = g.dA
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731

    print(f"[{r['case']}]  {'limited' if bool(d['flag_limiter']) else 'diverted'}"
          f" per FreeGSNKE, {int(d['nR'])}x{int(d['nZ'])}")
    print("   newton    : " + "  ".join(f"{x:.1e}" for x in r["res"].residual_history))
    print(f"   rel L2 psi_plasma : {rel(np.asarray(r['res'].psi_p), d['plasma_psi']):.4e}")
    print(f"   rel L2 psi        : {rel(r['psi'], r['ref_psi']):.4e}")
    print(f"   rel L2 Jtor       : {rel(r['jtor'], r['ref_jtor']):.4e}")
    print(f"   max|dpsi| / range : {np.abs(r['psi'] - r['ref_psi']).max() / np.ptp(r['ref_psi']):.4e}")
    print(f"   Ip                : {r['jtor'].sum() * dA:.6e} (target {r['Ip']:.6e})")

    ours = r["jtor"] > 1e-3 * r["jtor"].max()
    theirs = r["ref_jtor"] > 1e-3 * r["ref_jtor"].max()
    extra = ours & ~theirs
    print(f"   plasma cells      : jags {ours.sum()}, freegsnke {theirs.sum()}, "
          f"jags-only {extra.sum()}, freegsnke-only {(theirs & ~ours).sum()}")
    if extra.any():
        cur = r["jtor"][extra].sum() * dA
        print(f"   current misplaced : {cur:.3e} A ({100 * cur / (r['jtor'].sum() * dA):.3f}% of Ip)")


def plot(runs, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 15))
    fig.suptitle("jags vs FreeGSNKE, MAST-U", fontsize=13, y=0.997)

    for row, r in enumerate(runs):
        g = r["grid"]
        R, Z = g.R, g.Z
        lim = np.concatenate([r["limiter"], r["limiter"][:1]], axis=0)
        psi, ref = r["psi"], r["ref_psi"]
        levels = np.linspace(psi.min(), psi.max(), 45)

        ax = axes[row, 0]
        ax.contour(R, Z, ref, levels=levels, colors="0.55", linewidths=1.6)
        ax.contour(R, Z, psi, levels=levels, colors="C0", linewidths=0.7)
        ax.contour(R, Z, ref, levels=[float(r["d"]["psi_bndry"])], colors="k", linewidths=2.0)
        ax.contour(R, Z, psi, levels=[r["psi_x"]], colors="crimson", linewidths=1.1)
        ax.plot(*r["axis"], "r+", ms=11, mew=2)
        ax.plot(r["xpts"][:, 0], r["xpts"][:, 1], "rx", ms=8, mew=1.8)
        ax.set_title(f"{r['case']} currents\ngrey/black = FreeGSNKE, blue/red = jags")

        ax = axes[row, 1]
        im = ax.contourf(R, Z, r["jtor"] * 1e-6, levels=30)
        ax.contour(R, Z, r["ref_jtor"] * 1e-6, levels=8, colors="k", linewidths=0.8)
        plt.colorbar(im, ax=ax, label="MA/m$^2$", fraction=0.046)
        e = np.linalg.norm(r["jtor"] - r["ref_jtor"]) / np.linalg.norm(r["ref_jtor"])
        ax.set_title(f"$J_\\phi$ (fill = jags, lines = FreeGSNKE)\nrel L2 = {e:.2e}")

        ax = axes[row, 2]
        diff = psi - ref
        vmax = np.abs(diff).max()
        im = ax.contourf(R, Z, diff, levels=np.linspace(-vmax, vmax, 31), cmap="RdBu_r")
        plt.colorbar(im, ax=ax, label=r"$\Delta\psi$", fraction=0.046)
        ax.set_title(f"jags $-$ FreeGSNKE\nmax$|\\Delta\\psi|$/range = {vmax / np.ptp(ref):.1e}")

        for ax in axes[row]:
            ax.plot(lim[:, 0], lim[:, 1], "k-", lw=1.3)
            ax.set_aspect("equal")
            ax.set_xlabel("R [m]")
        axes[row, 0].set_ylabel("Z [m]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"wrote {out_path}")


def main(out_path=None, use_reach=True):
    out_path = out_path or HERE / "comparison.png"
    runs = []
    for case in CASES:
        r = run(case, use_reach)
        report(r)
        runs.append(r)
    plot(runs, out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(
        args[0] if args else None,
        use_reach="--no-reach" not in sys.argv,
    )
