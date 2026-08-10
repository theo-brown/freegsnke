"""Stage 2: forward-solve the ITER hybrid case in jags, side by side with FreeGSNKE.

Reads ``case_iter.npz`` from ``iter_inverse.py`` -- same grid, same limiter, same
vacuum flux from the inverse-solved coil currents, same Topeol profile
coefficients -- and re-solves from a cold start. jags is given FreeGSNKE's
converged ``psi_bndry`` as ``psi_edge``, exactly as in the MAST-U cross-check, so
both codes are posed the same problem and any difference is the model rather than
the setup.

The profile is ``jags.profiles.topeol``, the p(psi)/F(psi) form of FreeGS4E's
``Fiesta_Topeol``. Both sides therefore use fixed coefficients with no inner
constraint solve, which is what makes this a solver comparison.

Runs in the jags venv (no FreeGSNKE, no TORAX).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/iter_compare.py [case.npz] [out.png]
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

from jags import critical, profiles, reach, solver, torax_geom  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402
from jags.interp import make_interpolator  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
N_SURFACES = 60


def run(npz):
    d = np.load(npz, allow_pickle=True)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    machine = solver.build_machine(grid, coil_RZ=None)
    psi_axis, psi_edge = float(d["psi_axis"]), float(d["psi_bndry"])
    profile = profiles.topeol(
        psi_axis=psi_axis, psi_bndry=psi_edge,
        L=float(d["L"]), Beta0=float(d["Beta0"]),
        Raxis=float(d["profile_Raxis"]), fvac=float(d["fvac"]),
        alpha_m=float(d["alpha_m"]), alpha_n=int(d["alpha_n"]),
    )

    ref_j, Ip = np.asarray(d["jtor"]), float(d["Ip"])
    centre = (
        float((ref_j * grid.R).sum() / ref_j.sum()),
        float((ref_j * grid.Z).sum() / ref_j.sum()),
    )
    # ITER's minor radius is ~2 m on a 5.6 x 10 m box; the MAST-U blob fractions
    # would start the iteration with a plasma several times too small.
    res = solver.solve(
        machine, profile, jnp.asarray(d["tokamak_psi"]),
        solver.initial_guess(machine, centre, (1.6, 2.6), Ip),
        Ip=Ip, max_newton=40,
        reachability=reach.make_reachability(grid, n_samples=64, beta_norm=2e5),
    )

    value, _, _ = make_interpolator(grid)
    c = critical.make_critical_fns(grid)(res.psi, beta_norm=20000.0, use_limiter=False)
    return dict(
        d=d, grid=grid, res=res, limiter=limiter, profile=profile,
        psi=np.asarray(res.psi), ref_psi=np.asarray(d["psi"]),
        jtor=np.asarray(res.jtor), ref_jtor=ref_j, Ip=Ip,
        psi_axis=psi_axis, psi_edge=psi_edge,
        axis=np.asarray(c.axis), xpts=np.asarray(c.xpoints),
        psi_x=float(value(res.psi, c.xpoints[0])),
        psi_axis_jags=float(value(res.psi, c.axis)),
    )


def q_profile(r, psinorm):
    """jags' safety factor on the same psiN grid FreeGSNKE reported."""
    grid, psi = r["grid"], jnp.asarray(r["psi"])
    pa, pb = r["psi_axis"], r["psi_edge"]
    _, surfaces, _ = make_flux_surface_averager(grid)
    label = reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi)
    levels = jnp.asarray(pa + (pb - pa) * np.asarray(psinorm))
    fs = surfaces(psi, levels, pa, pb, label=label)
    F = r["profile"].F(levels)
    return np.asarray(torax_geom.safety_factor(fs, F)), np.asarray(fs.n_eff)


def report(r):
    g, d = r["grid"], r["d"]
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    print(f"[iter]  {'limited' if bool(d['flag_limiter']) else 'diverted'} per "
          f"FreeGSNKE, {int(d['nR'])}x{int(d['nZ'])}")
    print("   newton    : " + "  ".join(f"{x:.1e}" for x in r["res"].residual_history))
    print(f"   converged         : {bool(r['res'].converged)}")
    print(f"   rel L2 psi_plasma : {rel(np.asarray(r['res'].psi_p), d['plasma_psi']):.4e}")
    print(f"   rel L2 psi        : {rel(r['psi'], r['ref_psi']):.4e}")
    print(f"   rel L2 Jtor       : {rel(r['jtor'], r['ref_jtor']):.4e}")
    print(f"   max|dpsi| / range : "
          f"{np.abs(r['psi'] - r['ref_psi']).max() / np.ptp(r['ref_psi']):.4e}")
    print(f"   Ip                : {r['jtor'].sum() * g.dA:.6e} (target {r['Ip']:.6e})")
    opt = np.asarray(d["opt"])[0]
    print(f"   axis   jags ({r['axis'][0]:.4f}, {r['axis'][1]:.4f})  "
          f"freegsnke ({opt[0]:.4f}, {opt[1]:.4f})")
    xpt = np.asarray(d["xpt"])[0]
    print(f"   X-pt   jags ({r['xpts'][0, 0]:.4f}, {r['xpts'][0, 1]:.4f})  "
          f"freegsnke ({xpt[0]:.4f}, {xpt[1]:.4f})")
    ours = r["jtor"] > 1e-3 * r["jtor"].max()
    theirs = r["ref_jtor"] > 1e-3 * r["ref_jtor"].max()
    print(f"   plasma cells      : jags {ours.sum()}, freegsnke {theirs.sum()}, "
          f"jags-only {(ours & ~theirs).sum()}, freegsnke-only {(theirs & ~ours).sum()}")


def plot(r, q_jags, out_path):
    d, g = r["d"], r["grid"]
    R, Z = g.R, g.Z
    lim = np.concatenate([r["limiter"], r["limiter"][:1]], axis=0)
    psi, ref = r["psi"], r["ref_psi"]
    levels = np.linspace(ref.min(), ref.max(), 45)
    tR, tZ = np.asarray(d["target_lcfs_R"]), np.asarray(d["target_lcfs_Z"])
    opt = np.asarray(d["opt"])[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 13.5))
    fig.suptitle("ITER hybrid: FreeGSNKE vs jags forward solve at the same coil "
                 "currents", fontsize=13, y=0.998)

    for ax, field, name, col in (
        (axes[0, 0], ref, "FreeGSNKE", "0.35"),
        (axes[0, 1], psi, "jags", "C0"),
    ):
        ax.contour(R, Z, field, levels=levels, colors=col, linewidths=0.7)
        b = float(d["psi_bndry"]) if name == "FreeGSNKE" else r["psi_x"]
        ax.contour(R, Z, field, levels=[b], colors="crimson", linewidths=2.0)
        ax.plot(tR, tZ, "--", color="darkgreen", lw=1.4, label="TORAX eqdsk LCFS")
        a = opt if name == "FreeGSNKE" else r["axis"]
        ax.plot(a[0], a[1], "r+", ms=12, mew=2)
        ax.set_title(f"{name}\nred = separatrix, green = target")
        ax.legend(loc="lower left", fontsize=7)

    ax = axes[0, 2]
    diff = psi - ref
    vmax = np.abs(diff).max()
    im = ax.contourf(R, Z, diff, levels=np.linspace(-vmax, vmax, 31), cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label=r"$\Delta\psi$", fraction=0.046)
    ax.set_title(f"jags $-$ FreeGSNKE\nmax$|\\Delta\\psi|$/range = "
                 f"{vmax / np.ptp(ref):.1e}")

    ax = axes[1, 0]
    im = ax.contourf(R, Z, r["jtor"] * 1e-6, levels=30)
    ax.contour(R, Z, r["ref_jtor"] * 1e-6, levels=8, colors="k", linewidths=0.8)
    plt.colorbar(im, ax=ax, label="MA/m$^2$", fraction=0.046)
    e = np.linalg.norm(r["jtor"] - r["ref_jtor"]) / np.linalg.norm(r["ref_jtor"])
    ax.set_title(f"$J_\\phi$ (fill = jags, lines = FreeGSNKE)\nrel L2 = {e:.2e}")

    ax = axes[1, 1]
    ax.contour(R, Z, ref, levels=[float(d["psi_bndry"])], colors="0.35", linewidths=2.4)
    ax.contour(R, Z, psi, levels=[r["psi_x"]], colors="C0", linewidths=1.3)
    ax.plot(tR, tZ, "--", color="darkgreen", lw=1.4)
    ax.plot(np.asarray(d["isoflux_R"]), np.asarray(d["isoflux_Z"]), "o",
            color="darkgreen", ms=4)
    ax.set_title("separatrix overlay\ngrey = FreeGSNKE, blue = jags, green = target")

    for ax in list(axes[0]) + [axes[1, 0], axes[1, 1]]:
        ax.plot(lim[:, 0], lim[:, 1], "k-", lw=1.3)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
    axes[0, 0].set_ylabel("Z [m]")
    axes[1, 0].set_ylabel("Z [m]")

    ax = axes[1, 2]
    pn, q_ref = np.asarray(d["psinorm"]), np.asarray(d["q"])
    ok = np.isfinite(q_ref)
    ax.plot(pn[ok], q_ref[ok], "o-", color="0.35", lw=1.6, ms=4, label="FreeGSNKE")
    ax.plot(pn, q_jags, "s--", color="C0", lw=1.3, ms=4, label="jags")
    ax.set_xlabel(r"$\psi_N$"), ax.set_ylabel("q")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    if ok.any():
        err = np.abs(q_jags[ok] - q_ref[ok]) / np.abs(q_ref[ok])
        ax.set_title(f"safety factor\nmedian rel diff = {np.median(err):.2e}")
    else:
        ax.set_title("safety factor")

    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"wrote {out_path}")


def save(r, path):
    """The jags solution, in the form ``iter_torax.py`` needs to build a geometry.

    Carries the Topeol coefficients rather than a sampled F, so the TORAX
    geometry can evaluate F(psi) in closed form instead of interpolating off a
    contour -- the one place the direct route beats reading a geqdsk.
    """
    d, g = r["d"], r["grid"]
    np.savez_compressed(
        path,
        Rmin=g.Rmin, Rmax=g.Rmax, Zmin=g.Zmin, Zmax=g.Zmax, nR=g.nR, nZ=g.nZ,
        limiter_R=d["limiter_R"], limiter_Z=d["limiter_Z"],
        psi=r["psi"], jtor=r["jtor"],
        # jags' own critical values, not the reference ones: these set the flux
        # levels the surfaces are traced on.
        psi_axis=float(r["psi_axis_jags"]), psi_bndry=r["psi_x"],
        R_axis=r["axis"][0], Z_axis=r["axis"][1],
        # the profile, still normalised against the reference axis and boundary,
        # because that is how the problem was posed to both codes.
        ref_psi_axis=r["psi_axis"], ref_psi_bndry=r["psi_edge"],
        L=d["L"], Beta0=d["Beta0"], profile_Raxis=d["profile_Raxis"],
        fvac=d["fvac"], alpha_m=d["alpha_m"], alpha_n=d["alpha_n"], Ip=r["Ip"],
    )
    print(f"wrote {path}")


def main(npz=None, out_path=None):
    npz = npz or HERE / "case_iter.npz"
    out_path = out_path or HERE / "iter_comparison.png"
    r = run(npz)
    report(r)
    pn = np.asarray(r["d"]["psinorm"])
    q_jags, n_eff = q_profile(r, pn)
    q_ref = np.asarray(r["d"]["q"])
    print(f"   q jags            : {np.array2string(q_jags, precision=3)}")
    print(f"   q freegsnke       : {np.array2string(q_ref, precision=3)}")
    print(f"   n_eff             : {np.array2string(n_eff, precision=0)}")
    save(r, str(npz).replace(".npz", "_jags.npz"))
    plot(r, q_jags, out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, args[1] if len(args) > 1 else None)
