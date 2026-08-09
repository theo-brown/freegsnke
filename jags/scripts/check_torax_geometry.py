"""Every geometry value TORAX needs: its eqdsk parser vs jags, on the same psi.

TORAX consumes a ``StandardGeometryIntermediates`` bundle. Its EQDSK path builds
it by tracing contours; ``jags.torax_geom`` builds the same bundle from the
co-area averages, differentiably. Both are run here on FreeGSNKE's own converged
MAST-U equilibrium, so the comparison is of the two geometry pipelines and not
of the equilibrium solve.

``F`` is taken from TORAX for the comparison, so that every difference shown is
geometry. In a coupled solve jags would use its own profile function, which is
exact rather than interpolated off a contour.

Usage:
    /home/user/jaxgs/.venv/bin/python scripts/check_torax_geometry.py [--all] [out.png]
"""

import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import critical, reach, torax_geom  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402

CASES = [
    ("65", "scripts/case_diverted.npz", "/tmp/torax_diverted_65.npz"),
    ("129", "/tmp/case_diverted_129.npz", "/tmp/torax_diverted_129.npz"),
    ("193", "/tmp/case_diverted_193.npz", "/tmp/torax_diverted_193.npz"),
]

# TORAX profile fields, in the order they appear in its dataclass.
PROFILES = [
    "psi", "Ip_profile", "Phi", "R_in", "R_out", "F", "int_dl_over_Bp",
    "flux_surf_avg_1_over_R", "flux_surf_avg_1_over_R2",
    "flux_surf_avg_grad_psi", "flux_surf_avg_grad_psi2",
    "flux_surf_avg_grad_psi2_over_R2", "flux_surf_avg_B2",
    "flux_surf_avg_1_over_B2", "delta_upper_face", "delta_lower_face",
    "elongation", "vpr",
]
SCALARS = ["R_major", "a_minor", "B_0", "z_magnetic_axis"]

# TORAX fields with no jags counterpart, and why.
NOT_GEOMETRY = {
    "geometry_type": "configuration",
    "Ip_from_parameters": "configuration",
    "face_centers": "configuration (TORAX's own rho grid)",
    "hires_factor": "configuration",
    "diverted": "SOL; None in TORAX's own EQDSK path",
    "connection_length_target": "SOL; None in TORAX's own EQDSK path",
    "connection_length_divertor": "SOL; None in TORAX's own EQDSK path",
    "angle_of_incidence_target": "SOL; None in TORAX's own EQDSK path",
    "R_OMP": "SOL; None in TORAX's own EQDSK path",
    "R_target": "SOL; None in TORAX's own EQDSK path",
    "B_pol_OMP": "SOL; None in TORAX's own EQDSK path",
}


def build(npz, torax_npz):
    d, tx = np.load(npz), np.load(torax_npz)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    psi = jnp.asarray(d["psi"])
    psi_axis, psi_edge = float(d["psi_axis"]), float(d["psi_bndry"])

    # TORAX's own surfaces, minus the axis placeholder which has no contour.
    levels = jnp.asarray((psi_axis - np.asarray(tx["psi"]) / (2 * np.pi))[1:])
    F = jnp.asarray(tx["F"])[1:]

    average, surfaces, grad_psi = make_flux_surface_averager(grid)
    m = reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi)
    fs = surfaces(psi, levels, psi_axis, psi_edge, label=m)

    one_over_B2 = torax_geom.avg_1_over_B2(
        average, psi, levels, psi_axis, psi_edge, grad_psi, F,
        jnp.asarray(grid.R), label=m,
    )
    axis, _ = critical.make_axis_finder(grid)[0](psi)
    ours = torax_geom.intermediates(fs, F, psi_axis, float(axis[1]), one_over_B2)

    theirs = {k: np.asarray(tx[k])[1:] for k in PROFILES if k in tx}
    theirs.update({k: float(tx[k]) for k in SCALARS if k in tx})
    pn = (np.asarray(levels) - psi_axis) / (psi_edge - psi_axis)
    return ours, theirs, pn, grid


def report(label, npz, torax_npz):
    ours, theirs, pn, grid = build(npz, torax_npz)
    band = (pn >= 0.2) & (pn <= 0.8)
    print(f"\n{'=' * 96}\n{label}x{label}   "
          f"{len(pn)} surfaces, psi_N {pn[0]:.3f} to {pn[-1]:.3f}\n{'=' * 96}")

    print(f"{'scalar':<26} {'TORAX':>14} {'jags':>14} {'rel':>10}")
    for k in SCALARS:
        a, b = float(getattr(ours, k)), theirs.get(k)
        if b is None:
            continue
        # z_magnetic_axis is ~1e-8 in an up-down symmetric case, so a relative
        # error on it is meaningless; report the absolute difference instead.
        if k == "z_magnetic_axis":
            print(f"  {k:<24} {b:>14.6g} {a:>14.6g} {abs(a - b):>10.2e} (abs)")
            continue
        r = abs(a - b) / max(abs(b), 1e-30)
        print(f"  {k:<24} {b:>14.6g} {a:>14.6g} {r:>10.2e}")

    print(f"\n{'profile':<32} {'TORAX @0.5':>12} {'jags @0.5':>12} "
          f"{'med':>9} {'0.2-0.8':>9} {'worst':>9} {'at':>6}")
    rows = {}
    for k in PROFILES:
        if k not in theirs:
            continue
        a, b = np.asarray(getattr(ours, k)), theirs[k]
        scale = np.maximum(np.abs(b), np.abs(b).max() * 1e-6)
        rel = np.abs(a - b) / scale
        i = int(np.nanargmax(rel))
        j = int(np.argmin(np.abs(pn - 0.5)))
        rows[k] = rel
        note = "" if np.all(np.isfinite(a)) else "  [not computed]"
        print(f"  {k:<30} {b[j]:>12.5g} {a[j]:>12.5g} "
              f"{np.nanmedian(rel):>9.2e} {np.nanmax(rel[band]):>9.2e} "
              f"{rel[i]:>9.2e} {pn[i]:>6.3f}{note}")

    print(f"\n  n_eff {np.asarray(ours.n_eff).min():.0f} to "
          f"{np.asarray(ours.n_eff).max():.0f}")
    return dict(label=label, pn=pn, rows=rows, ours=ours, theirs=theirs)


def plot(runs, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in PROFILES if k in runs[-1]["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("TORAX StandardGeometryIntermediates: eqdsk tracing vs jags co-area",
                 y=0.98)

    ax = axes[0]
    r = runs[-1]
    for k in keys:
        ax.semilogy(r["pn"], np.maximum(r["rows"][k], 1e-12), lw=1.2, label=k)
    ax.set_xlabel(r"$\psi_N$"), ax.set_ylabel("relative difference")
    ax.set_title(f"every profile, {r['label']}$\\times${r['label']}")
    ax.legend(fontsize=6.5, ncol=2), ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(keys))
    for run, marker in zip(runs, "o^s"):
        band = (run["pn"] >= 0.2) & (run["pn"] <= 0.8)
        vals = [np.nanmax(run["rows"][k][band]) for k in keys]
        ax.semilogy(x, np.maximum(vals, 1e-12), marker, ls="-", ms=5,
                    label=f"{run['label']}x{run['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"max relative difference, $0.2\leq\psi_N\leq0.8$")
    ax.set_title("convergence with grid"), ax.legend(fontsize=8), ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"\nwrote {out_path}")


def main(cases, out_path):
    runs = [report(*c) for c in cases
            if os.path.exists(c[1]) and os.path.exists(c[2])]
    print("\nTORAX fields with no jags counterpart:")
    for k, why in NOT_GEOMETRY.items():
        print(f"  {k:<32} {why}")
    if runs:
        plot(runs, out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(
        CASES if "--all" in sys.argv else CASES[:1],
        args[0] if args else "scripts/torax_geometry.png",
    )
