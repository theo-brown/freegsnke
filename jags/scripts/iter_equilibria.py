"""Plot the three ITER equilibria side by side.

* CHEASE     TORAX's ``iterhybrid_cocos11.eqdsk`` -- the equilibrium the shape
             target was taken from, and the one TORAX normally runs on.
* FreeGSNKE  the forward solve from ``iter_inverse.py``.
* jags       the forward solve from ``iter_compare.py``, same grid, same vacuum
             flux, same Topeol coefficients.

All three are drawn in normalised flux, ``(psi - psi_axis) / (psi_bndry -
psi_axis)``, so they share a colour scale and the shapes can be read against
each other directly -- the raw psi values do not agree in offset or sign
convention and would show nothing.

The CHEASE separatrix is drawn on all three panels, which is what makes the
stage 1 caveat visible: the two solves reproduce each other closely and neither
reproduces CHEASE.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/iter_equilibria.py [out.png]
"""

import pathlib
import sys

import numpy as np

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
# psi_N contours: inside the plasma, then a few outside to show the legs.
LEVELS = np.concatenate([np.linspace(0.0, 1.0, 21), np.linspace(1.05, 1.6, 8)])


def load():
    t = np.load(HERE / "iter_target.npz")
    cases = [(
        "CHEASE (TORAX eqdsk)",
        np.asarray(t["map_R"]), np.asarray(t["map_Z"]), np.asarray(t["map_psi"]),
        float(t["map_psi_axis"]), float(t["map_psi_bndry"]),
        (float(t["R_axis"]), float(t["Z_axis"])), None,
    )]

    def grid_of(d):
        return (np.linspace(float(d["Rmin"]), float(d["Rmax"]), int(d["nR"])),
                np.linspace(float(d["Zmin"]), float(d["Zmax"]), int(d["nZ"])))

    f = np.load(HERE / "case_iter.npz", allow_pickle=True)
    R, Z = grid_of(f)
    lim = np.stack([f["limiter_R"], f["limiter_Z"]], axis=-1)
    cases.append((
        "FreeGSNKE", R, Z, np.asarray(f["psi"]),
        float(f["psi_axis"]), float(f["psi_bndry"]),
        tuple(np.asarray(f["opt"])[0][:2]), lim,
    ))

    j_path = HERE / "case_iter_jags.npz"
    if j_path.exists():
        j = np.load(j_path)
        R, Z = grid_of(j)
        cases.append((
            "jags", R, Z, np.asarray(j["psi"]),
            float(j["psi_axis"]), float(j["psi_bndry"]),
            (float(j["R_axis"]), float(j["Z_axis"])), lim,
        ))
    else:
        print(f"{j_path} not found -- run iter_compare.py; plotting without jags")
    return t, cases


def main(out_path=None):
    out_path = out_path or HERE / "iter_equilibria.png"
    t, cases = load()
    tR, tZ = np.asarray(t["lcfs_R"]), np.asarray(t["lcfs_Z"])
    Rx, Zx = float(t["Rx"]), float(t["Zx"])

    fig, axes = plt.subplots(1, len(cases) + 1, figsize=(4.6 * (len(cases) + 1), 9.5))
    fig.suptitle("ITER hybrid: the CHEASE equilibrium and the two forward solves "
                 "of it, in normalised flux", fontsize=13, y=0.98)

    for ax, (name, R, Z, psi, pa, pb, axis, lim) in zip(axes, cases):
        # .T because matplotlib wants C[j, i] for (X[i], Y[j]) while jags and
        # the eqdsk both store psi[R_i, Z_j]. With a square grid the wrong one
        # plots happily and silently, as the transpose of the truth.
        pn = ((psi - pa) / (pb - pa)).T
        im = ax.contourf(R, Z, pn, levels=LEVELS, cmap="viridis", extend="max")
        ax.contour(R, Z, pn, levels=[1.0], colors="crimson", linewidths=2.2)
        ax.plot(tR, tZ, "--", color="w", lw=1.5)
        ax.plot(*axis, "w+", ms=13, mew=2.2)
        ax.plot(Rx, Zx, "wx", ms=9, mew=2.0)
        if lim is not None:
            closed = np.concatenate([lim, lim[:1]], axis=0)
            ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1.4)
        ax.set_title(f"{name}\naxis ({axis[0]:.2f}, {axis[1]:.2f})")
        ax.set_xlabel("R [m]")
        ax.set_aspect("equal")
        ax.set_xlim(3.2, 8.8), ax.set_ylim(-5.0, 5.0)
    axes[0].set_ylabel("Z [m]")
    plt.colorbar(im, ax=axes[len(cases) - 1], label=r"$\psi_N$", fraction=0.046)

    ax = axes[-1]
    styles = [("CHEASE (TORAX eqdsk)", "darkgreen", "-", 2.4),
              ("FreeGSNKE", "0.25", "--", 2.0), ("jags", "C0", ":", 2.0)]
    for (name, R, Z, psi, pa, pb, axis, lim), (_, c, ls, lw) in zip(cases, styles):
        pn = ((psi - pa) / (pb - pa)).T
        ax.contour(R, Z, pn, levels=[1.0], colors=c, linewidths=lw, linestyles=ls)
        # a proxy artist for the legend: contour sets are not legend handles,
        # and .collections was removed in matplotlib 3.8
        ax.plot([], [], color=c, ls=ls, lw=lw, label=name)
        ax.plot(*axis, "+", color=c, ms=13, mew=2.2)
    if cases[-1][7] is not None:
        closed = np.concatenate([cases[-1][7], cases[-1][7][:1]], axis=0)
        ax.plot(closed[:, 0], closed[:, 1], "k-", lw=1.2)
    ax.plot(Rx, Zx, "kx", ms=10, mew=2.0)
    ax.set_title("separatrices\nCHEASE X-point marked")
    ax.set_xlabel("R [m]"), ax.set_aspect("equal")
    ax.set_xlim(3.2, 8.8), ax.set_ylim(-5.0, 5.0)
    ax.grid(alpha=0.3), ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
