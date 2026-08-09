"""Cross-check jags/fsa.py on a diverted equilibrium against two contour-tracing codes.

The co-area formula replaces contour tracing, so the references are the codes
that trace:

**FreeGS4E** (``freegs4e/critical.py:1066``) computes ``q`` by ray-casting 128
points per surface from the O-point and integrating along the traced curve:

    q = (1 / 2 pi) contour_int F dl / (R^2 B_p),   B_p = |grad psi| / R

which is exactly ``fsa.safety_factor``: ``int_dl_over_Bp`` is the contour
integral of ``R dl / |grad psi|``, so multiplying by ``<1/R^2>`` (same weight,
so the normalisation cancels) leaves ``contour_int dl / (R |grad psi|)``.

**TORAX** (``torax/_src/geometry/eqdsk.py:330``) uses ``contourpy`` for the curve
and a bicubic spline for the gradient, and computes the whole
``StandardGeometryIntermediates`` list with ``FSA(G) = int G dl/Bp / int dl/Bp``
-- the same convention. Run via ``check_fsa_torax.py`` on a geqdsk written from
the same equilibrium, so the two codes share nothing but the psi grid.

The two references disagree with **each other** by a couple of percent, which is
the honest floor for "agreement" here; that spread is reported alongside.

The equilibrium is FreeGSNKE's *own* converged solution read from the .npz, so
this tests the averaging alone, not the jags solve.

Usage:
    /home/user/jaxgs/.venv/bin/python scripts/check_fsa.py [--all] [out.png]

``--all`` adds the higher-resolution cases under /tmp, if they have been
generated (see the resolution study in the module docstring of ``jags/fsa.py``).
"""

import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import reach  # noqa: E402
from jags.fsa import make_flux_surface_averager, safety_factor  # noqa: E402
from jags.grid import Grid  # noqa: E402

TWO_PI = 2.0 * np.pi

# (label, npz, torax npz). The 65x65 pair is committed; the rest are produced by
# dump_freegsnke_case.py + check_fsa_torax.py at higher n and left in /tmp.
CASES = [
    ("65", "scripts/case_diverted.npz", "/tmp/torax_diverted_65.npz"),
    ("129", "/tmp/case_diverted_129.npz", "/tmp/torax_diverted_129.npz"),
    ("193", "/tmp/case_diverted_193.npz", "/tmp/torax_diverted_193.npz"),
]

# jags quantity -> (TORAX name, factor to multiply the jags value by).
# B_p is convention-free so int_dl_over_Bp and the <1/R^n> match directly;
# TORAX's psi carries an extra 2 pi (COCOS 11, Wb not Wb/rad), which shows up
# only in the quantities that mention grad psi explicitly.
TORAX_MAP = {
    "int_dl_over_Bp": ("int_dl_over_Bp", 1.0),
    "avg_1_over_R": ("flux_surf_avg_1_over_R", 1.0),
    "avg_1_over_R2": ("flux_surf_avg_1_over_R2", 1.0),
    "avg_grad_psi": ("flux_surf_avg_grad_psi", TWO_PI),
    "avg_grad_psi2": ("flux_surf_avg_grad_psi2", TWO_PI**2),
    "avg_grad_psi2_over_R2": ("flux_surf_avg_grad_psi2_over_R2", TWO_PI**2),
}


def load(path):
    d = np.load(path)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    return d, grid


def jags_surfaces(d, grid, levels, variant="label"):
    """FluxSurfaces on given psi levels, by one of three level-set choices.

    A raw level set of psi is not one closed curve in a diverted equilibrium:
    contours near the separatrix reappear in the divertor legs. ``mask`` and
    ``label`` are the two ways of using ``reach.py`` to restrict to the core.
    """
    psi = jnp.asarray(d["psi"])
    psi_axis, psi_edge = float(d["psi_axis"]), float(d["psi_bndry"])
    scale = abs(psi_axis - psi_edge)

    _, surfaces, _ = make_flux_surface_averager(grid)
    if variant == "raw":
        return surfaces(psi, levels, psi_axis, psi_edge)

    m = reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi)
    if variant == "mask":
        mask = jax.nn.sigmoid((m - psi_edge) / (0.02 * scale))
        return surfaces(psi, levels, psi_axis, psi_edge, mask=mask)
    if variant == "label":
        return surfaces(psi, levels, psi_axis, psi_edge, label=m)
    raise ValueError(variant)


def against_freegs4e(d, grid):
    """jags q vs FreeGS4E's traced q, for each of the three level-set choices."""
    pn = np.asarray(d["psinorm"])
    ref = np.asarray(d["q"])
    pa, pb = float(d["psi_axis"]), float(d["psi_bndry"])
    levels = jnp.asarray(pa + pn * (pb - pa))
    F = jnp.asarray(d["fpol"])

    out = {}
    for variant in ("raw", "mask", "label"):
        fs = jags_surfaces(d, grid, levels, variant)
        q = np.asarray(safety_factor(fs, F))
        out[variant] = (q, np.abs(q - ref) / ref, fs)
    return pn, ref, out


def against_torax(d, grid, tx):
    """Every FSA quantity, on TORAX's own surfaces.

    TORAX puts psi(axis) = 0, flips the sign so psi grows outward, and works in
    Wb; jags keeps FreeGS4E's Wb/rad. The magnetic axis itself has no contour,
    so TORAX's first entry is a hard-coded placeholder and is dropped here.
    """
    pa = float(d["psi_axis"])
    levels_jags = pa - np.asarray(tx["psi"]) / TWO_PI
    fs = jags_surfaces(d, grid, jnp.asarray(levels_jags[1:]))

    rows = {}
    for ours, (theirs, factor) in TORAX_MAP.items():
        a = np.asarray(getattr(fs, ours)) * factor
        b = np.asarray(tx[theirs])[1:]
        rows[ours] = (a, b, np.abs(a - b) / np.abs(b))

    pn = (levels_jags[1:] - pa) / (float(d["psi_bndry"]) - pa)
    q_jags = np.asarray(safety_factor(fs, jnp.asarray(tx["F"])[1:]))
    return pn, rows, q_jags, np.asarray(tx["q_rebuilt"])[1:]


def report(label, npz, torax_npz):
    d, grid = load(npz)
    print(f"\n=== diverted, {int(d['nR'])}x{int(d['nZ'])} ===")

    pn, ref, out = against_freegs4e(d, grid)
    print("  vs FreeGS4E traced q")
    print("    psinorm    " + " ".join(f"{p:7.2f}" for p in pn[::3]))
    print("    reference  " + " ".join(f"{v:7.3f}" for v in ref[::3]))
    for variant, (q, rel, _) in out.items():
        print(f"    {variant:<9}  " + " ".join(f"{v:7.3f}" for v in q[::3]))
        print(f"    {'':<9}  rel err  median {np.median(rel):.2e}  "
              f"psinorm<=0.8 {rel[pn <= 0.8].max():.2e}  all {rel.max():.2e}")

    result = dict(label=label, d=d, grid=grid, pn=pn, ref=ref, out=out, tx=None)

    if not os.path.exists(torax_npz):
        print(f"  (no TORAX reference at {torax_npz})")
        return result

    tx = np.load(torax_npz)
    tpn, rows, q_jags, q_tx = against_torax(d, grid, tx)
    # TORAX's surfaces start at psinorm ~ 0.016, a curve about two cells across
    # that no grid method resolves, so the useful band is quoted separately.
    # Every quantity's worst error sits on those innermost surfaces.
    band = (tpn >= 0.2) & (tpn <= 0.8)
    print(f"  vs TORAX eqdsk parser ({tx['n_surfaces']} surfaces, "
          f"last_surface_factor={float(tx['last_surface_factor'])})")
    print(f"    {'':<24} {'all surfaces':>22}    {'0.2 <= psinorm <= 0.8':>24}")
    for name, (_, _, rel) in rows.items():
        i = int(np.argmax(rel))
        print(f"    {name:<24} med {np.median(rel):.2e} "
              f"worst {rel[i]:.2e}@{tpn[i]:.3f}    "
              f"med {np.median(rel[band]):.2e}  max {rel[band].max():.2e}")

    # Reference-vs-reference: the two tracing codes on the same psi grid. This
    # is the floor -- jags cannot meaningfully beat the spread between the codes
    # it is being checked against.
    ref_i = np.interp(tpn, pn, ref)
    spread = np.abs(q_tx - ref_i) / ref_i
    rel_j = np.abs(q_jags - q_tx) / q_tx
    for name, rel in [("q: TORAX vs FreeGS4E", spread), ("q: jags vs TORAX", rel_j)]:
        tail = "   <-- reference spread" if "TORAX vs" in name else ""
        print(f"    {name:<24} med {np.median(rel):.2e} {'':>17}    "
              f"med {np.median(rel[band]):.2e}  max {rel[band].max():.2e}{tail}")

    result["tx"] = dict(pn=tpn, rows=rows, q_jags=q_jags, q_tx=q_tx, spread=spread)
    return result


def plot(runs, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(runs), 3, figsize=(15, 4.6 * len(runs)),
                             squeeze=False)
    fig.suptitle("jags flux-surface averages vs contour tracing (MAST-U, diverted)",
                 y=0.998)

    for row, r in enumerate(runs):
        d, grid, pn, ref = r["d"], r["grid"], r["pn"], r["ref"]

        ax = axes[row][0]
        ax.plot(pn, ref, "k-o", ms=4, lw=1.6, label="FreeGS4E (traced)")
        if r["tx"]:
            ax.plot(r["tx"]["pn"], r["tx"]["q_tx"], "-", color="0.55", lw=2.4,
                    label="TORAX (traced)")
        for variant, style in [("raw", "C3--"), ("mask", "C1-."), ("label", "C0-")]:
            ax.plot(pn, r["out"][variant][0], style, lw=1.5, label=f"jags {variant}")
        ax.set_xlabel(r"$\psi_N$"), ax.set_ylabel("q"), ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.set_title(f"{r['label']}$\\times${r['label']}: safety factor")

        ax = axes[row][1]
        for variant, style in [("raw", "C3--"), ("mask", "C1-."), ("label", "C0-")]:
            ax.semilogy(pn, r["out"][variant][1], style, lw=1.5, label=f"jags {variant}")
        if r["tx"]:
            ax.semilogy(r["tx"]["pn"], r["tx"]["spread"], "-", color="0.55", lw=2.4,
                        label="TORAX vs FreeGS4E")
        ax.set_xlabel(r"$\psi_N$"), ax.set_ylabel("relative error in q")
        ax.legend(fontsize=8), ax.set_title("error, vs the reference spread")

        ax = axes[row][2]
        psi = np.asarray(d["psi"])
        m = np.asarray(
            reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(jnp.asarray(psi))
        )
        pa, pb = float(d["psi_axis"]), float(d["psi_bndry"])
        levels = np.sort(pa + pn * (pb - pa))
        ax.contour(grid.R, grid.Z, psi, levels=levels, colors="0.72", linewidths=0.9)
        ax.contour(grid.R, grid.Z, m, levels=levels, colors="C0", linewidths=1.0)
        lim = np.stack([d["limiter_R"], d["limiter_Z"]], -1)
        ax.plot(lim[:, 0], lim[:, 1], "k-", lw=1.2)
        ax.set_aspect("equal"), ax.set_xlabel("R [m]"), ax.set_ylabel("Z [m]")
        ax.set_title("grey: level sets of $\\psi$ (lobes included)\n"
                     "blue: level sets of the reachability")

    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"\nwrote {out_path}")


def main(cases, out_path):
    runs = [report(*c) for c in cases if os.path.exists(c[1])]
    if runs:
        plot(runs, out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(
        CASES if "--all" in sys.argv else CASES[:1],
        args[0] if args else "scripts/fsa_check.png",
    )
