"""Loose coupling: run TORAX on a jags equilibrium, and check what it sees.

The coupling itself is three lines (``jags.torax_bridge``): jags psi ->
``StandardGeometryIntermediates`` -> ``StandardGeometry`` -> a
``GeometryProvider``. What needs checking is whether TORAX's *transport-facing*
geometry -- the metric coefficients g0..g3, vpr, spr, and the rho grid they live
on -- comes out the same as when TORAX builds it itself from a geqdsk of the
identical equilibrium.

That is a stronger check than comparing the intermediates, because
``build_standard_geometry`` interpolates onto TORAX's own rho_norm grid and
forms the metric coefficients, and an error in the rho mapping would show up
here and nowhere earlier.

Both paths run in one process, in the TORAX venv, which also runs jags.

Usage:
    /home/user/torax/.venv/bin/python scripts/couple_torax.py [case.npz] [out.png]
"""

import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import critical, reach, torax_bridge  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402

N_SURFACES = 60
N_RHO = 25

# Geometry fields worth comparing: the ones the transport equations read, on
# the cell grid. The *_face, *_hires and bookkeeping entries are dropped -- they
# are the same quantities on other grids.
FIELDS = [
    "Phi", "volume", "area", "vpr", "spr",
    "g0", "g1", "g2", "g3", "gm4", "gm5", "g2g3_over_rhon",
    "F", "R_in", "R_out", "elongation",
]
SCALARS = ["R_major", "a_minor", "B_0"]


def from_jags(npz, n_rho=N_RHO):
    d = np.load(npz)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    psi = jnp.asarray(d["psi"])
    psi_axis, psi_edge = float(d["psi_axis"]), float(d["psi_bndry"])

    averager = make_flux_surface_averager(grid)
    label = reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi)
    axis, _ = critical.make_axis_finder(grid)[0](psi)

    # F(psi) from the reference profile. A coupled solve would pass the closed
    # form straight in; the .npz only carries it sampled, so interpolate.
    pn_ref, f_ref = np.asarray(d["psinorm"]), np.asarray(d["fpol"])

    def F_of_psi(levels):
        pn = (np.asarray(levels) - psi_axis) / (psi_edge - psi_axis)
        return np.interp(pn, pn_ref, f_ref)

    inter = torax_bridge.build_intermediates(
        grid, averager, psi, F_of_psi, psi_axis, psi_edge,
        float(axis[0]), float(axis[1]),
        n_surfaces=N_SURFACES, label=label, n_rho=n_rho,
    )
    return torax_bridge.build_geometry(inter), d


def from_eqdsk(gfile, n_rho=N_RHO):
    from torax._src.geometry import eqdsk as torax_eqdsk

    inter = torax_eqdsk._construct_intermediates_from_eqdsk(
        geometry_directory=None, geometry_file=None,
        eqdsk_object=_load(gfile), hires_factor=4, Ip_from_parameters=False,
        face_centers=np.linspace(0.0, 1.0, n_rho + 1),
        n_surfaces=N_SURFACES, last_surface_factor=torax_bridge.LAST_SURFACE_FACTOR,
        cocos=7,
    )
    from torax._src.geometry import standard_geometry

    return standard_geometry.build_standard_geometry(inter)


def _load(gfile):
    import eqdsk

    return eqdsk.EQDSKInterface.from_file(gfile, from_cocos=7, no_cocos=False)


def compare(a, b, label_a="jags", label_b="eqdsk"):
    print(f"{'scalar':<18} {label_b:>13} {label_a:>13} {'rel':>10}")
    for k in SCALARS:
        x, y = float(getattr(a, k)), float(getattr(b, k))
        print(f"  {k:<16} {y:>13.6g} {x:>13.6g} {abs(x - y) / abs(y):>10.2e}")

    print(f"\n{'field':<18} {label_b + ' @0.5':>13} {label_a + ' @0.5':>13} "
          f"{'med':>10} {'max':>10} {'at rho':>8}")
    rows = {}
    for k in FIELDS:
        x = np.asarray(getattr(a, k), dtype=float)
        y = np.asarray(getattr(b, k), dtype=float)
        if x.shape != y.shape:
            print(f"  {k:<16} shape {x.shape} vs {y.shape} -- skipped")
            continue
        scale = np.maximum(np.abs(y), np.abs(y).max() * 1e-9)
        rel = np.abs(x - y) / scale
        rows[k] = rel
        j = len(x) // 2
        i = int(np.argmax(rel))
        print(f"  {k:<16} {y[j]:>13.5g} {x[j]:>13.5g} "
              f"{np.median(rel):>10.2e} {rel[i]:>10.2e} {i / len(x):>8.2f}")
    return rows


def plot(rows, rho, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for k, rel in rows.items():
        ax.semilogy(rho, np.maximum(rel, 1e-14), lw=1.3, label=k)
    ax.set_xlabel(r"$\hat\rho$"), ax.set_ylabel("relative difference")
    ax.set_title("TORAX StandardGeometry: built from jags vs from its own eqdsk parser")
    ax.legend(fontsize=7, ncol=2), ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    print(f"\nwrote {out_path}")


def main(npz, out_path):
    geo_jags, _ = from_jags(npz)
    geo_eqdsk = from_eqdsk(npz.replace(".npz", ".geqdsk"))
    print(f"TORAX transport grid: {geo_jags.rho_norm.shape[0]} cells, "
          f"{N_SURFACES} flux surfaces from jags\n")
    rows = compare(geo_jags, geo_eqdsk)
    plot(rows, np.asarray(geo_jags.rho_norm), out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(
        args[0] if args else "scripts/case_diverted.npz",
        args[1] if len(args) > 1 else "scripts/couple_torax.png",
    )
