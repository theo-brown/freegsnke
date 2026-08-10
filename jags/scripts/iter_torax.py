"""Stage 3: run TORAX's ITER hybrid scenario on three geometries.

Same physics config -- ``torax.examples.iterhybrid_predictor_corrector``, which
is the scenario ``iterhybrid_cocos11.eqdsk`` belongs to -- with only the geometry
swapped:

* ``original``   TORAX's own eqdsk parser on ``iterhybrid_cocos11.eqdsk``.
* ``freegsnke``  TORAX's own eqdsk parser on the geqdsk that ``iter_inverse.py``
                 wrote from the FreeGSNKE forward solve.
* ``jags``       ``jags.torax_bridge`` on the psi that ``iter_compare.py``
                 solved, never touching a file format.

The first two differ only by the equilibrium; the third differs by the
equilibrium *and* the route into TORAX, so ``jags`` vs ``freegsnke`` is the
end-to-end question and ``original`` is the yardstick for how much a different
equilibrium moves the transport answer at all.

``n_surfaces``, ``last_surface_factor`` and ``Ip_from_parameters`` are held
identical across all three; without that the runs differ for reasons that have
nothing to do with the equilibrium.

Runs in the TORAX venv, which also has jags on the path.

Usage:
    PYTHONPATH=. /home/user/torax/.venv/bin/python scripts/iter_torax.py [out.png]
"""

import copy
import pathlib
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ORIGINAL = "iterhybrid_cocos11.eqdsk"

N_SURFACES = 60
# 0.85, not the 0.95 the MAST-U coupling uses. The FreeGSNKE and jags ITER
# equilibria are *limited*, so their outermost surfaces run into the limiter and
# TORAX's contour tracer stops producing closed contours: at 0.95 and 0.90 it
# fails on surfaces 55-59 of 60 with "Volumes are not monotonically increasing".
# The same value is used for all three geometries -- including the CHEASE one,
# which is happy at 0.95 -- because a comparison in which the runs traced
# different fractions of the plasma would not be measuring the equilibrium.
LAST_SURFACE_FACTOR = 0.85
IP_FROM_PARAMETERS = True
T_FINAL = 5.0

COLOURS = {"original": "0.25", "freegsnke": "C1", "jags": "C0"}
STYLES = {"original": "-", "freegsnke": "--", "jags": ":"}


def base_config():
    from torax.examples import iterhybrid_predictor_corrector as ex

    cfg = copy.deepcopy(ex.CONFIG)
    cfg["numerics"]["t_final"] = T_FINAL
    return cfg


def eqdsk_config(geometry_file, cocos, geometry_directory=None):
    cfg = base_config()
    cfg["geometry"] = dict(
        geometry_type="eqdsk", geometry_file=geometry_file, cocos=cocos,
        Ip_from_parameters=IP_FROM_PARAMETERS, n_surfaces=N_SURFACES,
        last_surface_factor=LAST_SURFACE_FACTOR,
    )
    if geometry_directory is not None:
        cfg["geometry"]["geometry_directory"] = geometry_directory
    return cfg


def jags_provider(npz, face_centers):
    """A ``GeometryProvider`` from the jags forward solve, via ``torax_bridge``."""
    from jags import profiles, reach, torax_bridge
    from jags.fsa import make_flux_surface_averager
    from jags.grid import Grid

    d = np.load(npz)
    limiter = np.stack([d["limiter_R"], d["limiter_Z"]], axis=-1)
    grid = Grid(
        float(d["Rmin"]), float(d["Rmax"]), float(d["Zmin"]), float(d["Zmax"]),
        int(d["nR"]), int(d["nZ"]), limiter,
    )
    psi = jnp.asarray(d["psi"])
    profile = profiles.topeol(
        psi_axis=float(d["ref_psi_axis"]), psi_bndry=float(d["ref_psi_bndry"]),
        L=float(d["L"]), Beta0=float(d["Beta0"]),
        Raxis=float(d["profile_Raxis"]), fvac=float(d["fvac"]),
        alpha_m=float(d["alpha_m"]), alpha_n=int(d["alpha_n"]),
    )
    n_rho = len(face_centers) - 1
    if not np.allclose(face_centers, np.linspace(0.0, 1.0, n_rho + 1)):
        raise ValueError("the bridge assumes a uniform rho grid, which this is not")

    inter = torax_bridge.build_intermediates(
        grid, make_flux_surface_averager(grid), psi, profile.F,
        float(d["psi_axis"]), float(d["psi_bndry"]),
        float(d["R_axis"]), float(d["Z_axis"]),
        n_surfaces=N_SURFACES,
        label=reach.make_reachability(grid, n_samples=64, beta_norm=2e5)(psi),
        n_rho=n_rho, last_surface_factor=LAST_SURFACE_FACTOR,
        Ip_from_parameters=IP_FROM_PARAMETERS,
    )
    return torax_bridge.geometry_provider(torax_bridge.build_geometry(inter))


def run(cfg, provider=None):
    """Run TORAX, optionally on a geometry provider the config cannot express.

    ``run_simulation`` builds the provider from the config, so injecting one
    means assembling the step function first. The initial state is built from
    ``step_fn.geometry_provider`` too, hence the patch before that call rather
    than after.
    """
    from torax._src.orchestration import (
        initial_state as initial_state_lib,
        run_loop,
        run_simulation,
    )
    from torax._src.output_tools import output
    from torax._src.torax_pydantic import model_config

    tc = model_config.ToraxConfig.from_dict(cfg)
    step_fn = run_simulation.make_step_fn(tc)
    if provider is not None:
        step_fn._geometry_provider = provider  # noqa: SLF001

    t0 = time.time()
    state0, pp0 = initial_state_lib.get_initial_state_and_post_processed_outputs(
        step_fn=step_fn
    )
    history, pp_history, err = run_loop.run_loop(
        state0, pp0, step_fn, progress_bar=False
    )
    wall = time.time() - t0
    sh = output.StateHistory(history, pp_history, err, tc)
    return sh.simulation_output_to_xr(), err, wall


def build_runs(case_npz, jags_npz):
    from torax._src.torax_pydantic import model_config

    runs = {}
    cfg = eqdsk_config(ORIGINAL, cocos=11)
    runs["original"] = (cfg, None)

    gfile = pathlib.Path(str(case_npz).replace(".npz", ".geqdsk"))
    if gfile.exists():
        runs["freegsnke"] = (
            eqdsk_config(gfile.name, cocos=7, geometry_directory=str(gfile.parent)),
            None,
        )
    else:
        print(f"skipping freegsnke: {gfile} not found (run iter_inverse.py)")

    if pathlib.Path(jags_npz).exists():
        cfg = base_config()
        # The geometry block is a placeholder: it is replaced wholesale by the
        # injected provider, but ToraxConfig still has to validate.
        cfg["geometry"] = dict(
            geometry_type="eqdsk", geometry_file=ORIGINAL, cocos=11,
            Ip_from_parameters=IP_FROM_PARAMETERS, n_surfaces=N_SURFACES,
            last_surface_factor=LAST_SURFACE_FACTOR,
        )
        faces = model_config.ToraxConfig.from_dict(cfg).geometry.get_face_centers()
        runs["jags"] = (cfg, jags_provider(jags_npz, np.asarray(faces)))
    else:
        print(f"skipping jags: {jags_npz} not found (run iter_compare.py)")
    return runs


PROFILES = [
    ("T_i", "keV", r"$T_i$"),
    ("T_e", "keV", r"$T_e$"),
    ("n_e", "m$^{-3}$", r"$n_e$"),
    ("q", "", "q"),
    ("j_total", "A/m$^2$", r"$j_\phi$"),
    ("pressure_total", "Pa", "p"),
]
SCALARS = [
    ("P_fusion_total", "W", "fusion power"),
    ("W_thermal_total", "J", "thermal energy"),
    ("I_bootstrap", "A", "bootstrap current"),
    ("q95", "", r"$q_{95}$"),
]


def get(tree, name):
    for group in ("profiles", "scalars", "numerics"):
        if group in tree.children and name in tree[group]:
            return tree[group][name]
    raise KeyError(name)


def report(results):
    print(f"\n{'quantity':<22} " + "".join(f"{k:>14}" for k in results) +
          f"{'jags vs fgnke':>15}")
    ref = "freegsnke" if "freegsnke" in results else None
    for name, _, _ in SCALARS:
        row = {}
        for k, (tree, _, _) in results.items():
            try:
                row[k] = float(np.asarray(get(tree, name))[-1])
            except KeyError:
                row[k] = np.nan
        line = f"  {name:<20} " + "".join(f"{row[k]:>14.5g}" for k in results)
        if ref and "jags" in row and np.isfinite(row[ref]) and row[ref] != 0:
            line += f"{abs(row['jags'] - row[ref]) / abs(row[ref]):>15.2e}"
        print(line)

    print(f"\n{'final profile':<22} " + f"{'rel L2 vs freegsnke':>22}")
    if ref:
        for name, _, _ in PROFILES:
            try:
                b = np.asarray(get(results[ref][0], name))[-1]
            except KeyError:
                continue
            for k in results:
                if k == ref:
                    continue
                a = np.asarray(get(results[k][0], name))[-1]
                e = np.linalg.norm(a - b) / np.linalg.norm(b)
                print(f"  {name:<20} {k:>10} {e:>11.2e}")


def plot(results, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("TORAX ITER hybrid, identical physics config, three geometries",
                 fontsize=13, y=0.998)

    for ax, (name, unit, tex) in zip(axes.flat[:6], PROFILES):
        for k, (tree, _, _) in results.items():
            try:
                v = get(tree, name)
            except KeyError:
                continue
            rho = np.asarray(v.coords[v.dims[-1]])
            ax.plot(rho, np.asarray(v)[-1], STYLES[k], color=COLOURS[k], lw=1.8,
                    label=k)
        ax.set_xlabel(r"$\hat\rho$"), ax.set_ylabel(f"{tex} [{unit}]" if unit else tex)
        ax.set_title(f"{tex} at t = {T_FINAL} s"), ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=9)

    for ax, (name, unit, title) in zip(axes.flat[6:10], SCALARS):
        for k, (tree, _, _) in results.items():
            try:
                v = get(tree, name)
            except KeyError:
                continue
            ax.plot(np.asarray(v.coords["time"]), np.asarray(v), STYLES[k],
                    color=COLOURS[k], lw=1.8, label=k)
        ax.set_xlabel("t [s]"), ax.set_ylabel(f"[{unit}]" if unit else "")
        ax.set_title(title), ax.grid(alpha=0.3)

    # Difference panels: the whole point is how far apart the transport answers
    # end up, which is invisible on axes spanning the profile itself.
    ref = "freegsnke" if "freegsnke" in results else next(iter(results))
    for ax, (name, _, tex) in zip(axes.flat[10:12], PROFILES[:2]):
        try:
            b = np.asarray(get(results[ref][0], name))[-1]
        except KeyError:
            continue
        for k, (tree, _, _) in results.items():
            if k == ref:
                continue
            v = get(tree, name)
            rho = np.asarray(v.coords[v.dims[-1]])
            ax.plot(rho, np.asarray(v)[-1] - b, STYLES[k], color=COLOURS[k], lw=1.8,
                    label=k)
        ax.axhline(0, color=COLOURS[ref], lw=1.2)
        ax.set_xlabel(r"$\hat\rho$"), ax.set_ylabel(f"$\\Delta${tex}")
        ax.set_title(f"{tex} $-$ {ref}"), ax.grid(alpha=0.3), ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")


def main(out_path=None, case_npz=None):
    out_path = out_path or HERE / "iter_torax.png"
    case_npz = case_npz or HERE / "case_iter.npz"
    jags_npz = str(case_npz).replace(".npz", "_jags.npz")

    results = {}
    for name, (cfg, provider) in build_runs(case_npz, jags_npz).items():
        print(f"running TORAX on the {name} geometry ...")
        tree, err, wall = run(cfg, provider)
        print(f"  {err.name if hasattr(err, 'name') else err} in {wall:.1f} s, "
              f"{len(np.asarray(tree['profiles']['T_e'].coords['time']))} steps")
        results[name] = (tree, err, wall)
    report(results)
    plot(results, out_path)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else None, args[1] if len(args) > 1 else None)
