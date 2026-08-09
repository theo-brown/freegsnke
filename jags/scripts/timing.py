"""How jags runtime and memory scale with grid size.

Times each stage separately, because they scale differently and the binding
constraint changes as the grid grows:

* ``build_machine`` builds ``Ainv``. It is *not* a dense inversion: ``splu``
  factorises the sparse Delstar in O(N^1.5) and the cost is the N
  back-substitutions against it, O(N log N) each, so **O(N^2 log N) = O(n^4 log
  n)** -- measured exponent 4.2, not the 6 a dense inverse would give. Storage
  is the honest O(N^2) = **O(n^4)**, and that is what runs out first.
* one residual evaluation. The expected bottleneck is the dense ``Ainv @ b``
  matvec, O(N^2) = O(n^4). It is **not**: measured at 97x97 the residual is
  45 ms, of which the reachability is 27 ms and the matvec 14 ms. The
  ``--breakdown`` mode prints that split.
* the reachability alone. Its *work* is only O(n_samples n^2), but it measures
  n^4.1 between 65 and 97 -- ``n_samples`` scattered bicubic gathers per grid
  point, so it is cache behaviour rather than flops. Being the largest single
  term with the smallest asymptotic work, it is the obvious thing to optimise.
* one matrix-free Newton step, which is GMRES on the JVP: a small multiple of
  the residual cost, set by the Krylov iteration count.
* the flux-surface averages, O(n^2 n_levels) with no linear algebra at all.

Run with ``--max N`` to stop before the dense inverse exhausts memory.

Usage:
    /home/user/jaxgs/.venv/bin/python scripts/timing.py [--max 97] [--breakdown]
"""

import gc
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jags import jacobian, profiles, reach, solver  # noqa: E402
from jags.fsa import make_flux_surface_averager  # noqa: E402
from jags.grid import Grid  # noqa: E402

BOX = np.array([[0.35, -0.85], [1.55, -0.85], [1.55, 0.85], [0.35, 0.85]])
COILS = np.array(
    [[0.25, -1.05], [0.25, 1.05], [1.75, -1.05], [1.75, 1.05], [1.9, -0.4], [1.9, 0.4]]
)
CURRENTS = np.array([3.0e5, 3.0e5, 1.6e5, 1.6e5, -1.0e5, -1.0e5])
IP = 4.0e5
SIZES = [25, 33, 41, 49, 65, 81, 97, 113]


def timeit(fn, repeat=3):
    """Best of ``repeat``, with the result forced to be materialised."""
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    return best


def run(n):
    grid = Grid(0.15, 2.0, -1.2, 1.2, n, n, BOX)
    N = n * n

    t_build = timeit(lambda: solver.build_machine(grid, COILS), repeat=1)
    machine = solver.build_machine(grid, COILS)
    ainv_gb = machine.Ainv.size * 8 / 2**30

    prof = profiles.compact(psi_edge=0.0, psi_scale=0.15, p0=8.0e3, fvac=0.5, f1=0.05)
    psi_coil = solver.coil_flux(machine, CURRENTS)
    psi0 = solver.initial_guess(machine, (0.9, 0.0), (0.45, 0.65), IP)
    reachability = reach.make_reachability(grid, n_samples=32, beta_norm=2e5)

    residual, _ = solver.make_residual(
        machine, prof, psi_coil, IP, reachability=reachability
    )
    residual = jax.jit(residual)
    x = psi0.reshape(-1)
    jax.block_until_ready(residual(x))  # compile
    t_res = timeit(lambda: residual(x))

    reach_j = jax.jit(reachability)
    psi2 = psi0.reshape(n, n)
    jax.block_until_ready(reach_j(psi2))
    t_reach = timeit(lambda: reach_j(psi2))

    step = jax.jit(jacobian.make_matrix_free_step(residual))
    F = residual(x)
    jax.block_until_ready(step(x, F))
    t_step = timeit(lambda: step(x, F))

    # Full solve, compile and steady-state separately.
    t0 = time.perf_counter()
    run_solver = solver.make_solver(
        machine, prof, psi_coil, Ip=IP, reachability=reachability
    )
    res = run_solver(psi0)
    jax.block_until_ready(res.psi_p)
    t_compile = time.perf_counter() - t0
    t_solve = timeit(lambda: run_solver(psi0).psi_p, repeat=2)

    _, surfaces, _ = make_flux_surface_averager(grid)
    psi = np.asarray(res.psi)
    levels = jnp.asarray(np.linspace(psi.max() * 0.9, psi.max() * 0.1, 25))
    surf_j = jax.jit(lambda p: surfaces(p, levels, float(psi.max()), 0.0))
    jax.block_until_ready(surf_j(jnp.asarray(psi)))
    t_fsa = timeit(lambda: surf_j(jnp.asarray(psi)))

    out = dict(
        n=n, N=N, ainv_gb=ainv_gb, build=t_build, residual=t_res, reach=t_reach,
        step=t_step, compile=t_compile - t_solve, solve=t_solve, fsa=t_fsa,
        # Per-iteration cost is the convergence-independent number; `solve`
        # depends on how many Newton steps this particular case happens to take.
        per_iter=t_res + t_step,
        newton=len(res.residual_history) - 1, converged=bool(res.converged),
    )
    del machine, res, run_solver, step, residual
    gc.collect()
    return out


def breakdown(sizes):
    """Split the residual into reachability, dense matvec, and the rest.

    The matvec turns out to run at the machine's streaming bandwidth -- it
    reads the whole of ``Ainv`` once -- so its cost is predictable as
    ``8 n^4 bytes / bandwidth`` and nothing about it is going to improve
    without making ``Ainv`` not dense.
    """
    print(f"{'n':>5} {'residual':>10} {'reach':>10} {'Ainv@v':>10} "
          f"{'other':>10} {'Ainv':>8} {'GB/s':>7}")
    for n in sizes:
        grid = Grid(0.15, 2.0, -1.2, 1.2, n, n, BOX)
        machine = solver.build_machine(grid, COILS)
        prof = profiles.compact(
            psi_edge=0.0, psi_scale=0.15, p0=8.0e3, fvac=0.5, f1=0.05
        )
        psi_coil = solver.coil_flux(machine, CURRENTS)
        psi0 = solver.initial_guess(machine, (0.9, 0.0), (0.45, 0.65), IP)
        reachability = reach.make_reachability(grid, n_samples=32, beta_norm=2e5)
        res, _ = solver.make_residual(
            machine, prof, psi_coil, IP, reachability=reachability
        )
        res = jax.jit(res)
        reach_j = jax.jit(reachability)
        matvec = jax.jit(lambda v: machine.Ainv @ v)
        x, psi2 = psi0.reshape(-1), psi0.reshape(n, n)
        v = jnp.asarray(np.zeros(grid.N) + 1.0)
        for f in (lambda: res(x), lambda: reach_j(psi2), lambda: matvec(v)):
            jax.block_until_ready(f())
        t_r, t_re = timeit(lambda: res(x), 5), timeit(lambda: reach_j(psi2), 5)
        t_m = timeit(lambda: matvec(v), 5)
        gb = machine.Ainv.size * 8 / 2**30
        print(f"{n:>5} {t_r * 1e3:>9.1f}m {t_re * 1e3:>9.1f}m {t_m * 1e3:>9.1f}m "
              f"{(t_r - t_re - t_m) * 1e3:>9.1f}m {gb:>7.2f}G {gb / t_m:>7.1f}")
        del machine
        gc.collect()


def fit(ns, ts):
    """Least-squares slope of log t against log n, i.e. the exponent p in n^p."""
    ok = np.asarray(ts) > 0
    if ok.sum() < 2:
        return np.nan
    return np.polyfit(np.log(np.asarray(ns)[ok]), np.log(np.asarray(ts)[ok]), 1)[0]


def main(nmax):
    rows = []
    cols = ["build", "residual", "reach", "step", "per_iter", "compile",
            "solve", "fsa"]
    hdr = (f"{'n':>5} {'N':>7} {'Ainv':>8} " +
           " ".join(f"{c:>9}" for c in cols) + f" {'newton':>7}")
    print(hdr)
    print("-" * len(hdr))
    for n in SIZES:
        if n > nmax:
            break
        try:
            r = run(n)
        except (MemoryError, RuntimeError) as exc:
            print(f"{n:>5}  failed: {type(exc).__name__}: {str(exc)[:60]}")
            break
        rows.append(r)
        print(f"{r['n']:>5} {r['N']:>7} {r['ainv_gb']:>7.2f}G " +
              " ".join(f"{r[c]:>9.3f}" for c in cols) +
              f" {r['newton']:>7}" + ("" if r["converged"] else "  (no conv)"))

    if len(rows) >= 3:
        ns = [r["n"] for r in rows]
        print("\nfitted exponent p in n^p (all sizes / largest three):")
        for c in cols + ["ainv_gb"]:
            ts = [r[c] for r in rows]
            note = ""
            if c == "solve" and not all(r["converged"] for r in rows):
                note = "   <- meaningless: Newton count varies, see per_iter"
            elif max(ts) < 0.05:
                note = "   <- sub-50ms, dominated by dispatch not flops"
            print(f"  {c:<10} {fit(ns, ts):>6.2f}   "
                  f"{fit(ns[-3:], ts[-3:]):>6.2f}{note}")
        print("\nThe synthetic case here is deliberately unconfined (see "
              "tests/conftest.py), so\nNewton sometimes hits the iteration cap; "
              "per_iter is the number to read.")


if __name__ == "__main__":
    nmax = 10**9
    if "--max" in sys.argv:
        nmax = int(sys.argv[sys.argv.index("--max") + 1])
    if "--breakdown" in sys.argv:
        breakdown([n for n in SIZES if n <= nmax])
    else:
        main(nmax)
