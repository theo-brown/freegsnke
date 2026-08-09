"""Free-boundary Grad-Shafranov residual and exact Newton solver.

The residual follows FreeGSNKE's arrangement (``GSstaticsolver.py:185-213``):

    F(psi_p) = psi_p - A^-1 b(psi_p)

with ``A`` the fourth-order Delstar operator carrying identity rows on the
domain edge, and ``b`` holding ``-mu0 R Jtor`` in the interior and the
Green's-function free-boundary values on the edge. Writing it this way makes the
Jacobian ``I - A^-1 db/dpsi``, a compact perturbation of the identity, which is
much better conditioned than the raw operator form.

What differs from FreeGSNKE is that ``Jtor`` here is a *pointwise* function of
psi. Profiles are given on unnormalised psi with compact support (see
``profiles.py``), so the plasma region is the level set ``{psi > psi_edge}``
and falls out of the solution rather than being detected. Nothing in the
residual needs the magnetic axis, the boundary flux or a core mask, so the
residual is smooth by construction and with no smoothing parameters to bias it.
Critical points are still needed for diagnostics -- see ``critical.py`` -- but
only after the solve, where non-differentiability is harmless.

For a diverted plasma the level set is not enough: it also contains lobes beyond
a null. Pass ``reachability`` (see ``reach.py``) to exclude those.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import jacobian, operators, profiles
from .grid import Grid

MU0 = operators.MU0


class Machine(NamedTuple):
    """Precomputed, psi-independent geometry for one device and grid."""

    grid: Grid
    Ainv: jnp.ndarray  # (N, N) inverse Delstar with identity boundary rows
    G_bnd: jnp.ndarray  # (n_bnd, N) plasma current -> flux on the domain edge
    G_coil: jnp.ndarray  # (N, n_coil) coil current -> flux on the grid
    vessel_mask: jnp.ndarray  # (nR, nZ) fixed region where plasma is allowed
    R: jnp.ndarray  # (nR, nZ)
    bnd_flat: jnp.ndarray  # (n_bnd,) flat indices of the domain edge


def build_machine(grid: Grid, coil_RZ=None) -> Machine:
    """Assemble all geometry-only quantities once.

    Everything here is NumPy/SciPy: the Green's functions need elliptic
    integrals, which JAX does not provide, and none of it depends on psi.

    ``coil_RZ`` may be None when the vacuum flux is supplied directly rather
    than computed from filament positions, as in the FreeGSNKE cross-check.
    """
    bnd = grid.boundary_indices
    G_coil = (
        jnp.zeros((grid.N, 0))
        if coil_RZ is None
        else jnp.asarray(operators.coil_greens_matrix(grid, coil_RZ))
    )
    return Machine(
        grid=grid,
        Ainv=jnp.asarray(operators.inverse_delstar(grid)),
        G_bnd=jnp.asarray(operators.boundary_greens_matrix(grid)),
        G_coil=G_coil,
        vessel_mask=jnp.asarray(grid.limiter_mask(), dtype=float),
        R=jnp.asarray(grid.R),
        bnd_flat=jnp.asarray(bnd[:, 0] * grid.nZ + bnd[:, 1]),
    )


def coil_flux(machine: Machine, currents) -> jnp.ndarray:
    """Flux on the grid from the coil currents, (nR, nZ)."""
    shape = (machine.grid.nR, machine.grid.nZ)
    return (machine.G_coil @ jnp.asarray(currents)).reshape(shape)


def make_residual(machine: Machine, profile, psi_coil, Ip=None, reachability=None):
    """Build ``F(psi_p)`` for a flattened plasma flux vector.

    ``reachability`` optionally replaces psi by its running minimum along the ray
    from the magnetic axis before the profiles are evaluated, which excludes
    high-flux lobes lying beyond a null. See ``reach.py``. Needed for diverted
    plasmas; a no-op for limited ones.
    """
    grid = machine.grid
    shape = (grid.nR, grid.nZ)
    dA = grid.dA

    def current(psi_p_flat):
        psi = psi_p_flat.reshape(shape) + psi_coil
        if reachability is not None:
            psi = reachability(psi)
        return profiles.jtor(
            profile, machine.R, psi, machine.vessel_mask, Ip=Ip, dA=dA
        )

    def rhs(jtor):
        """Right-hand side: source in the interior, free boundary on the edge."""
        b = (-MU0 * machine.R * jtor).reshape(-1)
        edge = machine.G_bnd @ jtor.reshape(-1)
        return b.at[machine.bnd_flat].set(edge)

    def residual(psi_p_flat):
        return psi_p_flat - machine.Ainv @ rhs(current(psi_p_flat))

    return residual, current


def initial_guess(machine: Machine, centre, radii, Ip) -> jnp.ndarray:
    """Plasma flux from a prescribed elliptical current blob carrying ``Ip``.

    A starting point with no plasma is a genuine fixed point of the residual --
    zero current gives zero flux gives zero current -- so the iteration must be
    started with a plasma already present.
    """
    grid = machine.grid
    R, Z = jnp.asarray(grid.R), jnp.asarray(grid.Z)
    r2 = ((R - centre[0]) / radii[0]) ** 2 + ((Z - centre[1]) / radii[1]) ** 2
    blob = jnp.where(r2 < 1.0, 1.0 - r2, 0.0) * machine.vessel_mask
    blob = blob * (Ip / (jnp.sum(blob) * grid.dA))

    b = (-MU0 * R * blob).reshape(-1)
    edge = machine.G_bnd @ blob.reshape(-1)
    return machine.Ainv @ b.at[machine.bnd_flat].set(edge)


class Result(NamedTuple):
    psi_p: jnp.ndarray  # (nR, nZ) converged plasma flux
    psi: jnp.ndarray  # (nR, nZ) total flux
    jtor: jnp.ndarray  # (nR, nZ) toroidal current density
    residual_history: np.ndarray
    converged: bool


def make_solver(
    machine: Machine,
    profile,
    psi_coil,
    Ip=None,
    jacobian_mode="matrix_free",
    reachability=None,
):
    """Compile once, solve many times.

    Everything jitted is built here rather than inside the iteration, so a sweep
    over initial guesses or coil currents pays compilation only on the first
    call. ``solve`` is a one-shot wrapper around this and recompiles each time.

    Both ``jacobian_mode`` values give an exact Newton step and the same
    iterates; they differ only in cost.

    ``"matrix_free"`` (default)
        GMRES on the Jacobian's action, nothing assembled. See ``jacobian.py``.
    ``"jacfwd"``
        Dense Jacobian in memory-bounded column blocks, then a direct solve.
        Assumption-free but O(N^3); kept as the reference the tests check
        ``matrix_free`` against.
    """
    residual_fn, current_fn = make_residual(
        machine, profile, psi_coil, Ip, reachability
    )
    residual = jax.jit(residual_fn)
    current = jax.jit(current_fn)
    shape = (machine.grid.nR, machine.grid.nZ)

    if jacobian_mode == "matrix_free":
        newton_step = jacobian.make_matrix_free_step(residual_fn)
    elif jacobian_mode == "jacfwd":
        linsolve = jax.jit(jnp.linalg.solve)

        def newton_step(psi, F):
            return linsolve(jacobian.dense_jacobian(residual_fn, psi), -F)
    else:
        raise ValueError(f"unknown jacobian_mode {jacobian_mode!r}")

    def run(psi_p0, tol=1e-11, max_newton=30, n_picard=3, picard_omega=0.5,
            verbose=False):
        psi = jnp.asarray(psi_p0).reshape(-1)

        # A Picard step is the same residual under-relaxed: psi <- psi - omega F.
        for _ in range(n_picard):
            psi = psi - picard_omega * residual(psi)

        hist = []
        converged = False
        for it in range(max_newton):
            F = residual(psi)
            nF = float(jnp.linalg.norm(F))
            scale = float(jnp.linalg.norm(psi)) + 1e-300
            hist.append(nF / scale)
            if verbose:
                print(f"  newton {it:2d}  |F|/|psi| = {nF / scale:.3e}")
            if nF / scale < tol:
                converged = True
                break

            step = newton_step(psi, F)

            # Armijo backtracking: a full Newton step can overshoot while the
            # plasma boundary is still moving between iterations.
            t = 1.0
            for _ in range(20):
                if float(jnp.linalg.norm(residual(psi + t * step))) <= (
                    1 - 1e-4 * t
                ) * nF:
                    break
                t *= 0.5
            psi = psi + t * step

        psi2d = psi.reshape(shape)
        return Result(
            psi_p=psi2d,
            psi=psi2d + psi_coil,
            jtor=current(psi),
            residual_history=np.array(hist),
            converged=converged,
        )

    return run


def solve(
    machine: Machine,
    profile,
    psi_coil,
    psi_p0,
    Ip=None,
    tol=1e-11,
    max_newton=30,
    n_picard=3,
    picard_omega=0.5,
    verbose=False,
    jacobian_mode="matrix_free",
    reachability=None,
):
    """One-shot damped Picard warm-up followed by exact Newton.

    Convenience wrapper; this recompiles on every call. Use ``make_solver`` when
    solving more than once with the same machine and profile.
    """
    return make_solver(
        machine, profile, psi_coil, Ip, jacobian_mode, reachability
    )(psi_p0, tol, max_newton, n_picard, picard_omega, verbose)
