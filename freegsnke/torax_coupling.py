"""
Loose coupling of FreeGSNKE with the TORAX core transport code.

In loose coupling the two codes are advanced alternately over each coupling
interval and iterated to convergence, as opposed to tight coupling where the
transport and equilibrium equations are solved together as one system. Within
a coupling interval [t, t + dt]:

    1. TORAX provides the plasma profiles that source the Grad-Shafranov
       equation, p'(psi) and FF'(psi), together with the plasma current, as an
       IMAS `equilibrium` IDS (see `torax.experimental.torax_state_to_imas_equilibrium`).
    2. FreeGSNKE solves the free-boundary equilibrium for those profiles (and
       the coil currents at t + dt) and returns an IMAS `equilibrium` IDS
       describing the new geometry (see `freegsnke.imas_read_write`).
    3. TORAX builds its flux-surface-averaged geometry from that IDS and takes
       a (jitted) transport step from t to t + dt, using the geometry at t and
       at t + dt.
    4. Steps 1-3 are repeated with the updated profiles at t + dt until the
       exchanged p' and FF' profiles stop changing (fixed-point iteration,
       optionally under-relaxed), after which the interval is accepted.

The IMAS `equilibrium` IDS is the interchange format in both directions.

TORAX is an optional dependency of FreeGSNKE: this module can only be used when
`torax` is importable.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

import dataclasses
import time as _time

import numpy as np
from freegs4e.gradshafranov import mu0

from . import GSstaticsolver, imas_read_write
from .jtor_update import GeneralPprimeFFprime

try:
    import jax.numpy as jnp
    import torax
    from torax import experimental as torax_experimental

    _TORAX_IMPORT_ERROR = None
except ImportError as error:  # pragma: no cover - exercised only without torax
    torax = None
    torax_experimental = None
    jnp = None
    _TORAX_IMPORT_ERROR = error


# smallest coupling interval considered non-zero (fraction of coupling_dt)
_MIN_RELATIVE_DT = 1e-8
# floor for the profile norms used to make the convergence residual relative
_RESIDUAL_NORM_FLOOR = 1e-300


def _require_torax():
    """Raises an informative error if TORAX is not installed."""
    if torax is None:
        raise ImportError(
            "The FreeGSNKE-TORAX coupling requires the `torax` package "
            "(https://github.com/google-deepmind/torax) to be installed."
        ) from _TORAX_IMPORT_ERROR


def default_psi_n_grid(n_points=129, psi_n_min=0.01, psi_n_max=0.99):
    """
    Normalised poloidal flux grid used when writing FreeGSNKE equilibria to an
    IDS for TORAX. Points are spaced uniformly in sqrt(psi_n), which is
    approximately uniform in the normalised toroidal flux coordinate used by
    TORAX, so that the near-axis region is resolved.

    Parameters
    ----------
    n_points : int
        Number of flux surfaces.
    psi_n_min : float
        Innermost surface (must be > 0; flux surfaces cannot be traced on axis).
    psi_n_max : float
        Outermost surface (must be < 1; the separatrix of a diverted plasma
        cannot be used for flux surface averages).

    Returns
    -------
    np.array
        Strictly increasing normalised flux values in (0, 1).
    """
    return np.linspace(np.sqrt(psi_n_min), np.sqrt(psi_n_max), n_points) ** 2


class StaticEquilibriumSolver:
    """
    Free-boundary equilibrium provider for the loose coupling, based on
    FreeGSNKE static forward Grad-Shafranov solves.

    At each request the p' and FF' profiles are read from the `equilibrium` IDS
    supplied by TORAX into a `GeneralPprimeFFprime` profile object, the active
    coil currents are (optionally) updated to their values at the requested
    time, the static forward problem is solved (warm-started from the previous
    solution held in `eq`) and the result is written to a new `equilibrium` IDS.

    Any object with the same `solve(time, equilibrium_ids)` and
    `initial_equilibrium_ids(time)` interface can be used in place of this
    class with `run_loose_coupling`, e.g. to include a shape controller or an
    inverse solve for the coil currents.
    """

    def __init__(
        self,
        eq,
        profiles,
        coil_currents=None,
        solver=None,
        psi_n=None,
        fvac=None,
        target_relative_tolerance=1e-6,
        solver_kwargs=None,
        Raxis=1.0,
        Ip_logic=True,
        interpolator="univariate_spline",
    ):
        """
        Parameters
        ----------
        eq : freegsnke.equilibrium_update.Equilibrium
            Equilibrium object holding the machine, grid and the initial
            (already solved) plasma flux used to warm-start the first solve.
        profiles : freegsnke.jtor_update profile object
            Profile object used to obtain the initial equilibrium `eq`. If it is
            not a `GeneralPprimeFFprime` instance, one is created from the first
            IDS received from TORAX; otherwise it is updated in place.
        coil_currents : callable, optional
            Function `coil_currents(time) -> dict` mapping active coil labels to
            currents [A] at the requested time. If None, the coil currents set
            on `eq.tokamak` are used throughout.
        solver : freegsnke.GSstaticsolver.NKGSsolver, optional
            Static solver instance. Created for `eq` if not provided.
        psi_n : np.array, optional
            Normalised flux grid used when writing the equilibrium IDS. Defaults
            to `default_psi_n_grid()`.
        fvac : float, optional
            Vacuum field function R*Btor [T m] for the coupled profiles. Defaults
            to `profiles.fvac()`.
        target_relative_tolerance : float
            Relative tolerance requested from the static solver.
        solver_kwargs : dict, optional
            Additional keyword arguments passed to `NKGSsolver.solve`.
        Raxis : float
            Radial scaling parameter of `GeneralPprimeFFprime`.
        Ip_logic : bool
            If True the current density is renormalised to the TORAX plasma
            current exactly.
        interpolator : str
            Interpolator used by `GeneralPprimeFFprime`.
        """
        self.eq = eq
        self.profiles = profiles
        self.coil_currents = coil_currents
        self.solver = solver if solver is not None else GSstaticsolver.NKGSsolver(eq)
        self.psi_n = default_psi_n_grid() if psi_n is None else np.asarray(psi_n)
        self.fvac = profiles.fvac() if fvac is None else fvac
        self.target_relative_tolerance = target_relative_tolerance
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.Raxis = Raxis
        self.Ip_logic = Ip_logic
        self.interpolator = interpolator
        self.n_solves = 0
        # relative Grad-Shafranov residual reached by each static solve
        self.relative_changes = []

    def initial_equilibrium_ids(self, time=0.0):
        """
        Writes the current (initial) equilibrium to an `equilibrium` IDS.

        Parameters
        ----------
        time : float
            Time [s] assigned to the IDS time slice.

        Returns
        -------
        imas.ids_toplevel.IDSToplevel
            The `equilibrium` IDS of the current equilibrium.
        """
        return imas_read_write.write_equilibrium_to_ids(
            self.eq, self.profiles, psi_n=self.psi_n, time=time
        )

    def set_coil_currents(self, time):
        """Applies the coil currents at `time` to the tokamak, if prescribed."""
        if self.coil_currents is None:
            return
        for label, current in self.coil_currents(time).items():
            self.eq.tokamak.set_coil_current(coil_label=label, current_value=current)

    def solve(self, time, equilibrium_ids):
        """
        Solves the static forward free-boundary problem for the p' and FF'
        profiles in `equilibrium_ids` and the coil currents at `time`.

        Parameters
        ----------
        time : float
            Time [s] of the requested equilibrium.
        equilibrium_ids : imas.ids_toplevel.IDSToplevel
            `equilibrium` IDS containing `profiles_1d.psi_norm`,
            `profiles_1d.dpressure_dpsi`, `profiles_1d.f_df_dpsi` and
            `global_quantities.ip` (e.g. as written by TORAX).

        Returns
        -------
        imas.ids_toplevel.IDSToplevel
            `equilibrium` IDS describing the new FreeGSNKE equilibrium.
        """
        if isinstance(self.profiles, GeneralPprimeFFprime):
            imas_read_write.update_profiles_from_equilibrium_ids(
                self.profiles, equilibrium_ids
            )
        else:
            self.profiles = imas_read_write.profiles_from_equilibrium_ids(
                self.eq,
                equilibrium_ids,
                fvac=self.fvac,
                Raxis=self.Raxis,
                Ip_logic=self.Ip_logic,
                interpolator=self.interpolator,
            )
        self.set_coil_currents(time)
        self.solver.solve(
            eq=self.eq,
            profiles=self.profiles,
            constrain=None,
            target_relative_tolerance=self.target_relative_tolerance,
            verbose=False,
            **self.solver_kwargs,
        )
        self.n_solves += 1
        self.relative_changes.append(getattr(self.solver, "relative_change", np.nan))
        if not np.all(np.isfinite(self.eq.plasma_psi)):
            raise RuntimeError(
                "The FreeGSNKE static solve produced a non-finite plasma flux."
            )
        return imas_read_write.write_equilibrium_to_ids(
            self.eq, self.profiles, psi_n=self.psi_n, time=time
        )


def torax_geometry_from_ids(equilibrium_ids, torax_config, Ip_from_parameters=True):
    """
    Builds a TORAX `StandardGeometry` (on the radial mesh of `torax_config`)
    from an `equilibrium` IDS written by `freegsnke.imas_read_write`.

    Parameters
    ----------
    equilibrium_ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS.
    torax_config : torax.ToraxConfig
        TORAX configuration; only its geometry mesh settings (`n_rho` or
        `face_centers`, and `hires_factor`) are used.
    Ip_from_parameters : bool
        Passed to the TORAX IMAS geometry builder. If True the geometry's
        plasma current profile is rescaled to the current prescribed in the
        TORAX config (only relevant to the initial condition).

    Returns
    -------
    torax.Geometry
        The TORAX geometry.
    """
    _require_torax()
    geometry_config = torax_config.geometry
    configs = geometry_config.geometry_configs
    if isinstance(configs, dict):
        first_config = next(iter(configs.values())).config
    else:
        first_config = configs.config
    return torax_experimental.geometry.IMASConfig(
        equilibrium_object=equilibrium_ids,
        face_centers=np.asarray(geometry_config.get_face_centers()),
        hires_factor=first_config.hires_factor,
        Ip_from_parameters=Ip_from_parameters,
        explicit_convert=False,
    ).build_geometry()


def _check_finite_geometry(geometry):
    """Raises if any array in a TORAX geometry contains non-finite values."""
    bad = [
        field.name
        for field in dataclasses.fields(geometry)
        if isinstance(getattr(geometry, field.name), (np.ndarray, jnp.ndarray))
        and not np.all(np.isfinite(np.asarray(getattr(geometry, field.name))))
    ]
    if bad:
        raise RuntimeError(
            "The TORAX geometry built from the FreeGSNKE equilibrium IDS contains "
            f"non-finite values in: {', '.join(bad)}. This usually indicates a "
            "poorly converged or unphysical equilibrium (e.g. flux surfaces "
            "crossing the limiter)."
        )
    return geometry


def _exchanged_profiles(equilibrium_ids):
    """Returns (psi_n, dp/dpsi, FF', R0) from the first time slice of an IDS."""
    time_slice = equilibrium_ids.time_slice[0]
    profiles_1d = time_slice.profiles_1d
    return (
        np.asarray(profiles_1d.psi_norm, dtype=float),
        np.asarray(profiles_1d.dpressure_dpsi, dtype=float),
        np.asarray(profiles_1d.f_df_dpsi, dtype=float),
        float(equilibrium_ids.vacuum_toroidal_field.r0),
    )


def profile_residual(equilibrium_ids, previous_equilibrium_ids):
    """
    Relative change of the exchanged p' and FF' profiles between two
    `equilibrium` IDSs, used as the loose-coupling convergence measure.

    The two profiles enter the Grad-Shafranov equation through the toroidal
    current density Jtor = R p' + FF' / (mu0 R), so they are compared on a
    common current-density scale using the reference major radius R0 of the
    IDS (`vacuum_toroidal_field.r0`):

        a = R0 p',  b = FF' / (mu0 R0),
        residual = sqrt(||da||^2 + ||db||^2) / sqrt(||a||^2 + ||b||^2),

    with the old profiles interpolated onto the new normalised-flux grid. This
    avoids spurious large residuals when one of the two profiles (typically FF'
    for a plasma close to the force-free/diamagnetic balance) is small.

    Parameters
    ----------
    equilibrium_ids : imas.ids_toplevel.IDSToplevel
        Newest IDS from TORAX.
    previous_equilibrium_ids : imas.ids_toplevel.IDSToplevel
        IDS from the previous iteration.

    Returns
    -------
    float
        The residual.
    """
    psi_n, pprime, ffprime, R0 = _exchanged_profiles(equilibrium_ids)
    psi_n_old, pprime_old, ffprime_old, _ = _exchanged_profiles(
        previous_equilibrium_ids
    )
    a = R0 * pprime
    b = ffprime / (mu0 * R0)
    da = a - R0 * np.interp(psi_n, psi_n_old, pprime_old)
    db = b - np.interp(psi_n, psi_n_old, ffprime_old) / (mu0 * R0)
    norm = np.sqrt(np.sum(a**2) + np.sum(b**2))
    return float(
        np.sqrt(np.sum(da**2) + np.sum(db**2)) / max(norm, _RESIDUAL_NORM_FLOOR)
    )


def relax_profiles(equilibrium_ids, previous_equilibrium_ids, relaxation):
    """
    Under-relaxes the exchanged p' and FF' profiles in place,

        x <- relaxation * x_new + (1 - relaxation) * x_old,

    where the old profiles are interpolated onto the new normalised-flux grid.

    Parameters
    ----------
    equilibrium_ids : imas.ids_toplevel.IDSToplevel
        Newest IDS from TORAX (modified in place).
    previous_equilibrium_ids : imas.ids_toplevel.IDSToplevel
        IDS passed to the equilibrium solver in the previous iteration.
    relaxation : float
        Relaxation factor in (0, 1]; 1 leaves the new profiles unchanged.

    Returns
    -------
    imas.ids_toplevel.IDSToplevel
        The (same) relaxed IDS.
    """
    if relaxation == 1.0:
        return equilibrium_ids
    psi_n, pprime, ffprime, _ = _exchanged_profiles(equilibrium_ids)
    psi_n_old, pprime_old, ffprime_old, _ = _exchanged_profiles(
        previous_equilibrium_ids
    )
    profiles_1d = equilibrium_ids.time_slice[0].profiles_1d
    profiles_1d.dpressure_dpsi = relaxation * pprime + (1 - relaxation) * np.interp(
        psi_n, psi_n_old, pprime_old
    )
    profiles_1d.f_df_dpsi = relaxation * ffprime + (1 - relaxation) * np.interp(
        psi_n, psi_n_old, ffprime_old
    )
    return equilibrium_ids


@dataclasses.dataclass
class LooseCouplingResult:
    """
    Output of `run_loose_coupling`.

    Attributes
    ----------
    torax_output : xarray.DataTree
        TORAX simulation output (as returned by `torax.run_simulation`), with
        one entry per coupling time.
    torax_history : torax.StateHistory
        TORAX state history (states and post-processed outputs at each
        coupling time).
    equilibrium_ids : list
        FreeGSNKE `equilibrium` IDSs at each coupling time (including the
        initial one), i.e. the geometry TORAX used at that time.
    torax_equilibrium_ids : list
        TORAX-written `equilibrium` IDSs at each coupling time (the p' and FF'
        handed to FreeGSNKE).
    times : np.array
        Coupling times [s].
    iterations : np.array
        Number of coupling iterations used to reach each coupling time (the
        first entry counts the initial-condition iterations).
    residuals : list
        Per coupling time, the list of convergence residuals of each iteration.
    converged : np.array
        Whether the iteration converged to `tolerance` at each coupling time.
    sim_error : torax.SimError
        TORAX error state (NO_ERROR when the run completed).
    equilibria : list
        Copies of the FreeGSNKE equilibrium object at each coupling time, if
        requested (otherwise empty).
    """

    torax_output: object
    torax_history: object
    equilibrium_ids: list
    torax_equilibrium_ids: list
    times: np.ndarray
    iterations: np.ndarray
    residuals: list
    converged: np.ndarray
    sim_error: object
    equilibria: list = dataclasses.field(default_factory=list)


def _copy_equilibrium(equilibrium_solver):
    """Returns a copy of the solver's equilibrium object, if it exposes one."""
    eq = getattr(equilibrium_solver, "eq", None)
    if eq is None or not hasattr(eq, "create_auxiliary_equilibrium"):
        return None
    return eq.create_auxiliary_equilibrium()


def run_loose_coupling(
    torax_config,
    equilibrium_solver,
    coupling_dt,
    max_iterations=10,
    tolerance=1e-3,
    relaxation=0.5,
    initial_iterations=1,
    Ip_from_parameters=True,
    store_equilibria=False,
    verbose=True,
):
    """
    Runs a loosely coupled FreeGSNKE-TORAX simulation.

    See the module docstring for a description of the algorithm. The TORAX
    simulation runs from `torax_config.numerics.t_initial` to
    `torax_config.numerics.t_final` in coupling intervals of `coupling_dt`
    (TORAX may take several internal transport time steps within each interval,
    using its own time step calculator). The geometry section of `torax_config`
    is only used for the radial mesh: the geometry itself is provided by
    FreeGSNKE.

    Parameters
    ----------
    torax_config : torax.ToraxConfig
        TORAX configuration.
    equilibrium_solver : StaticEquilibriumSolver (or compatible)
        Provides `initial_equilibrium_ids(time)` and `solve(time, equilibrium_ids)`.
    coupling_dt : float
        Coupling interval [s].
    max_iterations : int
        Maximum number of equilibrium/transport iterations per coupling interval.
        With 1 iteration the scheme reduces to a staggered (explicit) coupling.
    tolerance : float
        Convergence tolerance on the relative change of the exchanged p' and FF'
        profiles between iterations (see `profile_residual`).
    relaxation : float
        Under-relaxation factor in (0, 1] applied to the exchanged profiles
        between iterations. The unrelaxed fixed-point iteration (1.0) tends to
        oscillate with a period of two iterations, since the near-axis current
        density seen by TORAX reacts strongly to small changes of the
        equilibrium; the default of 0.5 damps this and typically converges
        linearly.
    initial_iterations : int
        Number of iterations used to make the initial TORAX state and the
        initial FreeGSNKE equilibrium consistent before time stepping: the
        equilibrium is re-solved with TORAX's initial p' and FF' and the TORAX
        initial state is rebuilt on the new geometry. 0 uses the initial
        equilibrium as is. The default of 1 brings the equilibrium in line with
        the TORAX pressure profile; further iterations rarely improve the
        consistency below a few per cent, since re-initialising TORAX rebuilds
        its poloidal flux from the geometry's current profile (unlike the time
        steps, which evolve TORAX's own flux), so any remaining mismatch is
        removed by the iteration of the first coupling interval instead.
    Ip_from_parameters : bool
        If True (default) the plasma current of the initial condition is taken
        from `torax_config.profile_conditions.Ip`; otherwise from the initial
        FreeGSNKE equilibrium. During the time evolution the plasma current is
        evolved by TORAX and passed to FreeGSNKE.
    store_equilibria : bool
        If True, a copy of the FreeGSNKE equilibrium object is stored at each
        coupling time in the result.
    verbose : bool
        Print progress information.

    Returns
    -------
    LooseCouplingResult
        The coupled simulation results.
    """
    _require_torax()
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1].")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")
    if initial_iterations < 0:
        raise ValueError("initial_iterations must be non-negative.")

    numerics = torax_config.numerics
    t_initial = float(numerics.t_initial)
    t_final = float(numerics.t_final)
    calcphibdot = torax_config.geometry.calcphibdot

    step_fn = torax_experimental.make_step_fn(torax_config)

    def geometry_from(ids):
        return _check_finite_geometry(
            torax_geometry_from_ids(
                ids, torax_config, Ip_from_parameters=Ip_from_parameters
            )
        )

    def torax_ids(state, post_processed):
        return torax_experimental.torax_state_to_imas_equilibrium(state, post_processed)

    def geometry_provider_for(times_and_geometries):
        return torax_experimental.geometry.StandardGeometryProvider.create_provider(
            times_and_geometries, calcphibdot=calcphibdot
        )

    def initial_state_for(geometry):
        # The same (interpolating) provider type is used for the initial state
        # and for the time steps so that the geometry held in the TORAX state
        # has identical array types throughout (required by the jitted loops).
        return torax_experimental.get_initial_state_and_post_processed_outputs(
            step_fn,
            geometry_overrides=geometry_provider_for(
                {t_initial: geometry, t_initial + coupling_dt: geometry}
            ),
        )

    def log(message):
        if verbose:
            print(message, flush=True)

    # ------------------------------------------------------------------ #
    # Initial condition: make the TORAX initial state consistent with the
    # FreeGSNKE equilibrium obtained from TORAX's own initial p' and FF'.
    # ------------------------------------------------------------------ #
    wall_start = _time.time()
    eq_ids = equilibrium_solver.initial_equilibrium_ids(t_initial)
    geo = geometry_from(eq_ids)
    state, post_processed = initial_state_for(geo)
    guess_ids = torax_ids(state, post_processed)
    initial_residuals = []
    converged_initial = initial_iterations == 0
    for iteration in range(initial_iterations):
        eq_ids = equilibrium_solver.solve(t_initial, guess_ids)
        geo = geometry_from(eq_ids)
        state, post_processed = initial_state_for(geo)
        sim_error = step_fn.check_for_errors(state, post_processed)
        if sim_error != torax.SimError.NO_ERROR:
            raise RuntimeError(
                f"TORAX reported {sim_error.name} for the initial state built "
                "from the FreeGSNKE equilibrium."
            )
        new_ids = torax_ids(state, post_processed)
        residual = profile_residual(new_ids, guess_ids)
        initial_residuals.append(residual)
        log(
            f"Loose coupling: initial condition, iteration {iteration + 1}, "
            f"residual {residual:.3e}"
        )
        if residual < tolerance:
            guess_ids = new_ids
            converged_initial = True
            break
        guess_ids = relax_profiles(new_ids, guess_ids, relaxation)

    state_history = [state]
    post_processed_history = [post_processed]
    equilibrium_ids_history = [eq_ids]
    torax_ids_history = [guess_ids]
    times = [t_initial]
    iterations = [len(initial_residuals)]
    residuals = [initial_residuals]
    converged = [converged_initial]
    equilibria = [_copy_equilibrium(equilibrium_solver)] if store_equilibria else []
    sim_error = torax.SimError.NO_ERROR

    # ------------------------------------------------------------------ #
    # Time loop over coupling intervals.
    # ------------------------------------------------------------------ #
    t = t_initial
    while t < t_final - _MIN_RELATIVE_DT * coupling_dt:
        dt = min(coupling_dt, t_final - t)
        t_next = t + dt
        step_start = _time.time()

        # First iteration uses the profiles at the start of the interval.
        guess_ids = torax_ids_history[-1]
        step_residuals = []
        step_converged = False
        for iteration in range(max_iterations):
            eq_ids = equilibrium_solver.solve(t_next, guess_ids)
            geo_next = geometry_from(eq_ids)
            new_state, new_post_processed = step_fn.jitted_fixed_time_step(
                jnp.asarray(dt),
                state,
                post_processed,
                geo_overrides=geometry_provider_for({t: geo, t_next: geo_next}),
            )
            sim_error = step_fn.check_for_errors(new_state, new_post_processed)
            if sim_error != torax.SimError.NO_ERROR:
                break
            new_ids = torax_ids(new_state, new_post_processed)
            residual = profile_residual(new_ids, guess_ids)
            step_residuals.append(residual)
            if residual < tolerance:
                step_converged = True
                break
            guess_ids = relax_profiles(new_ids, guess_ids, relaxation)

        if sim_error != torax.SimError.NO_ERROR:
            log(
                f"Loose coupling: TORAX reported {sim_error.name} at t = {t_next:.5f} s."
            )
            sim_error.log_error()
            break

        state, post_processed, geo = new_state, new_post_processed, geo_next
        t = float(state.t)
        state_history.append(state)
        post_processed_history.append(post_processed)
        equilibrium_ids_history.append(eq_ids)
        torax_ids_history.append(new_ids)
        times.append(t)
        iterations.append(len(step_residuals))
        residuals.append(step_residuals)
        converged.append(step_converged)
        if store_equilibria:
            equilibria.append(_copy_equilibrium(equilibrium_solver))
        log(
            f"Loose coupling: t = {t:.5f} s, {len(step_residuals)} iteration(s), "
            "residuals "
            + ", ".join(f"{r:.2e}" for r in step_residuals)
            + ("" if step_converged else " (not converged)")
            + f", {_time.time() - step_start:.1f} s wall time"
        )

    log(
        f"Loose coupling: finished {len(times) - 1} coupling step(s) in "
        f"{_time.time() - wall_start:.1f} s wall time."
    )

    torax_history = torax.StateHistory(
        state_history=state_history,
        post_processed_outputs_history=tuple(post_processed_history),
        sim_error=sim_error,
        torax_config=torax_config,
    )
    return LooseCouplingResult(
        torax_output=torax_history.simulation_output_to_xr(),
        torax_history=torax_history,
        equilibrium_ids=equilibrium_ids_history,
        torax_equilibrium_ids=torax_ids_history,
        times=np.asarray(times),
        iterations=np.asarray(iterations),
        residuals=residuals,
        converged=np.asarray(converged),
        sim_error=sim_error,
        equilibria=equilibria,
    )
