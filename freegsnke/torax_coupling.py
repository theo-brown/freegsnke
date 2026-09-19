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

import copy
import dataclasses
import time as _time

import numpy as np
from freegs4e.gradshafranov import mu0

from . import GSstaticsolver, imas_read_write
from .jtor_update import GeneralPprimeFFprime
from .metal_evolution import MetalCurrentsEvolution

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
# width (in normalised flux) of the layer over which the p' and FF' received
# from TORAX are brought to zero at the separatrix (see
# `imas_read_write.read_profiles_from_equilibrium_ids`)
DEFAULT_EDGE_TAPER_WIDTH = 0.02


def _require_torax():
    """Raises an informative error if TORAX is not installed."""
    if torax is None:
        raise ImportError(
            "The FreeGSNKE-TORAX coupling requires the `torax` package "
            "(https://github.com/google-deepmind/torax) to be installed."
        ) from _TORAX_IMPORT_ERROR


def _make_torax_substepper(step_fn, verbose_log):
    """
    Returns a function advancing a TORAX state by an interval `dt` with TORAX's
    own (fixed or adaptive) time steps, each capped at the end of the interval.

    Unlike `step_fn.jitted_fixed_time_step`, the sub-stepping loop runs in
    Python so that every TORAX time step is recorded: the function returns the
    lists of states and post-processed outputs after each sub-step (excluding
    the input state) and the TORAX error status. Each individual step is
    jitted (the step function is a JAX pytree, so it can be passed as an
    argument); the compilation happens once per array-shape signature.
    """
    import jax

    @jax.jit
    def single_step(step_fn, state, post_processed, max_dt, geo_provider):
        return step_fn(state, post_processed, max_dt=max_dt, geo_overrides=geo_provider)

    adaptive = bool(step_fn.runtime_params_provider.numerics.adaptive_dt)

    def substep(state, post_processed, dt, geo_provider):
        """
        Advances by `dt`. Returns the lists of states and post-processed
        outputs after each TORAX step and the TORAX error status.
        """
        states, post_processed_outputs = [], []
        remaining = float(dt)
        sim_error = torax.SimError.NO_ERROR
        while remaining > _MIN_RELATIVE_DT * dt:
            state, post_processed = single_step(
                step_fn, state, post_processed, jnp.asarray(remaining), geo_provider
            )
            sim_error = step_fn.check_for_errors(state, post_processed)
            states.append(state)
            post_processed_outputs.append(post_processed)
            if sim_error != torax.SimError.NO_ERROR:
                break
            if adaptive and int(state.solver_numeric_outputs.solver_error_state) == 1:
                # the adaptive stepper reached min_dt without converging
                sim_error = torax.SimError.NAN_DETECTED
                verbose_log(
                    "Loose coupling: TORAX adaptive time stepping reached min_dt "
                    f"without converging at t = {float(state.t):.5f} s."
                )
                break
            remaining -= float(state.dt)
        return states, post_processed_outputs, sim_error

    return substep


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


def boundary_targets(eq, reference_xpoint=None):
    """
    Default shape targets for `LinearShapeController`: the inboard and
    outboard midplane radii of the plasma boundary (at the height of the
    magnetic axis), the height of the magnetic axis and the (R, Z) position
    of an X-point: the one closest to `reference_xpoint` if given, otherwise
    the one closest in flux to the plasma boundary (the active X-point of a
    diverted plasma; for a limited plasma with a nearby field null, the null
    that would become the active X-point). Tracking a reference position
    keeps the target well defined for double-null plasmas, where the X-point
    closest in flux can switch between the two nulls.

    Returns
    -------
    np.array
        [R_in, R_out, Z_axis, R_x, Z_x] (the last two are NaN without any
        X-point in the domain).
    """
    from freegs4e.critical import find_critical

    R_axis, Z_axis = eq.magneticAxis()[0:2]
    R_in, R_out = eq.innerOuterSeparatrix(Z=Z_axis)
    targets = [R_in, R_out, Z_axis, np.nan, np.nan]
    opoints, xpoints = find_critical(eq.R, eq.Z, eq.psi())
    if len(xpoints) > 0:
        if reference_xpoint is None:
            xpoint = min(list(xpoints), key=lambda x: abs(x[2] - eq.psi_bndry))
        else:
            xpoint = min(
                list(xpoints),
                key=lambda x: np.hypot(
                    x[0] - reference_xpoint[0], x[1] - reference_xpoint[1]
                ),
            )
        targets[3:] = [xpoint[0], xpoint[1]]
    return np.asarray(targets, dtype=float)


class LinearShapeController:
    """
    Ideal linear shape controller for the loose coupling.

    Holds a set of shape targets T(eq) (by default `boundary_targets`: the
    midplane boundary radii, the axis height and the X-point position) at
    their reference values by adjusting the currents of a set of active coils.
    The response matrix S = dT/dI is obtained by finite differences (one static
    forward solve per coil) and reused for a number of coupling intervals; at
    each equilibrium solve, Gauss-Newton iterations dI = S^+ (T_ref - T(eq))
    (least-squares, minimum-norm current changes) followed by a forward solve
    are applied until the targets are met to `tolerance`.

    This mimics a perfect shape/position control system on the transport
    timescale (the same role the inverse solve plays when building an
    equilibrium), at a small cost: a few forward solves per equilibrium.
    """

    def __init__(
        self,
        coils,
        target_calculator=None,
        target_values=None,
        tolerance=2e-3,
        max_iterations=4,
        relative_step=0.005,
        min_step=1e3,
        relinearise_every=1,
    ):
        """
        Parameters
        ----------
        coils : list of str
            Labels of the active coils used for control.
        target_calculator : callable, optional
            `target_calculator(eq) -> np.array` of shape targets. By default
            `boundary_targets`, tracking the X-point closest to its position
            at the first linearisation.
        target_values : np.array, optional
            Reference target values; by default those of the equilibrium at the
            first `linearise` call.
        tolerance : float
            Largest acceptable absolute target error (all targets are lengths,
            in m).
        max_iterations : int
            Maximum Gauss-Newton iterations per equilibrium solve.
        relative_step : float
            Relative coil current perturbation used for the finite-difference
            response matrix.
        min_step : float
            Smallest absolute current perturbation [A].
        relinearise_every : int
            The response matrix is rebuilt every this many committed coupling
            intervals (0 keeps the first one throughout).
        """
        self.coils = list(coils)
        self._reference_xpoint = None
        self.target_calculator = (
            self._default_targets if target_calculator is None else target_calculator
        )
        self.target_values = (
            None if target_values is None else np.asarray(target_values, float)
        )
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.relative_step = relative_step
        self.min_step = min_step
        self.relinearise_every = relinearise_every
        self.response_matrix = None
        self.intervals_since_linearisation = 0
        self.history = []

    def _default_targets(self, eq):
        targets = boundary_targets(eq, reference_xpoint=self._reference_xpoint)
        if self._reference_xpoint is None and np.all(np.isfinite(targets[3:])):
            self._reference_xpoint = tuple(targets[3:])
        return targets

    def _currents(self, eq):
        currents = eq.tokamak.getCurrents()
        return np.array([currents[label] for label in self.coils], float)

    def _set_currents(self, eq, values):
        for label, value in zip(self.coils, values):
            eq.tokamak.set_coil_current(coil_label=label, current_value=float(value))

    def linearise(self, eq, profiles, solver, target_relative_tolerance):
        """Builds the response matrix dT/dI by finite differences about `eq`."""
        currents_0 = self._currents(eq)
        psi_0 = eq.plasma_psi.copy()
        targets_0 = self.target_calculator(eq)
        if self.target_values is None:
            self.target_values = targets_0.copy()
        matrix = np.zeros((len(targets_0), len(self.coils)))
        for j in range(len(self.coils)):
            step = max(self.relative_step * abs(currents_0[j]), self.min_step)
            currents = currents_0.copy()
            currents[j] += step
            self._set_currents(eq, currents)
            eq.plasma_psi = psi_0.copy()
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=target_relative_tolerance,
                verbose=False,
            )
            matrix[:, j] = (self.target_calculator(eq) - targets_0) / step
        self._set_currents(eq, currents_0)
        eq.plasma_psi = psi_0
        solver.solve(
            eq=eq,
            profiles=profiles,
            constrain=None,
            target_relative_tolerance=target_relative_tolerance,
            verbose=False,
        )
        self.response_matrix = matrix
        self.intervals_since_linearisation = 0

    def control(self, eq, profiles, solver, target_relative_tolerance):
        """
        Adjusts the coil currents of the (solved) equilibrium `eq` so that
        the targets meet their reference values. Returns the final target
        error (max abs).
        """
        if self.response_matrix is None:
            self.linearise(eq, profiles, solver, target_relative_tolerance)
        error = np.nan
        for _ in range(self.max_iterations):
            targets = self.target_calculator(eq)
            if len(targets) != len(self.target_values):
                raise RuntimeError(
                    "The number of shape targets changed (e.g. the plasma went "
                    "from diverted to limited); cannot control the shape."
                )
            mismatch = self.target_values - targets
            if not np.all(np.isfinite(mismatch)):
                raise RuntimeError("A shape target is undefined (e.g. no X-point).")
            error = float(np.max(np.abs(mismatch)))
            if error < self.tolerance:
                break
            dI = np.linalg.lstsq(self.response_matrix, mismatch, rcond=None)[0]
            self._set_currents(eq, self._currents(eq) + dI)
            solver.solve(
                eq=eq,
                profiles=profiles,
                constrain=None,
                target_relative_tolerance=target_relative_tolerance,
                verbose=False,
            )
        else:
            targets = self.target_calculator(eq)
            error = float(np.max(np.abs(self.target_values - targets)))
        self.history.append(error)
        return error

    def committed(self, eq, profiles, solver, target_relative_tolerance):
        """Called once per accepted coupling interval (re-linearisation)."""
        self.intervals_since_linearisation += 1
        if (
            self.relinearise_every
            and self.intervals_since_linearisation >= self.relinearise_every
        ):
            self.linearise(eq, profiles, solver, target_relative_tolerance)


class StaticEquilibriumSolver:
    """
    Free-boundary equilibrium provider for the loose coupling, based on
    FreeGSNKE static forward Grad-Shafranov solves.

    At each request the p' and FF' profiles are read from the `equilibrium` IDS
    supplied by TORAX into a `GeneralPprimeFFprime` profile object, the active
    coil currents are (optionally) updated to their values at the requested
    time, the static forward problem is solved (warm-started from the previous
    solution held in `eq`) and the result is written to a new `equilibrium` IDS.

    With a `shape_controller` given, the coil currents are adjusted at each
    solve so that the plasma shape keeps its targets (an ideal shape
    controller, see `LinearShapeController`); otherwise they are prescribed.

    Any object with the same `solve(time, equilibrium_ids)` and
    `initial_equilibrium_ids(time)` interface can be used in place of this
    class with `run_loose_coupling`, e.g. to include a different controller. An optional `commit(time,
    equilibrium_ids)` method is called by `run_loose_coupling` once a coupling
    interval has been accepted (see `EvolutiveEquilibriumSolver`).
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
        edge_taper_width=DEFAULT_EDGE_TAPER_WIDTH,
        shape_controller=None,
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
        edge_taper_width : float
            Width in normalised flux over which the received p' and FF' are
            brought to zero at the separatrix (0 disables it). See
            `imas_read_write.read_profiles_from_equilibrium_ids`.
        shape_controller : LinearShapeController, optional
            If given, the currents of its control coils are adjusted at each
            solve so that the plasma shape targets (by default the midplane
            boundary radii, axis height and X-point position) keep their
            initial values while the profiles evolve, i.e. an ideal shape
            controller. Without it the coil currents are prescribed (see
            `coil_currents`) and the plasma shape and position are free to
            change with the profiles, which over a long evolution with large
            profile changes can move the plasma against the limiter. Cannot be
            combined with `coil_currents`.
        """
        if shape_controller is not None and coil_currents is not None:
            raise ValueError(
                "shape_controller and coil_currents (prescribed currents) "
                "cannot both be given."
            )
        self.eq = eq
        self.profiles = profiles
        self.coil_currents = coil_currents
        self.shape_controller = shape_controller
        self.solver = solver if solver is not None else GSstaticsolver.NKGSsolver(eq)
        self.psi_n = default_psi_n_grid() if psi_n is None else np.asarray(psi_n)
        self.fvac = profiles.fvac() if fvac is None else fvac
        self.target_relative_tolerance = target_relative_tolerance
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.edge_taper_width = edge_taper_width
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
                self.profiles,
                equilibrium_ids,
                edge_taper_width=self.edge_taper_width,
            )
        else:
            self.profiles = imas_read_write.profiles_from_equilibrium_ids(
                self.eq,
                equilibrium_ids,
                fvac=self.fvac,
                Raxis=self.Raxis,
                Ip_logic=self.Ip_logic,
                interpolator=self.interpolator,
                edge_taper_width=self.edge_taper_width,
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
        if self.shape_controller is not None:
            self.shape_controller.control(
                self.eq, self.profiles, self.solver, self.target_relative_tolerance
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

    def commit(self, time, equilibrium_ids):
        """Accepts the last solve (re-linearises the shape controller)."""
        if self.shape_controller is not None:
            self.shape_controller.committed(
                self.eq, self.profiles, self.solver, self.target_relative_tolerance
            )


def _interpolated_profile_update(
    ids_start, ids_end, t_start, t_end, edge_taper_width=0.0
):
    """
    Returns `update_profiles(profiles, time)` setting the plasma current and
    the p', FF' profiles of a `GeneralPprimeFFprime` object by linear
    interpolation in time between two equilibrium IDSs (each read in
    FreeGSNKE's conventions, with the edge taper applied). The profiles are
    interpolated on the normalised flux grid of `ids_end`.
    """
    end = imas_read_write.read_profiles_from_equilibrium_ids(
        ids_end, edge_taper_width=edge_taper_width
    )
    if ids_start is None or t_end <= t_start:
        start = end
    else:
        start = imas_read_write.read_profiles_from_equilibrium_ids(
            ids_start, edge_taper_width=edge_taper_width
        )
    psi_n = end["psi_n"]
    pprime_start = np.interp(psi_n, start["psi_n"], start["pprime"])
    ffprime_start = np.interp(psi_n, start["psi_n"], start["ffprime"])

    def update_profiles(profiles, time):
        if t_end <= t_start:
            weight = 1.0
        else:
            weight = float(np.clip((time - t_start) / (t_end - t_start), 0.0, 1.0))
        profiles.psi_n = psi_n
        profiles.pprime_data = (1 - weight) * pprime_start + weight * end["pprime"]
        profiles.ffprime_data = (1 - weight) * ffprime_start + weight * end["ffprime"]
        profiles.p_data = None
        profiles.f_data = None
        profiles.Ip = (1 - weight) * start["Ip"] + weight * end["Ip"]
        profiles.initialize_profile()

    return update_profiles


class EvolutiveEquilibriumSolver:
    """
    Free-boundary equilibrium provider for the loose coupling in which the coil
    and passive-structure currents are evolved on the vessel timescale.

    Over each coupling interval [t, t + dt] the metal circuit equations are
    integrated with `metal_evolution.MetalCurrentsEvolution` in sub-steps of
    `vessel_timestep`, driven by the applied coil voltages, while the plasma
    current and the p', FF' profiles are prescribed by TORAX (interpolated
    linearly in time between the IDS at t and the IDS at t + dt). At each
    sub-step the equilibrium is the static free-boundary solution for the
    instantaneous metal currents and profiles, so the vertical displacement and
    eddy-current dynamics are resolved while the current diffusion remains
    with TORAX.

    Since `run_loose_coupling` iterates each coupling interval, the evolution
    always restarts from the last committed state; `commit` is called by the
    loop once the interval has converged.
    """

    def __init__(
        self,
        eq,
        profiles,
        active_voltages=None,
        solver=None,
        vessel_timestep=5e-4,
        max_mode_frequency=None,
        fixed_n_passive_modes=None,
        psi_n=None,
        fvac=None,
        Raxis=1.0,
        Ip_logic=True,
        interpolator="univariate_spline",
        edge_taper_width=DEFAULT_EDGE_TAPER_WIDTH,
        verbose=False,
        **evolution_kwargs,
    ):
        """
        Parameters
        ----------
        eq : freegsnke.equilibrium_update.Equilibrium
            Solved initial equilibrium, with the initial currents set on its
            tokamak (active coils and passive structures).
        profiles : freegsnke.jtor_update profile object
            Profile object used to solve `eq`. A `GeneralPprimeFFprime` object
            with the same `fvac` is used for the coupled evolution.
        active_voltages : np.ndarray or callable, optional
            Voltages applied to the active coils [V] (order of
            `eq.tokamak.coils_list`): a constant vector or
            `active_voltages(time, evolution)` (e.g. a
            `metal_evolution.VerticalPositionController`). Defaults to the
            steady-state voltages of the initial currents.
        solver : freegsnke.GSstaticsolver.NKGSsolver, optional
            Static solver; created if not provided.
        vessel_timestep : float
            Sub-step [s] of the metal current evolution.
        max_mode_frequency : float, optional
            Cut-off rate (1/s) of the retained passive-structure modes; see
            `MetalCurrentsEvolution`.
        fixed_n_passive_modes : int, optional
            Alternatively, number of slowest passive modes to retain.
        psi_n : np.array, optional
            Normalised flux grid used when writing the equilibrium IDS.
        fvac : float, optional
            Vacuum field function R*Btor [T m]; defaults to `profiles.fvac()`.
        Raxis, Ip_logic, interpolator
            Passed to `GeneralPprimeFFprime`.
        edge_taper_width : float
            Width in normalised flux over which the received p' and FF' are
            brought to zero at the separatrix (0 disables it). See
            `imas_read_write.read_profiles_from_equilibrium_ids`.
        verbose : bool
            Print information on each vessel sub-step.
        **evolution_kwargs
            Further keyword arguments for `MetalCurrentsEvolution` (tolerances,
            relaxation, custom resistances/inductances).
        """
        self.eq = eq
        self.psi_n = default_psi_n_grid() if psi_n is None else np.asarray(psi_n)
        self.fvac = profiles.fvac() if fvac is None else fvac
        self.edge_taper_width = edge_taper_width
        self.initial_profiles = profiles
        if isinstance(profiles, GeneralPprimeFFprime):
            self.profiles = profiles
        else:
            # tabulate the initial profiles so that they can be updated from IDSs
            initial_ids = imas_read_write.write_equilibrium_to_ids(
                eq, profiles, psi_n=self.psi_n
            )
            self.profiles = imas_read_write.profiles_from_equilibrium_ids(
                eq,
                initial_ids,
                fvac=self.fvac,
                Raxis=Raxis,
                Ip_logic=Ip_logic,
                interpolator=interpolator,
            )
            # evaluate the tabulated profiles on the equilibrium (sets jtor)
            self.profiles.Jtor(eq.R, eq.Z, eq.psi(), eq.psi_bndry)
        self.evolution = MetalCurrentsEvolution(
            eq,
            self.profiles,
            solver=solver,
            vessel_timestep=vessel_timestep,
            max_mode_frequency=max_mode_frequency,
            fixed_n_passive_modes=fixed_n_passive_modes,
            verbose=verbose,
            **evolution_kwargs,
        )
        self.active_voltages = (
            self.evolution.steady_state_voltages()
            if active_voltages is None
            else active_voltages
        )
        self.verbose = verbose
        self.committed_state = self.evolution.snapshot()
        self.committed_ids = None
        self.n_solves = 0
        self.substep_history = []

    def initial_equilibrium_ids(self, time=0.0):
        """Writes the initial equilibrium to an IDS and labels the state with `time`."""
        self.evolution.set_time(time)
        self.committed_state = self.evolution.snapshot()
        return imas_read_write.write_equilibrium_to_ids(
            self.eq, self.profiles, psi_n=self.psi_n, time=time
        )

    def solve(self, time, equilibrium_ids):
        """
        Evolves the metal currents from the last committed time to `time`,
        with the plasma profiles interpolated between the last committed TORAX
        IDS and `equilibrium_ids`, and returns the equilibrium IDS at `time`.
        If `time` equals the committed time the equilibrium is re-solved at
        fixed currents for the profiles in `equilibrium_ids`.
        """
        self.evolution.restore(self.committed_state)
        t_start = self.committed_state.time
        update_profiles = _interpolated_profile_update(
            self.committed_ids,
            equilibrium_ids,
            t_start,
            time,
            edge_taper_width=self.edge_taper_width,
        )
        if time - t_start < _MIN_RELATIVE_DT * self.evolution.vessel_timestep:
            self.evolution.resolve_static(update_profiles=update_profiles)
        else:
            self.evolution.advance(
                time, self.active_voltages, update_profiles=update_profiles
            )
        self.n_solves += 1
        return imas_read_write.write_equilibrium_to_ids(
            self.eq, self.profiles, psi_n=self.psi_n, time=time
        )

    def commit(self, time, equilibrium_ids):
        """
        Accepts the last `solve` as the state at `time`: the next interval
        starts from it and interpolates the profiles from `equilibrium_ids`.
        """
        if abs(self.evolution.time - time) > _MIN_RELATIVE_DT * max(
            self.evolution.vessel_timestep, abs(time)
        ):
            raise RuntimeError(
                f"commit at t = {time} does not match the last solve at "
                f"t = {self.evolution.time}."
            )
        n_previous = len(self.substep_history)
        self.substep_history = [
            h for h in self.substep_history if h["time"] < self.committed_state.time
        ] + [
            h for h in self.evolution.history if h["time"] >= self.committed_state.time
        ]
        del n_previous
        self.committed_state = self.evolution.snapshot()
        self.committed_ids = equilibrium_ids


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


def _midplane_radii(equilibrium_ids, psi_n):
    """
    Inboard and outboard midplane radii of the flux surfaces of the first time
    slice of an IDS (`r_inboard`, `r_outboard`). If they are missing or
    invalid, the reference major radius R0 is returned for both.
    """
    profiles_1d = equilibrium_ids.time_slice[0].profiles_1d
    R0 = float(equilibrium_ids.vacuum_toroidal_field.r0)
    r_in = np.asarray(profiles_1d.r_inboard, dtype=float)
    r_out = np.asarray(profiles_1d.r_outboard, dtype=float)
    valid = (
        len(r_in) == len(psi_n)
        and len(r_out) == len(psi_n)
        and np.all(r_in > 0)
        and np.all(r_out > 0)
    )
    if not valid:
        r_in = r_out = np.full_like(psi_n, R0)
    return r_in, r_out


def _toroidal_current_density(pprime, ffprime, radii):
    """Jtor = R p' + FF' / (mu0 R) stacked for each array of radii."""
    return np.concatenate([R * pprime + ffprime / (mu0 * R) for R in radii])


def profile_residual(equilibrium_ids, previous_equilibrium_ids):
    """
    Relative change of the exchanged p' and FF' profiles between two
    `equilibrium` IDSs, used as the loose-coupling convergence measure.

    The two profiles enter the Grad-Shafranov equation only through the
    toroidal current density Jtor = R p' + FF' / (mu0 R), so they are compared
    through Jtor evaluated at the inboard and outboard midplane radii of each
    flux surface (`r_inboard`, `r_outboard` of the newest IDS):

        residual = ||Jtor_new - Jtor_old|| / ||Jtor_new||,

    with the old profiles interpolated onto the new normalised-flux grid and
    the norm taken over both radii and all flux surfaces. Comparing Jtor
    rather than p' and FF' separately avoids two spurious contributions: a
    large relative change of a profile that is small (typically FF' for a
    plasma close to the force-free/diamagnetic balance), and the exchange of
    current between the p' and FF' terms on the innermost flux surfaces, where
    Jtor is well defined but its split into the two terms is not (TORAX
    computes the on-axis p' from a second difference of the poloidal flux and
    sets FF' to conserve <Jtor/R>, so the split there responds strongly to
    small changes of the equilibrium while Jtor does not). If the IDS carries
    no midplane radii, both are replaced by the reference major radius R0.

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
    psi_n, pprime, ffprime, _ = _exchanged_profiles(equilibrium_ids)
    psi_n_old, pprime_old, ffprime_old, _ = _exchanged_profiles(
        previous_equilibrium_ids
    )
    radii = _midplane_radii(equilibrium_ids, psi_n)
    jtor = _toroidal_current_density(pprime, ffprime, radii)
    jtor_old = _toroidal_current_density(
        np.interp(psi_n, psi_n_old, pprime_old),
        np.interp(psi_n, psi_n_old, ffprime_old),
        radii,
    )
    norm = np.linalg.norm(jtor)
    return float(np.linalg.norm(jtor - jtor_old) / max(norm, _RESIDUAL_NORM_FLOOR))


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


class AndersonAccelerator:
    """
    Anderson acceleration of the loose-coupling fixed-point iteration on the
    exchanged p' and FF' profiles.

    The iteration maps the profiles handed to the equilibrium solver, x, to the
    profiles TORAX returns after the coupling interval, G(x). Plain relaxation
    updates x <- x + relaxation * (G(x) - x), which converges linearly with a
    rate set by the (case dependent) slope of G. Anderson mixing uses the last
    `memory` differences of x and of the residual f = G(x) - x to extrapolate
    towards the fixed point (a multi-secant quasi-Newton update),

        x_next = x + beta f - (dX + beta dF) gamma,   gamma = argmin ||f - dF gamma||,

    with beta the relaxation factor and dX, dF the matrices of successive
    differences. With `memory` = 0 the update reduces to plain relaxation. The
    mixed vector holds the two profiles on a common current-density scale
    (a = R0 p', b = FF' / (mu0 R0)) on the normalised-flux grid of the newest
    IDS, and a fresh history is started for each coupling interval with
    `reset`.
    """

    def __init__(self, memory=4, relaxation=0.5, rcond=1e-8):
        """
        Parameters
        ----------
        memory : int
            Number of previous iterates kept (0 gives plain relaxation).
        relaxation : float
            Damping (mixing) factor beta in (0, 1].
        rcond : float
            Cut-off ratio for the singular values in the least-squares
            problem for the mixing coefficients.
        """
        if memory < 0:
            raise ValueError("memory must be non-negative.")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1].")
        self.memory = memory
        self.relaxation = relaxation
        self.rcond = rcond
        self.reset()

    def reset(self):
        """Forgets the iteration history (call at the start of an interval)."""
        self.psi_n = None
        self.R0 = None
        self.x_history = []
        self.f_history = []

    def _vector(self, equilibrium_ids, psi_n):
        """The (a, b) vector of an IDS interpolated onto the grid `psi_n`."""
        psi_n_ids, pprime, ffprime, _ = _exchanged_profiles(equilibrium_ids)
        return np.concatenate(
            [
                self.R0 * np.interp(psi_n, psi_n_ids, pprime),
                np.interp(psi_n, psi_n_ids, ffprime) / (mu0 * self.R0),
            ]
        )

    def _regrid(self, vector, psi_n_new):
        """Re-interpolates a stored vector from the current grid onto a new one."""
        n = len(self.psi_n)
        return np.concatenate(
            [
                np.interp(psi_n_new, self.psi_n, vector[:n]),
                np.interp(psi_n_new, self.psi_n, vector[n:]),
            ]
        )

    def update(self, equilibrium_ids, previous_equilibrium_ids):
        """
        Computes the next profiles to hand to the equilibrium solver.

        The vectors are held on the normalised-flux grid of the newest IDS
        (the history is re-interpolated whenever the grid changes, which it
        does slightly from one TORAX output to the next). Since successive
        grids converge together with the iteration, the fixed point is free of
        interpolation error, consistently with `profile_residual`: with a
        fixed reference grid instead, the interpolation of TORAX's grid-scale
        edge structure back and forth leaves a residual floor.

        Parameters
        ----------
        equilibrium_ids : imas.ids_toplevel.IDSToplevel
            Newest IDS from TORAX, G(x) (not modified).
        previous_equilibrium_ids : imas.ids_toplevel.IDSToplevel
            IDS passed to the equilibrium solver in this iteration, x.

        Returns
        -------
        imas.ids_toplevel.IDSToplevel
            A copy of `equilibrium_ids` with the mixed p' and FF' profiles.
        """
        if self.memory == 0 and self.relaxation == 1.0:
            return equilibrium_ids
        psi_n_new, _, _, R0 = _exchanged_profiles(equilibrium_ids)
        if self.psi_n is None:
            self.R0 = R0
        elif len(psi_n_new) != len(self.psi_n) or np.any(psi_n_new != self.psi_n):
            self.x_history = [self._regrid(v, psi_n_new) for v in self.x_history]
            self.f_history = [self._regrid(v, psi_n_new) for v in self.f_history]
        self.psi_n = psi_n_new.copy()

        x = self._vector(previous_equilibrium_ids, self.psi_n)
        f = self._vector(equilibrium_ids, self.psi_n) - x
        beta = self.relaxation
        x_next = x + beta * f
        if self.memory > 0 and self.x_history:
            dX = np.array([x - x_old for x_old in self.x_history]).T
            dF = np.array([f - f_old for f_old in self.f_history]).T
            gamma = np.linalg.lstsq(dF, f, rcond=self.rcond)[0]
            x_next = x_next - (dX + beta * dF) @ gamma
        self.x_history = (self.x_history + [x])[-self.memory :] if self.memory else []
        self.f_history = (self.f_history + [f])[-self.memory :] if self.memory else []

        mixed = copy.deepcopy(equilibrium_ids)
        profiles_1d = mixed.time_slice[0].profiles_1d
        n = len(self.psi_n)
        profiles_1d.dpressure_dpsi = x_next[:n] / self.R0
        profiles_1d.f_df_dpsi = x_next[n:] * mu0 * self.R0
        return mixed


@dataclasses.dataclass
class LooseCouplingResult:
    """
    Output of `run_loose_coupling`.

    Attributes
    ----------
    torax_output : xarray.DataTree
        TORAX simulation output (as returned by `torax.run_simulation`), with
        one entry per TORAX time step. TORAX takes its own (fixed or adaptive)
        time steps within each coupling interval, so this is typically finer
        than the coupling times.
    torax_history : torax.StateHistory
        TORAX state history (states and post-processed outputs at each TORAX
        time step).
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
    torax_substeps : np.array
        Number of TORAX time steps taken within each coupling interval (the
        first entry, for the initial condition, is 0).
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
    torax_substeps: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, int)
    )


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
    anderson_memory=0,
):
    """
    Runs a loosely coupled FreeGSNKE-TORAX simulation.

    See the module docstring for a description of the algorithm. The TORAX
    simulation runs from `torax_config.numerics.t_initial` to
    `torax_config.numerics.t_final` in coupling intervals of `coupling_dt`
    (TORAX takes its own fixed or adaptive time steps within each interval, on
    the geometry interpolated linearly in time between the equilibria at the
    interval ends; every TORAX step is kept in the output). The geometry
    section of `torax_config`
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
    anderson_memory : int
        If positive, the profiles handed to the equilibrium solver are updated
        by Anderson acceleration with this memory (see `AndersonAccelerator`,
        `relaxation` then acts as the damping factor) instead of plain
        relaxation. Useful when the plain iteration converges slowly, e.g.
        because the profiles TORAX returns respond strongly (with a positive
        slope) to the equilibrium, as found for ITER-like cases.

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

    advance_torax = _make_torax_substepper(step_fn, log)
    mixer = AndersonAccelerator(memory=anderson_memory, relaxation=relaxation)

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
        guess_ids = mixer.update(new_ids, guess_ids)

    if hasattr(equilibrium_solver, "commit"):
        equilibrium_solver.commit(t_initial, guess_ids)

    state_history = [state]
    post_processed_history = [post_processed]
    equilibrium_ids_history = [eq_ids]
    torax_ids_history = [guess_ids]
    times = [t_initial]
    iterations = [len(initial_residuals)]
    residuals = [initial_residuals]
    converged = [converged_initial]
    substeps = [0]
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
        mixer.reset()
        for iteration in range(max_iterations):
            eq_ids = equilibrium_solver.solve(t_next, guess_ids)
            geo_next = geometry_from(eq_ids)
            # TORAX advances over the interval with its own (fixed or adaptive)
            # time steps, on the geometry interpolated linearly in time between
            # the equilibria at the start and end of the interval.
            new_states, new_post_processed_outputs, sim_error = advance_torax(
                state,
                post_processed,
                dt,
                geometry_provider_for({t: geo, t_next: geo_next}),
            )
            if sim_error != torax.SimError.NO_ERROR:
                break
            new_state = new_states[-1]
            new_post_processed = new_post_processed_outputs[-1]
            new_ids = torax_ids(new_state, new_post_processed)
            residual = profile_residual(new_ids, guess_ids)
            step_residuals.append(residual)
            if residual < tolerance:
                step_converged = True
                break
            guess_ids = mixer.update(new_ids, guess_ids)

        if sim_error != torax.SimError.NO_ERROR:
            log(
                f"Loose coupling: TORAX reported {sim_error.name} at t = {t_next:.5f} s."
            )
            sim_error.log_error()
            break

        state, post_processed, geo = new_state, new_post_processed, geo_next
        t = float(state.t)
        if hasattr(equilibrium_solver, "commit"):
            equilibrium_solver.commit(t, new_ids)
        state_history.extend(new_states)
        post_processed_history.extend(new_post_processed_outputs)
        equilibrium_ids_history.append(eq_ids)
        torax_ids_history.append(new_ids)
        times.append(t)
        iterations.append(len(step_residuals))
        residuals.append(step_residuals)
        converged.append(step_converged)
        substeps.append(len(new_states))
        if store_equilibria:
            equilibria.append(_copy_equilibrium(equilibrium_solver))
        log(
            f"Loose coupling: t = {t:.5f} s, {len(step_residuals)} iteration(s), "
            "residuals "
            + ", ".join(f"{r:.2e}" for r in step_residuals)
            + ("" if step_converged else " (not converged)")
            + f", {len(new_states)} TORAX step(s)"
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
        torax_substeps=np.asarray(substeps),
    )
