"""
Evolution of the coil and passive-structure currents on the vessel timescale
for a plasma whose total current and current-profile shape are prescribed.

This module complements `nonlinear_solve`, which evolves the metal currents
*and* the total plasma current (through a lumped plasma circuit equation) with
a parametric profile family. Here the plasma current and the p'(psi), FF'(psi)
profiles are instead supplied externally, e.g. by a core transport code that
evolves the current diffusion equation itself (see `torax_coupling`), and only
the metal circuit equations are integrated:

    Lambda^-1 dI_d/dt + I_d = P^-1 R^-1 (U - M_ey dI_y/dt),

where I_d are the metal currents in the (truncated) vessel normal-mode basis of
`circuit_eq_metal.metal_currents`, U the applied active coil voltages and I_y
the plasma current distribution on the reduced plasma domain. At each implicit
Euler step the plasma distribution at t + dt is the static free-boundary
Grad-Shafranov solution for the metal currents at t + dt. On timescales
shorter than the vessel L/R times the passive structures respond to a change
of the plasma current distribution with order-one image currents, and the
free-boundary plasma responds to those in turn, so a plain fixed-point
iteration between the circuit stepper and the static solver converges slowly
(or not at all). The implicit step is therefore solved with FreeGSNKE's
Newton-Krylov solver on the (small) vector of metal mode currents, each Krylov
direction costing one static solve.

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

import numpy as np

from . import GSstaticsolver
from .circuit_eq_metal import metal_currents
from .nk_solver_H import nksolver

# smallest advance (as a fraction of the vessel timestep) treated as non-zero
_MIN_RELATIVE_ADVANCE = 1e-8
# floor for the norms used to make the residuals relative
_NORM_FLOOR = 1e-300
# a step is also accepted when the residual is this small relative to the
# metal currents themselves (e.g. a stationary state, where the increment is 0)
_ABSOLUTE_TOLERANCE_FACTOR = 1e-3
# minimum norm of the exploratory Newton-Krylov steps relative to the norm of
# the metal mode currents: keeps the finite-difference directional derivatives
# above the noise floor of the static solves (whose current distribution
# changes discontinuously as grid cells enter or leave the plasma)
_MIN_EXPLORATORY_STEP_FRACTION = 1e-4


@dataclasses.dataclass
class MetalEvolutionState:
    """
    Snapshot of the evolving state (used to restore the start of a coupling
    interval when the interval is iterated).

    Attributes
    ----------
    time : float
        Time [s].
    currents : np.ndarray
        Currents in all metal elements (active coils first, then passives) [A].
    Id : np.ndarray
        Metal currents in the truncated normal-mode basis.
    Iy : np.ndarray
        Plasma current distribution on the reduced plasma domain [A].
    plasma_psi : np.ndarray
        Plasma poloidal flux on the grid [Wb/rad].
    """

    time: float
    currents: np.ndarray
    Id: np.ndarray
    Iy: np.ndarray
    plasma_psi: np.ndarray


class MetalCurrentsEvolution:
    """
    Integrates the coil and passive-structure circuit equations on the vessel
    timescale for a plasma with prescribed total current and profiles.

    The plasma is described by a FreeGSNKE profile object (its `Ip` and profile
    data may be updated between steps through the `update_profiles` callback of
    `step`/`advance`, e.g. to follow a transport code) and the equilibrium is
    obtained from a static free-boundary solve at each step. Vessel modes faster
    than `max_mode_frequency` are removed from the circuit model, as in
    `nonlinear_solve`, so that the retained dynamics are those on timescales
    longer than the timestep. Each implicit step is solved with a Newton-Krylov
    iteration on the metal mode currents (see the module docstring).
    """

    def __init__(
        self,
        eq,
        profiles,
        solver=None,
        vessel_timestep=5e-4,
        max_mode_frequency=None,
        fixed_n_passive_modes=None,
        custom_coil_resist=None,
        custom_self_ind=None,
        target_relative_tolerance=1e-8,
        step_tolerance=1e-2,
        max_step_iterations=10,
        max_n_directions=4,
        target_relative_unexplained_residual=0.2,
        nk_step_size=1.0,
        nk_clip=10.0,
        vertical_control=None,
        position_constraint_scale=1e4,
        solver_kwargs=None,
        verbose=False,
    ):
        """
        Parameters
        ----------
        eq : freegsnke.equilibrium_update.Equilibrium
            Solved equilibrium (with `profiles`) defining the initial state:
            the currents currently set on `eq.tokamak` (active coils and
            passive structures) and the plasma flux.
        profiles : freegsnke.jtor_update profile object
            Profile object used to solve `eq`; `profiles.jtor` must be
            available (i.e. the equilibrium has been solved with it).
        solver : freegsnke.GSstaticsolver.NKGSsolver, optional
            Static solver for `eq`; created if not provided.
        vessel_timestep : float
            Time step [s] of the implicit Euler integration of the circuit
            equations; the vessel dynamics are resolved on this scale.
        max_mode_frequency : float, optional
            Passive-structure normal modes with a characteristic rate (1/s)
            above this value are dropped. Defaults to 1 / (5 * vessel_timestep),
            i.e. modes decaying faster than a few timesteps are not resolved.
        fixed_n_passive_modes : int, optional
            Alternatively, retain exactly this many slowest passive modes.
        custom_coil_resist : np.ndarray, optional
            Resistances of all metal elements [Ohm]; machine values by default.
        custom_self_ind : np.ndarray, optional
            Mutual inductance matrix of all metal elements [H]; machine values
            by default.
        target_relative_tolerance : float
            Relative tolerance of the static Grad-Shafranov solves.
        step_tolerance : float
            Convergence tolerance of the implicit step: the norm of the
            circuit-equation residual relative to the norm of the change of
            the metal mode currents over the step (or, for a stationary state,
            relative to the mode currents themselves, scaled by 1e-3).
        max_step_iterations : int
            Maximum number of Newton-Krylov iterations per step.
        max_n_directions : int
            Maximum number of Krylov directions explored per Newton-Krylov
            iteration (each costs one static solve).
        target_relative_unexplained_residual : float
            Fraction of the residual left unexplained at which the exploration
            of further Krylov directions stops.
        nk_step_size : float
            Size of the exploratory steps in units of the residual norm.
        nk_clip : float
            Maximum coefficient of each explored direction in the combined step.
        vertical_control : tuple (coil_index, z_target), optional
            Ideal vertical position control: the circuit equation of the given
            active coil is replaced by the constraint that the magnetic axis
            sits at `z_target` [m] (a float or a callable of time) at the end
            of each step, i.e. the coil is driven by whatever voltage is needed
            (reported in the history as `implied_voltage`). This represents a
            fast vertical controller without having to tune voltage feedback
            gains, which is delicate since the early-time response of the
            plasma to a coil outside the passive structures can have the
            opposite sign to the static one.
        position_constraint_scale : float
            Scale [A/m] converting the position error into the current units
            of the residual (of the order of the coil's static authority
            dI/dZ).
        solver_kwargs : dict, optional
            Additional keyword arguments for `NKGSsolver.solve`.
        verbose : bool
            Print information on the mode selection and on each step.
        """
        if not hasattr(profiles, "jtor"):
            raise ValueError(
                "`profiles` has no current distribution: solve `eq` with it first."
            )
        self.eq = eq
        self.profiles = profiles
        self.solver = solver if solver is not None else GSstaticsolver.NKGSsolver(eq)
        self.vessel_timestep = float(vessel_timestep)
        if max_mode_frequency is None:
            max_mode_frequency = 1.0 / (5.0 * self.vessel_timestep)
        self.max_mode_frequency = max_mode_frequency
        self.target_relative_tolerance = target_relative_tolerance
        self.step_tolerance = step_tolerance
        self.max_step_iterations = max_step_iterations
        self.max_n_directions = max_n_directions
        self.target_relative_unexplained_residual = target_relative_unexplained_residual
        self.nk_step_size = nk_step_size
        self.nk_clip = nk_clip
        self.vertical_control = vertical_control
        self.position_constraint_scale = position_constraint_scale
        if vertical_control is not None:
            coil_index = int(vertical_control[0])
            if not 0 <= coil_index < eq.tokamak.n_active_coils:
                raise ValueError("vertical_control coil_index must be an active coil.")
        self.solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
        self.verbose = verbose

        self.limiter_handler = eq.limiter_handler
        self.n_coils = eq.tokamak.n_coils
        self.n_active_coils = eq.tokamak.n_active_coils

        # metal circuit equations in the vessel normal-mode basis, coupled to
        # the plasma current distribution on the reduced domain
        self.metal = metal_currents(
            eq=eq,
            flag_vessel_eig=True,
            flag_plasma=True,
            plasma_pts=self.limiter_handler.plasma_pts,
            max_mode_frequency=self.max_mode_frequency,
            max_internal_timestep=self.vessel_timestep,
            full_timestep=self.vessel_timestep,
            coil_resist=custom_coil_resist,
            coil_self_ind=custom_self_ind,
            verbose=verbose,
        )
        # apply the timescale-based mode selection
        self.metal.initialize_for_eig(
            selected_modes_mask=None,
            mode_coupling_masks=None,
            verbose=verbose,
            fixed_n_passive_modes=fixed_n_passive_modes,
        )
        self.n_modes = self.metal.n_independent_vars
        self._full_voltages = np.zeros(self.n_coils)
        self.nk = nksolver(problem_dimension=self.n_modes)
        self._last_evaluation = None
        self._previous_increment = None

        # initial state
        currents = np.asarray(eq.tokamak.getCurrentsVec(), dtype=float).copy()
        self.state = MetalEvolutionState(
            time=0.0,
            currents=currents,
            Id=self.metal.IvesseltoId(currents),
            Iy=self.limiter_handler.Iy_from_jtor(profiles.jtor).copy(),
            plasma_psi=np.copy(eq.plasma_psi),
        )
        self.history = []
        self._record_history(iterations=0, residual=0.0)

    # ------------------------------------------------------------------ #
    # state handling
    # ------------------------------------------------------------------ #
    @property
    def time(self):
        """Current time [s] of the evolving state."""
        return self.state.time

    @property
    def currents(self):
        """Currents in all metal elements at the current time [A]."""
        return self.state.currents

    @property
    def active_currents(self):
        """Active coil currents at the current time [A]."""
        return self.state.currents[: self.n_active_coils]

    def snapshot(self):
        """Returns a copy of the current state."""
        return MetalEvolutionState(
            time=self.state.time,
            currents=np.copy(self.state.currents),
            Id=np.copy(self.state.Id),
            Iy=np.copy(self.state.Iy),
            plasma_psi=np.copy(self.state.plasma_psi),
        )

    def restore(self, state):
        """Restores a state returned by `snapshot` (also on `eq`)."""
        self.state = MetalEvolutionState(
            time=state.time,
            currents=np.copy(state.currents),
            Id=np.copy(state.Id),
            Iy=np.copy(state.Iy),
            plasma_psi=np.copy(state.plasma_psi),
        )
        self.eq.tokamak.set_all_coil_currents(self.state.currents)
        self.eq.plasma_psi = np.copy(self.state.plasma_psi)
        self.history = [h for h in self.history if h["time"] <= state.time]
        self._previous_increment = None

    def set_time(self, time):
        """Sets the time label of the current state (does not evolve it)."""
        self.state.time = float(time)
        if self.history:
            self.history[-1]["time"] = float(time)

    def steady_state_voltages(self, currents=None):
        """
        Active coil voltages that sustain the given (default: current) active
        coil currents against their resistance, U = R I, i.e. the voltages that
        hold the coil currents constant in the absence of plasma or vessel
        changes.
        """
        if currents is None:
            currents = self.state.currents
        return (
            np.asarray(currents)[: self.n_active_coils]
            * self.metal.coil_resist[: self.n_active_coils]
        )

    def _axis_position(self):
        """(R, Z) of the magnetic axis of the last solve of `eq`."""
        opt = getattr(self.eq, "opt", None)
        if opt is not None and len(opt) > 0:
            return float(opt[0][0]), float(opt[0][1])
        return np.nan, np.nan

    def _record_history(self, iterations, residual, implied_voltage=np.nan):
        R_axis, Z_axis = self._axis_position()
        self.history.append(
            dict(
                time=self.state.time,
                currents=np.copy(self.state.currents),
                Ip=float(np.sum(self.state.Iy)),
                R_axis=R_axis,
                Z_axis=Z_axis,
                iterations=iterations,
                residual=residual,
                implied_voltage=implied_voltage,
            )
        )

    # ------------------------------------------------------------------ #
    # solves
    # ------------------------------------------------------------------ #
    def _solve_static(self):
        """Static solve of `eq` for the currents set on the tokamak."""
        self.solver.solve(
            eq=self.eq,
            profiles=self.profiles,
            constrain=None,
            target_relative_tolerance=self.target_relative_tolerance,
            verbose=False,
            **self.solver_kwargs,
        )
        if not np.all(np.isfinite(self.eq.plasma_psi)):
            raise RuntimeError("The static solve produced a non-finite plasma flux.")
        return self.limiter_handler.Iy_from_jtor(self.profiles.jtor)

    def resolve_static(self, update_profiles=None):
        """
        Re-solves the equilibrium at the current time and metal currents, e.g.
        after the plasma profiles have changed, without advancing the circuit
        equations. Useful to make the initial state consistent with externally
        supplied profiles.

        Parameters
        ----------
        update_profiles : callable, optional
            `update_profiles(profiles, time)` is called before the solve.
        """
        if update_profiles is not None:
            update_profiles(self.profiles, self.state.time)
        self.eq.tokamak.set_all_coil_currents(self.state.currents)
        self.state.Iy = self._solve_static()
        self.state.plasma_psi = np.copy(self.eq.plasma_psi)
        self.history.pop()
        self._record_history(iterations=0, residual=0.0)

    def _forcing(self, Iy_dot):
        """Right-hand side of the mode circuit equations for the plasma rate Iy_dot."""
        return self.metal.Pm1 @ (
            self.metal.Rm1 * (self._full_voltages - self.metal.Mey_matrix @ Iy_dot)
        )

    def _z_target(self, time):
        target = self.vertical_control[1]
        return float(target(time)) if callable(target) else float(target)

    def _residual(self, x, Id_0, Iy_0, dt, t_new):
        """
        Residual of the implicit Euler step for a trial vector of metal mode
        currents x at t + dt, in current units: A^-1 (A x - Lambda^-1 Id_0 - dt
        F(x)) with A = Lambda^-1 + dt, which equals x - Phi(x) where Phi is the
        circuit-equation step from Id_0 with the plasma response I_y(x) of the
        static equilibrium at the metal currents x. With vertical control, the
        equation of the controlling coil is replaced by the position
        constraint. Leaves `eq` solved at x.
        """
        currents = self.metal.IdtoIvessel(x)
        self.eq.tokamak.set_all_coil_currents(currents)
        Iy = self._solve_static()
        Iy_dot = (Iy - Iy_0) / dt
        forcing = self._forcing(Iy_dot)
        solver = self.metal.solver
        # equation form of the implicit Euler step (units of current * time)
        equations = solver.Mmatrix @ x + dt * x - solver.Lmatrix @ Id_0 - dt * forcing
        if self.vertical_control is None:
            return_residual = solver.inverse_operator @ equations
        else:
            # Drop the circuit equation of the controlled coil (its voltage is
            # free) before preconditioning with A^-1, then use the position
            # error, in current units, as that coil's residual. Since A is
            # invertible, a zero residual is equivalent to all other circuit
            # equations and the constraint being satisfied.
            coil = int(self.vertical_control[0])
            equations[coil] = 0.0
            return_residual = solver.inverse_operator @ equations
            _, z_axis = self._axis_position()
            return_residual[coil] = self.position_constraint_scale * (
                z_axis - self._z_target(t_new)
            )
        residual = return_residual
        self._last_evaluation = (np.copy(x), np.asarray(currents, dtype=float), Iy)
        return residual

    def _implied_voltage(self, x, Id_0, Iy_1, Iy_0, dt):
        """Voltage of the controlled coil consistent with its circuit equation."""
        coil = int(self.vertical_control[0])
        Id_dot = (x - Id_0) / dt
        Iy_dot = (Iy_1 - Iy_0) / dt
        # row `coil` of Lambda^-1 Id_dot + Id = P^-1 R^-1 (U - Mey Iy_dot); the
        # active-coil block of P is the identity
        lhs = self.metal.Lambdam1[coil] @ Id_dot + x[coil]
        induced = (
            self.metal.Pm1 @ (self.metal.Rm1 * (self.metal.Mey_matrix @ Iy_dot))
        )[coil]
        return float((lhs + induced) / self.metal.Rm1[coil])

    def step(self, dt, active_voltages, update_profiles=None):
        """
        Advances the metal currents (and the equilibrium) by one implicit Euler
        step of length `dt`, solving the coupled circuit/equilibrium step with
        a Newton-Krylov iteration on the metal mode currents.

        Parameters
        ----------
        dt : float
            Time step [s].
        active_voltages : np.ndarray
            Voltages applied to the active coils during the step [V], in the
            order of `eq.tokamak.coils_list` (the entry of a vertically
            controlling coil is ignored).
        update_profiles : callable, optional
            `update_profiles(profiles, time)` is called with the time at the
            end of the step before the plasma is solved, to set the prescribed
            plasma current and profiles at that time.

        Returns
        -------
        dict
            Number of Newton-Krylov iterations, number of static solves, final
            residual and whether the iteration converged.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if dt != self.metal.solver.full_timestep:
            self.metal.reset_timesteps(max_internal_timestep=dt, full_timestep=dt)
        t_new = self.state.time + dt
        if update_profiles is not None:
            update_profiles(self.profiles, t_new)

        self._full_voltages[:] = 0.0
        self._full_voltages[: self.n_active_coils] = np.asarray(active_voltages)

        Id_0 = np.copy(self.state.Id)
        Iy_0 = np.copy(self.state.Iy)
        args = [Id_0, Iy_0, dt, t_new]
        n_solves = 0

        # initial guess: extrapolate the previous increment (smooth evolution),
        # otherwise one fixed-point step from the currents at t
        if self._previous_increment is not None:
            x = Id_0 + self._previous_increment
        else:
            x = Id_0 - self._residual(Id_0, *args)
            n_solves += 1
            if self.vertical_control is not None:
                # the constrained coil keeps its current as initial guess
                coil = int(self.vertical_control[0])
                x[coil] = Id_0[coil]
        converged = False
        residual = np.nan
        for iteration in range(1, self.max_step_iterations + 1):
            R = self._residual(x, *args)
            n_solves += 1
            norm_R = np.linalg.norm(R)
            residual = float(norm_R / max(np.linalg.norm(x - Id_0), _NORM_FLOOR))
            residual_absolute = float(norm_R / max(np.linalg.norm(x), _NORM_FLOOR))
            if (
                residual < self.step_tolerance
                or residual_absolute < _ABSOLUTE_TOLERANCE_FACTOR * self.step_tolerance
            ):
                converged = True
                break
            # exploratory steps of at least a fixed fraction of the currents
            min_step = _MIN_EXPLORATORY_STEP_FRACTION * np.linalg.norm(x)
            step_size = max(self.nk_step_size, min_step / max(norm_R, _NORM_FLOOR))
            self.nk.Arnoldi_iteration(
                x0=np.copy(x),
                dx=-np.copy(R),
                R0=np.copy(R),
                F_function=self._residual,
                args=args,
                step_size=step_size,
                scaling_with_n=0,
                target_relative_unexplained_residual=self.target_relative_unexplained_residual,
                max_n_directions=self.max_n_directions,
                clip=self.nk_clip,
            )
            n_solves += self.nk.n_it + 1
            x = x + self.nk.dx
        if not converged:
            # leave the equilibrium consistent with the final iterate
            self._residual(x, *args)
            n_solves += 1
            if self.verbose:
                print(
                    f"Metal evolution step to t = {t_new:.6f} s did not converge: "
                    f"residual {residual:.2e} after {iteration} iterations."
                )

        x_final, currents_1, Iy_1 = self._last_evaluation
        self._previous_increment = x_final - Id_0
        implied_voltage = np.nan
        if self.vertical_control is not None:
            implied_voltage = self._implied_voltage(x_final, Id_0, Iy_1, Iy_0, dt)
        self.state = MetalEvolutionState(
            time=t_new,
            currents=currents_1,
            Id=np.asarray(x_final, dtype=float),
            Iy=np.asarray(Iy_1, dtype=float),
            plasma_psi=np.copy(self.eq.plasma_psi),
        )
        self._record_history(iteration, residual, implied_voltage)
        if self.verbose:
            print(
                f"Metal evolution: t = {t_new:.6f} s, {iteration} NK iteration(s), "
                f"{n_solves} static solves, residual {residual:.2e}, axis at R = "
                f"{self.history[-1]['R_axis']:.4f} m, Z = {self.history[-1]['Z_axis']:.4f} m, "
                f"Ip = {self.history[-1]['Ip']:.0f} A"
            )
        return dict(
            iterations=iteration,
            n_solves=n_solves,
            residual=residual,
            converged=converged,
        )

    def advance(self, t_end, active_voltages, update_profiles=None):
        """
        Advances from the current time to `t_end` in steps of at most
        `vessel_timestep` (equal steps).

        Parameters
        ----------
        t_end : float
            End time [s].
        active_voltages : np.ndarray or callable
            Either a constant voltage vector or `active_voltages(time, evolution)`
            returning the voltages to apply from the current time (`evolution`
            is this object, so e.g. a controller can use `evolution.eq` and
            `evolution.currents`).
        update_profiles : callable, optional
            See `step`.

        Returns
        -------
        list of dict
            The per-step information returned by `step`.
        """
        duration = float(t_end) - self.state.time
        if duration < _MIN_RELATIVE_ADVANCE * self.vessel_timestep:
            return []
        n_steps = max(1, int(np.ceil(duration / self.vessel_timestep - 1e-9)))
        dt = duration / n_steps
        infos = []
        for _ in range(n_steps):
            if callable(active_voltages):
                voltages = active_voltages(self.state.time, self)
            else:
                voltages = active_voltages
            infos.append(self.step(dt, voltages, update_profiles=update_profiles))
        return infos


class VerticalPositionController:
    """
    Proportional-derivative feedback on the vertical position of the magnetic
    axis through one active coil, on top of a base voltage waveform:

        U_coil = U_base_coil + gain_p * (Z_axis - Z_target)
                             + gain_d * d(Z_axis)/dt.

    The other coils are driven by their base voltages. The signs of the gains
    depend on the coil's field direction and must be chosen for the machine.
    """

    def __init__(
        self, coil_index, gain_p, gain_d=0.0, z_target=0.0, base_voltages=None
    ):
        """
        Parameters
        ----------
        coil_index : int
            Index of the controlling coil among the active coils.
        gain_p : float
            Proportional gain [V/m].
        gain_d : float
            Derivative gain [V s/m].
        z_target : float
            Target vertical position of the magnetic axis [m].
        base_voltages : np.ndarray or callable, optional
            Base voltage waveform (constant vector or `base_voltages(time)`).
            Defaults to the steady-state voltages of the evolution's initial
            currents.
        """
        self.coil_index = coil_index
        self.gain_p = gain_p
        self.gain_d = gain_d
        self.z_target = z_target
        self.base_voltages = base_voltages
        self._previous = None

    def __call__(self, time, evolution):
        if self.base_voltages is None:
            base = evolution.steady_state_voltages(evolution.history[0]["currents"])
        elif callable(self.base_voltages):
            base = np.asarray(self.base_voltages(time), dtype=float)
        else:
            base = np.asarray(self.base_voltages, dtype=float)
        z_axis = evolution.history[-1]["Z_axis"]
        dz_dt = 0.0
        if self._previous is not None and time > self._previous[0]:
            dz_dt = (z_axis - self._previous[1]) / (time - self._previous[0])
        self._previous = (time, z_axis)
        voltages = np.copy(base)
        voltages[self.coil_index] += (
            self.gain_p * (z_axis - self.z_target) + self.gain_d * dz_dt
        )
        return voltages
