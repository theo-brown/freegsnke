"""Tests for the loose coupling of FreeGSNKE with TORAX (skipped without torax)."""

import numpy as np
import pytest

torax = pytest.importorskip("torax")

from freegsnke import imas_read_write, torax_coupling  # noqa: E402


def _PARABOLIC_PROFILE(axis_value, edge_value, n_points=21):
    """{rho_norm: value} for a parabolic profile between axis and edge."""
    rho = np.linspace(0.0, 1.0, n_points)
    return {
        float(r): float(edge_value + (axis_value - edge_value) * (1 - r**2))
        for r in rho
    }


# A small TORAX configuration for a MAST-U-like plasma matching the test
# machine equilibrium (Ip = 0.62 MA, p_axis ~ 8 kPa). The geometry section only
# sets the radial mesh: the geometry itself comes from FreeGSNKE.
TORAX_CONFIG = {
    "profile_conditions": {
        "Ip": 6.2e5,
        # smooth (parabolic) profiles: profiles with a cusp on axis give a
        # divergent dp/dpsi there, which the equilibrium solver cannot handle
        "T_i": {0.0: _PARABOLIC_PROFILE(0.6, 0.1)},
        "T_e": {0.0: _PARABOLIC_PROFILE(0.6, 0.1)},
        "T_i_right_bc": 0.1,
        "T_e_right_bc": 0.1,
        "n_e": {0.0: _PARABOLIC_PROFILE(3.9e19, 1.8e19)},
        "nbar": 3e19,
        "n_e_nbar_is_fGW": False,
        "n_e_right_bc": 1e19,
        "n_e_right_bc_is_fGW": False,
        # initial poloidal flux from the FreeGSNKE equilibrium
        "initial_psi_mode": "geometry",
    },
    "plasma_composition": {},
    "numerics": {
        "t_initial": 0.0,
        "t_final": 0.02,
        "fixed_dt": 0.005,
        "adaptive_dt": False,
        "evolve_current": True,
        "evolve_density": False,
    },
    "geometry": {
        "geometry_type": "circular",
        "n_rho": 20,
        "R_major": 0.87,
        "a_minor": 0.53,
        "B_0": 0.58,
    },
    "neoclassical": {"bootstrap_current": {}},
    "sources": {"generic_heat": {"P_total": 2.0e5}, "ei_exchange": {}, "ohmic": {}},
    "transport": {
        "model_name": "combined",
        "transport_models": [{"model_name": "constant"}],
    },
    "solver": {"solver_type": "linear"},
    "pedestal": {},
    "time_step_calculator": {"calculator_type": "fixed"},
}


@pytest.fixture(scope="module")
def torax_config():
    return torax.ToraxConfig.from_dict(TORAX_CONFIG)


def test_default_psi_n_grid():
    psi_n = torax_coupling.default_psi_n_grid(n_points=50)
    assert len(psi_n) == 50
    assert psi_n[0] > 0.0 and psi_n[-1] < 1.0
    assert np.all(np.diff(psi_n) > 0)
    # uniform in sqrt(psi_n)
    np.testing.assert_allclose(np.diff(np.sqrt(psi_n)), np.diff(np.sqrt(psi_n))[0])


def test_torax_geometry_from_ids(solved_test_equilibrium, torax_config):
    """TORAX builds a consistent geometry from the FreeGSNKE equilibrium IDS."""
    eq, profiles = solved_test_equilibrium
    ids = imas_read_write.write_equilibrium_to_ids(
        eq, profiles, psi_n=torax_coupling.default_psi_n_grid()
    )
    geo = torax_coupling.torax_geometry_from_ids(
        ids, torax_config, Ip_from_parameters=False
    )
    assert len(geo.rho_face_norm) == TORAX_CONFIG["geometry"]["n_rho"] + 1
    np.testing.assert_allclose(geo.Ip_profile_face[-1], eq.plasmaCurrent(), rtol=1e-6)
    separatrix = eq.separatrix()
    R_major = 0.5 * (separatrix[:, 0].max() + separatrix[:, 0].min())
    np.testing.assert_allclose(geo.R_major, R_major, rtol=0.02)
    np.testing.assert_allclose(geo.B_0, profiles.fvac() / R_major, rtol=0.02)
    assert np.all(np.diff(np.asarray(geo.volume_face)) > 0)
    np.testing.assert_allclose(geo.volume_face[-1], eq.plasmaVolume(), rtol=0.05)
    # the current density seen by TORAX matches FreeGSNKE's flux-averaged jtor
    j_phi = np.asarray(ids.time_slice[0].profiles_1d.j_phi)
    rho_ids = np.asarray(ids.time_slice[0].profiles_1d.rho_tor_norm)
    j_torax = np.asarray(geo.j_total_face)
    rho_torax = np.asarray(geo.rho_face_norm)
    interior = (rho_torax > 0.15) & (rho_torax < 0.9)
    np.testing.assert_allclose(
        j_torax[interior], np.interp(rho_torax[interior], rho_ids, j_phi), rtol=0.05
    )


def test_profile_residual_and_relaxation(solved_test_equilibrium):
    eq, profiles = solved_test_equilibrium
    ids_a = imas_read_write.write_equilibrium_to_ids(eq, profiles)
    ids_b = imas_read_write.write_equilibrium_to_ids(eq, profiles)
    assert torax_coupling.profile_residual(ids_a, ids_b) == 0.0
    profiles_1d = ids_b.time_slice[0].profiles_1d
    profiles_1d.dpressure_dpsi = 1.1 * np.asarray(profiles_1d.dpressure_dpsi)
    residual = torax_coupling.profile_residual(ids_b, ids_a)
    assert 0.0 < residual < 0.1
    relaxed = torax_coupling.relax_profiles(ids_b, ids_a, relaxation=0.5)
    np.testing.assert_allclose(
        np.asarray(relaxed.time_slice[0].profiles_1d.dpressure_dpsi),
        1.05 * np.asarray(ids_a.time_slice[0].profiles_1d.dpressure_dpsi),
    )


def test_run_loose_coupling(solved_test_equilibrium, torax_config):
    """A short loosely coupled run completes with consistent currents."""
    eq, profiles = solved_test_equilibrium
    eq = eq.create_auxiliary_equilibrium()
    equilibrium_solver = torax_coupling.StaticEquilibriumSolver(
        eq, profiles, target_relative_tolerance=1e-6
    )
    result = torax_coupling.run_loose_coupling(
        torax_config,
        equilibrium_solver,
        coupling_dt=0.01,
        max_iterations=3,
        tolerance=1e-2,
        initial_iterations=1,
        store_equilibria=True,
        verbose=False,
    )
    assert result.sim_error == torax.SimError.NO_ERROR
    np.testing.assert_allclose(result.times, [0.0, 0.01, 0.02])
    assert len(result.equilibrium_ids) == 3
    assert len(result.torax_equilibrium_ids) == 3
    assert len(result.equilibria) == 3
    assert len(result.residuals) == 3
    assert np.all(result.iterations >= 1) and np.all(result.iterations <= 3)
    assert equilibrium_solver.n_solves == int(np.sum(result.iterations))
    # TORAX time coordinate matches the coupling times
    np.testing.assert_allclose(result.torax_output["time"].values, result.times)
    # the plasma current in the final FreeGSNKE equilibrium is the TORAX one
    Ip_torax = float(result.torax_output["scalars"]["Ip"].values[-1])
    np.testing.assert_allclose(
        result.equilibrium_ids[-1].time_slice[0].global_quantities.ip,
        Ip_torax,
        rtol=1e-6,
    )
    np.testing.assert_allclose(eq.plasmaCurrent(), Ip_torax, rtol=1e-6)
    # the p' handed to FreeGSNKE and the geometry handed to TORAX are finite
    for ids in result.torax_equilibrium_ids + result.equilibrium_ids:
        profiles_1d = ids.time_slice[0].profiles_1d
        assert np.all(np.isfinite(np.asarray(profiles_1d.dpressure_dpsi)))
        assert np.all(np.isfinite(np.asarray(profiles_1d.f_df_dpsi)))


def test_run_loose_coupling_rejects_bad_arguments(
    solved_test_equilibrium, torax_config
):
    eq, profiles = solved_test_equilibrium
    equilibrium_solver = torax_coupling.StaticEquilibriumSolver(eq, profiles)
    with pytest.raises(ValueError):
        torax_coupling.run_loose_coupling(
            torax_config, equilibrium_solver, coupling_dt=0.01, relaxation=0.0
        )
    with pytest.raises(ValueError):
        torax_coupling.run_loose_coupling(
            torax_config, equilibrium_solver, coupling_dt=0.01, max_iterations=0
        )


def test_run_loose_coupling_evolutive(solved_test_equilibrium, torax_config):
    """A short loosely coupled run with the vessel-timescale equilibrium
    evolution completes, commits its state and keeps Ip consistent."""
    eq, profiles = solved_test_equilibrium
    eq = eq.create_auxiliary_equilibrium()
    profiles = profiles.copy()
    from freegsnke import GSstaticsolver

    solver = GSstaticsolver.NKGSsolver(eq)
    solver.forward_solve(eq, profiles, 1e-9)
    config = torax.ToraxConfig.from_dict(
        {**TORAX_CONFIG, "numerics": {**TORAX_CONFIG["numerics"], "t_final": 0.004}}
    )
    equilibrium_solver = torax_coupling.EvolutiveEquilibriumSolver(
        eq, profiles, solver=solver, vessel_timestep=1e-3, verbose=False
    )
    result = torax_coupling.run_loose_coupling(
        config,
        equilibrium_solver,
        coupling_dt=0.004,
        max_iterations=2,
        tolerance=1e-2,
        initial_iterations=1,
        verbose=False,
    )
    assert result.sim_error == torax.SimError.NO_ERROR
    np.testing.assert_allclose(result.times, [0.0, 0.004])
    # the evolution was committed at the end of the interval, in 4 sub-steps
    assert equilibrium_solver.committed_state.time == pytest.approx(0.004)
    times = [h["time"] for h in equilibrium_solver.substep_history]
    np.testing.assert_allclose(times, np.linspace(0.0, 0.004, 5), atol=1e-12)
    Ip_torax = float(result.torax_output["scalars"]["Ip"].values[-1])
    np.testing.assert_allclose(
        equilibrium_solver.substep_history[-1]["Ip"], Ip_torax, rtol=1e-6
    )
    np.testing.assert_allclose(eq.plasmaCurrent(), Ip_torax, rtol=1e-6)
    # passive structures carry induced currents after the interval
    passive = equilibrium_solver.evolution.currents[
        equilibrium_solver.evolution.n_active_coils :
    ]
    assert np.all(np.isfinite(passive))
