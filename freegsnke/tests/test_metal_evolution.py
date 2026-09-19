"""Tests for the vessel-timescale metal current evolution with prescribed plasma."""

import numpy as np
import pytest

from freegsnke import GSstaticsolver
from freegsnke.metal_evolution import MetalCurrentsEvolution, VerticalPositionController


@pytest.fixture(scope="module")
def evolution(solved_test_equilibrium):
    eq, profiles = solved_test_equilibrium
    eq = eq.create_auxiliary_equilibrium()
    profiles = profiles.copy()
    solver = GSstaticsolver.NKGSsolver(eq)
    solver.forward_solve(eq, profiles, 1e-9)
    return MetalCurrentsEvolution(
        eq, profiles, solver=solver, vessel_timestep=5e-4, verbose=False
    )


def test_mode_selection_and_initial_state(evolution):
    metal = evolution.metal
    assert evolution.n_active_coils == 12
    # active coils are always retained, only the slow passive modes are kept
    assert evolution.n_active_coils < evolution.n_modes < evolution.n_coils
    assert np.all(
        metal.normal_modes.w_passive[metal.selected_modes_mask[12:]]
        < evolution.max_mode_frequency
    )
    # the initial currents are represented exactly in the truncated mode basis
    np.testing.assert_allclose(
        metal.IdtoIvessel(evolution.state.Id), evolution.state.currents, atol=1e-9
    )
    np.testing.assert_allclose(
        np.sum(evolution.state.Iy), evolution.profiles.Ip, rtol=1e-6
    )
    assert len(evolution.history) == 1


def test_steady_state_is_invariant(evolution):
    """With the steady-state voltages and unchanged profiles nothing moves."""
    start = evolution.snapshot()
    voltages = evolution.steady_state_voltages()
    infos = evolution.advance(start.time + 2e-3, voltages)
    assert len(infos) == 4
    assert all(info["converged"] for info in infos)
    np.testing.assert_allclose(
        evolution.currents, start.currents, rtol=1e-10, atol=1e-6
    )
    np.testing.assert_allclose(evolution.state.Iy, start.Iy, rtol=1e-8)
    assert evolution.time == pytest.approx(start.time + 2e-3)
    evolution.restore(start)
    np.testing.assert_allclose(evolution.currents, start.currents)
    assert evolution.time == start.time
    assert len(evolution.history) == 1


def test_current_ramp_induces_vessel_currents(evolution):
    """Ramping the plasma current induces opposing currents in the passives."""
    start = evolution.snapshot()
    Ip_0 = float(np.sum(start.Iy))

    def update_profiles(profiles, time):
        profiles.Ip = Ip_0 * (1.0 + 0.02 * (time - start.time) / 2e-3)

    voltages = evolution.steady_state_voltages()
    infos = evolution.advance(
        start.time + 2e-3, voltages, update_profiles=update_profiles
    )
    assert all(info["converged"] for info in infos)
    assert all(info["n_solves"] <= 15 for info in infos)
    np.testing.assert_allclose(np.sum(evolution.state.Iy), 1.02 * Ip_0, rtol=1e-6)
    passive = evolution.currents[evolution.n_active_coils :]
    assert np.abs(passive).sum() > 0.0
    # the net induced passive current opposes the plasma current increase
    assert np.sum(passive) < 0.0
    evolution.restore(start)


def test_vertical_controller_uses_axis_position(evolution):
    base = np.zeros(evolution.n_active_coils)
    controller = VerticalPositionController(
        coil_index=3, gain_p=-100.0, gain_d=0.0, z_target=0.0, base_voltages=base
    )
    z = evolution.history[-1]["Z_axis"]
    voltages = controller(evolution.time, evolution)
    assert voltages.shape == base.shape
    np.testing.assert_allclose(voltages[3], -100.0 * z)
    assert np.all(voltages[:3] == 0.0) and np.all(voltages[4:] == 0.0)


def test_step_rejects_non_positive_dt(evolution):
    with pytest.raises(ValueError):
        evolution.step(0.0, evolution.steady_state_voltages())
