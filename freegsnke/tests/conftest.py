"""Shared pytest fixtures for the FreeGSNKE test suite."""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = Path(__file__).resolve().parent / "baselines"
MACHINE_CONFIG_DIR = REPO_ROOT / "machine_configs" / "test"
STATIC_CURRENT_BASELINE = TEST_DATA_DIR / "test_controlCurrents.npy"


@pytest.fixture(scope="session")
def solved_test_equilibrium():
    """
    A forward-solved diverted equilibrium on the test machine, using the
    baseline control currents of the static solver test.

    Returns
    -------
    tuple
        (eq, profiles): the solved `Equilibrium` and the `ConstrainPaxisIp`
        profile object used to solve it.
    """
    from freegsnke import GSstaticsolver, build_machine, equilibrium_update
    from freegsnke.jtor_update import ConstrainPaxisIp

    tokamak = build_machine.tokamak(
        active_coils_path=str(MACHINE_CONFIG_DIR / "active_coils.pickle"),
        passive_coils_path=str(MACHINE_CONFIG_DIR / "passive_coils.pickle"),
        limiter_path=str(MACHINE_CONFIG_DIR / "limiter.pickle"),
        wall_path=str(MACHINE_CONFIG_DIR / "wall.pickle"),
        magnetic_probe_path=str(MACHINE_CONFIG_DIR / "magnetic_probes.pickle"),
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.1, Rmax=2.0, Zmin=-2.2, Zmax=2.2, nx=65, ny=129
    )
    profiles = ConstrainPaxisIp(eq, 8.1e3, 6.2e5, 0.5, alpha_m=1.8, alpha_n=1.2)
    eq.tokamak.set_coil_current("P6", 0)
    eq.tokamak["P6"].control = False
    eq.tokamak["Solenoid"].control = False
    eq.tokamak.set_coil_current("Solenoid", 15000)
    eq.tokamak.setControlCurrents(np.load(STATIC_CURRENT_BASELINE))
    solver = GSstaticsolver.NKGSsolver(eq)
    solver.forward_solve(eq, profiles, 1e-8)
    return eq, profiles
