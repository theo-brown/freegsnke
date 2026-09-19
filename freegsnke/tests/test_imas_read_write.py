"""Tests for writing FreeGSNKE equilibria to IMAS IDSs and reading profiles back."""

import numpy as np
import pytest

from freegsnke import imas_read_write
from freegsnke.jtor_update import GeneralPprimeFFprime


@pytest.fixture(scope="module")
def solved_equilibrium(solved_test_equilibrium):
    """A forward-solved diverted equilibrium on the test machine."""
    return solved_test_equilibrium


@pytest.fixture(scope="module")
def equilibrium_ids(solved_equilibrium):
    eq, profiles = solved_equilibrium
    return imas_read_write.write_equilibrium_to_ids(eq, profiles, time=0.25)


def test_ids_flux_conventions(solved_equilibrium, equilibrium_ids):
    """The IDS stores psi in Wb (2*pi times FreeGSNKE's Wb/rad) and the
    derivatives p' and FF' with respect to that flux."""
    eq, profiles = solved_equilibrium
    time_slice = equilibrium_ids.time_slice[0]
    psi_n = np.asarray(time_slice.profiles_1d.psi_norm)

    assert float(equilibrium_ids.time[0]) == 0.25
    assert float(time_slice.time) == 0.25
    np.testing.assert_allclose(
        time_slice.global_quantities.psi_axis, 2 * np.pi * eq.psi_axis
    )
    np.testing.assert_allclose(
        time_slice.global_quantities.psi_boundary, 2 * np.pi * eq.psi_bndry
    )
    np.testing.assert_allclose(
        np.asarray(time_slice.profiles_2d[0].psi), 2 * np.pi * eq.psi()
    )
    np.testing.assert_allclose(
        np.asarray(time_slice.profiles_1d.dpressure_dpsi),
        profiles.pprime(psi_n) / (2 * np.pi),
    )
    np.testing.assert_allclose(
        np.asarray(time_slice.profiles_1d.f_df_dpsi),
        profiles.ffprime(psi_n) / (2 * np.pi),
    )
    np.testing.assert_allclose(time_slice.global_quantities.ip, eq.plasmaCurrent())


def test_ids_profiles_start_on_axis(solved_equilibrium, equilibrium_ids):
    """The 1D profiles include the magnetic axis with sensible axis values and
    are monotonic where they should be."""
    eq, profiles = solved_equilibrium
    profiles_1d = equilibrium_ids.time_slice[0].profiles_1d
    psi_n = np.asarray(profiles_1d.psi_norm)
    mag_r, mag_z = eq.magneticAxis()[0:2]

    assert psi_n[0] == 0.0
    assert psi_n[-1] < 1.0
    assert np.all(np.diff(psi_n) > 0)
    assert np.asarray(profiles_1d.phi)[0] == 0.0
    assert np.asarray(profiles_1d.rho_tor_norm)[0] == 0.0
    assert np.all(np.diff(np.asarray(profiles_1d.phi)) > 0)
    assert np.all(np.diff(np.asarray(profiles_1d.volume)) > 0)
    np.testing.assert_allclose(np.asarray(profiles_1d.r_inboard)[0], mag_r)
    np.testing.assert_allclose(np.asarray(profiles_1d.r_outboard)[0], mag_r)
    np.testing.assert_allclose(np.asarray(profiles_1d.gm1)[0], 1 / mag_r**2)
    np.testing.assert_allclose(np.asarray(profiles_1d.gm9)[0], 1 / mag_r)
    np.testing.assert_allclose(np.asarray(profiles_1d.gm3)[0], 1.0)
    np.testing.assert_allclose(np.asarray(profiles_1d.gm7)[0], 1.0)
    # the axis current density is the current density at the magnetic axis
    i_r = np.argmin(np.abs(eq.R_1D - mag_r))
    i_z = np.argmin(np.abs(eq.Z_1D - mag_z))
    np.testing.assert_allclose(
        np.asarray(profiles_1d.j_phi)[0], profiles.jtor[i_r, i_z], rtol=0.02
    )
    for name in ["q", "gm2", "gm3", "gm4", "gm5", "gm7", "f", "pressure"]:
        values = np.asarray(getattr(profiles_1d, name))
        assert values.shape == psi_n.shape
        assert np.all(np.isfinite(values)), name


def test_integrated_profiles_match_quadrature(solved_equilibrium):
    """The vectorised integration of p' and FF' matches the freegs4e
    quadrature-based pressure and fpol."""
    eq, profiles = solved_equilibrium
    psi_n = np.array([0.0, 0.1, 0.45, 0.8, 0.99])
    pressure, fpol = imas_read_write._integrated_profiles(
        profiles, psi_n, eq.psi_axis, eq.psi_bndry
    )
    np.testing.assert_allclose(pressure, profiles.pressure(psi_n), rtol=1e-5, atol=1.0)
    np.testing.assert_allclose(fpol, profiles.fpol(psi_n), rtol=1e-6)


def test_read_profiles_roundtrip(solved_equilibrium, equilibrium_ids):
    """Profiles read back from the IDS are in FreeGSNKE's conventions."""
    eq, profiles = solved_equilibrium
    data = imas_read_write.read_profiles_from_equilibrium_ids(equilibrium_ids)
    np.testing.assert_allclose(data["pprime"], profiles.pprime(data["psi_n"]))
    np.testing.assert_allclose(data["ffprime"], profiles.ffprime(data["psi_n"]))
    np.testing.assert_allclose(data["Ip"], eq.plasmaCurrent())
    np.testing.assert_allclose(data["fvac"], profiles.fvac())
    np.testing.assert_allclose(data["psi_axis"], eq.psi_axis)
    np.testing.assert_allclose(data["psi_bndry"], eq.psi_bndry)
    assert data["time"] == 0.25


def test_read_profiles_flipped_convention(solved_equilibrium, equilibrium_ids):
    """An IDS written with the opposite plasma current sign (psi increasing from
    axis to boundary, as e.g. written by TORAX) reads back to the same
    FreeGSNKE profiles."""
    eq, profiles = solved_equilibrium
    flipped = imas_read_write.write_equilibrium_to_ids(eq, profiles)
    time_slice = flipped.time_slice[0]
    time_slice.profiles_1d.psi = -np.asarray(time_slice.profiles_1d.psi)
    time_slice.global_quantities.psi_axis = -time_slice.global_quantities.psi_axis
    time_slice.global_quantities.psi_boundary = (
        -time_slice.global_quantities.psi_boundary
    )
    time_slice.global_quantities.ip = -time_slice.global_quantities.ip
    time_slice.profiles_1d.dpressure_dpsi = -np.asarray(
        time_slice.profiles_1d.dpressure_dpsi
    )
    time_slice.profiles_1d.f_df_dpsi = -np.asarray(time_slice.profiles_1d.f_df_dpsi)
    flipped.vacuum_toroidal_field.b0 = -np.asarray(flipped.vacuum_toroidal_field.b0)

    data = imas_read_write.read_profiles_from_equilibrium_ids(flipped)
    reference = imas_read_write.read_profiles_from_equilibrium_ids(equilibrium_ids)
    for key in ["psi_n", "pprime", "ffprime", "Ip", "fvac", "psi_axis", "psi_bndry"]:
        np.testing.assert_allclose(data[key], reference[key], err_msg=key)


def test_profiles_from_ids_reproduce_current_density(
    solved_equilibrium, equilibrium_ids
):
    """A GeneralPprimeFFprime profile built from the IDS reproduces the
    current density of the original profile object."""
    eq, profiles = solved_equilibrium
    new_profiles = imas_read_write.profiles_from_equilibrium_ids(eq, equilibrium_ids)
    assert isinstance(new_profiles, GeneralPprimeFFprime)
    jtor_reference = profiles.Jtor(eq.R, eq.Z, eq.psi())
    jtor_new = new_profiles.Jtor(eq.R, eq.Z, eq.psi())
    np.testing.assert_allclose(
        jtor_new, jtor_reference, atol=0.01 * np.max(np.abs(jtor_reference))
    )
    np.testing.assert_allclose(np.sum(jtor_new), np.sum(jtor_reference), rtol=1e-6)

    # in-place update gives the same result
    updated = imas_read_write.update_profiles_from_equilibrium_ids(
        new_profiles, equilibrium_ids
    )
    assert updated is new_profiles
    np.testing.assert_allclose(updated.Jtor(eq.R, eq.Z, eq.psi()), jtor_new)


def test_write_rejects_invalid_psi_n(solved_equilibrium):
    eq, profiles = solved_equilibrium
    with pytest.raises(ValueError):
        imas_read_write.write_equilibrium_to_ids(eq, profiles, psi_n=[0.0, 0.5])
    with pytest.raises(ValueError):
        imas_read_write.write_equilibrium_to_ids(eq, profiles, psi_n=[0.5, 0.4])


def test_save_and_load_ids(equilibrium_ids, tmp_path):
    path = str(tmp_path / "equilibrium.nc")
    imas_read_write.save_equilibrium_ids(equilibrium_ids, path)
    loaded = imas_read_write.load_equilibrium_ids(path)
    np.testing.assert_allclose(
        np.asarray(loaded.time_slice[0].profiles_1d.dpressure_dpsi),
        np.asarray(equilibrium_ids.time_slice[0].profiles_1d.dpressure_dpsi),
    )


def test_edge_taper(solved_equilibrium, equilibrium_ids):
    """The edge taper brings p' and FF' smoothly to zero at the separatrix and
    leaves the profiles untouched away from it."""
    psi_n = np.linspace(0.0, 1.0, 101)
    taper = imas_read_write.edge_taper(psi_n, 0.02)
    assert taper[-1] == 0.0
    assert np.all(taper[psi_n <= 0.98] == 1.0)
    assert np.all(np.diff(taper[psi_n >= 0.98]) <= 0.0)
    np.testing.assert_array_equal(imas_read_write.edge_taper(psi_n, 0.0), 1.0)

    plain = imas_read_write.read_profiles_from_equilibrium_ids(equilibrium_ids)
    tapered = imas_read_write.read_profiles_from_equilibrium_ids(
        equilibrium_ids, edge_taper_width=0.05
    )
    inside = plain["psi_n"] < 0.95
    np.testing.assert_array_equal(tapered["pprime"][inside], plain["pprime"][inside])
    np.testing.assert_array_equal(tapered["ffprime"][inside], plain["ffprime"][inside])
    assert (
        abs(tapered["pprime"][-1]) < abs(plain["pprime"][-1])
        or plain["pprime"][-1] == 0.0
    )


def test_written_q_matches_toroidal_flux(solved_test_equilibrium):
    """The toroidal flux phi = int 2 pi q dpsi built from the written q must
    agree with the toroidal flux enclosed by the surfaces of the writer's own
    psi_n normalisation (independently computed here by summing F/R over grid
    cells). `eq.q` would instead normalise the flux to the X-point, which is
    not the plasma boundary for a limiter-bound plasma."""
    eq, profiles = solved_test_equilibrium
    ids = imas_read_write.write_equilibrium_to_ids(eq, profiles)
    profiles_1d = ids.time_slice[0].profiles_1d
    psi_n_ids = np.asarray(profiles_1d.psi_norm)
    phi_ids = np.asarray(profiles_1d.phi)
    psi_n_2d = (eq.psi() - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)
    inside = eq.limiter_handler.mask_inside_limiter
    fpol_2d = profiles.fpol(np.clip(psi_n_2d, 0.0, 1.0))
    dA = (eq.R[1, 0] - eq.R[0, 0]) * (eq.Z[0, 1] - eq.Z[0, 0])
    # (5% allows for the cell counting on the coarse test grid; the X-point
    # normalisation of `eq.q` gives errors of 10-20% for limited plasmas)
    for target in (0.5, 0.7, 0.9):
        phi_direct = np.sum((fpol_2d / eq.R)[inside & (psi_n_2d < target)]) * dA
        np.testing.assert_allclose(
            np.interp(target, psi_n_ids, phi_ids), phi_direct, rtol=0.05
        )
