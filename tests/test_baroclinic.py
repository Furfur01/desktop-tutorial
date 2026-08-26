from __future__ import annotations

from dataclasses import replace

import numpy as np

from atmos20 import AtmosphereModel, ModelConfig
from atmos20.baroclinic import (
    DryBaroclinicWaveSpec,
    build_baroclinic_wave_initial_state,
    configure_baroclinic_wave_model,
)


def test_unperturbed_state_is_balanced_stable_and_in_range() -> None:
    config = ModelConfig(dlon_deg=5.0, dlat_deg=2.5)
    state = build_baroclinic_wave_initial_state(config, add_perturbation=False)

    assert state.u.shape == (config.n_levels, 63, 72)
    assert np.isfinite(state.u).all()
    assert np.isfinite(state.temperature_k).all()
    assert np.isfinite(state.geopotential_m2_s2).all()
    assert np.count_nonzero(state.v) == 0
    assert np.count_nonzero(state.zonal_wind_perturbation) == 0
    assert np.all(state.surface_pressure_pa == 100_000.0)

    # The unperturbed basic state is zonally symmetric and meteorological in
    # magnitude.  No low, front, or vortex has been inserted.
    assert np.max(np.ptp(state.u, axis=2)) < 1.0e-12
    assert np.max(np.ptp(state.temperature_k, axis=2)) < 1.0e-12
    assert 34.0 < float(np.max(state.u)) < 36.0
    assert 200.0 < float(np.min(state.temperature_k)) < 215.0
    assert 300.0 < float(np.max(state.temperature_k)) < 315.0

    pressure_pa = config.pressure_levels_hpa * 100.0
    potential_temperature = state.temperature_k * (
        100_000.0 / pressure_pa[:, None, None]
    ) ** config.kappa
    # Levels are ordered from the surface upward, so positive increments are
    # static stability rather than a manually imposed frontal discontinuity.
    assert float(np.min(np.diff(potential_temperature, axis=0))) > 0.5

    latitude_rad = np.deg2rad(
        np.arange(-config.lat_limit_deg, config.lat_limit_deg + 0.1, config.dlat_deg)
    )
    phi_gradient = np.gradient(
        state.geopotential_m2_s2,
        latitude_rad,
        axis=1,
        edge_order=2,
    ) / config.earth_radius_m
    f = 2.0 * config.rotation_rate_s * np.sin(latitude_rad)[None, :, None]
    metric = (
        state.u
        * np.tan(latitude_rad)[None, :, None]
        / config.earth_radius_m
    )
    gradient_wind_residual = phi_gradient + (f + metric) * state.u
    assert float(np.max(np.abs(gradient_wind_residual[:, 2:-2, :]))) < 1.5e-5

    hydrostatic_derivative = np.gradient(
        state.geopotential_m2_s2,
        np.log(pressure_pa),
        axis=0,
        edge_order=2,
    )
    hydrostatic_residual = (
        hydrostatic_derivative + config.gas_constant_dry_air * state.temperature_k
    )
    hydrostatic_scale = float(
        np.max(config.gas_constant_dry_air * state.temperature_k[1:-1])
    )
    assert (
        float(np.max(np.abs(hydrostatic_residual[1:-1]))) / hydrostatic_scale
        < 0.012
    )


def test_local_wave_trigger_is_repeatable_and_selectable_by_hemisphere() -> None:
    config = ModelConfig(dlon_deg=2.5, dlat_deg=2.5)
    base_spec = DryBaroclinicWaveSpec(perturbation_hemisphere="north")
    north_a = build_baroclinic_wave_initial_state(config, spec=base_spec)
    north_b = build_baroclinic_wave_initial_state(config, spec=base_spec)
    south = build_baroclinic_wave_initial_state(
        config,
        spec=replace(base_spec, perturbation_hemisphere="south"),
    )

    np.testing.assert_array_equal(north_a.u, north_b.u)
    np.testing.assert_array_equal(north_a.temperature_k, north_b.temperature_k)
    np.testing.assert_array_equal(north_a.surface_pressure_pa, south.surface_pressure_pa)
    np.testing.assert_array_equal(north_a.temperature_k, south.temperature_k)

    north_trigger = north_a.zonal_wind_perturbation[0]
    south_trigger = south.zonal_wind_perturbation[0]
    north_peak = np.unravel_index(np.argmax(north_trigger), north_trigger.shape)
    south_peak = np.unravel_index(np.argmax(south_trigger), south_trigger.shape)
    latitude = np.arange(-77.5, 77.5 + 0.1, 2.5)
    longitude = np.arange(0.0, 360.0, 2.5)

    assert latitude[north_peak[0]] == 40.0
    assert latitude[south_peak[0]] == -40.0
    assert longitude[north_peak[1]] == 20.0
    assert longitude[south_peak[1]] == 20.0
    assert np.isclose(float(np.max(north_trigger)), 1.0)
    np.testing.assert_allclose(north_trigger[::-1], south_trigger, atol=2.0e-14)

    # The trigger changes only u.  Pressure and temperature must evolve into
    # a cyclone/front dynamically rather than being present at t=0.
    np.testing.assert_allclose(
        north_a.u - north_a.balanced_u,
        north_a.zonal_wind_perturbation,
        atol=5.0e-15,
        rtol=0.0,
    )
    assert np.count_nonzero(north_a.v) == 0


def test_configure_model_installs_matching_dry_experiment() -> None:
    model = AtmosphereModel(
        ModelConfig(dlon_deg=10.0, dlat_deg=5.0, dt_seconds=300.0)
    )
    model.time_seconds = 1234.0
    state = configure_baroclinic_wave_model(model, add_perturbation=False)

    np.testing.assert_allclose(model.u, state.u)
    np.testing.assert_allclose(model.v, state.v)
    np.testing.assert_allclose(model.temperature_k, state.temperature_k)
    np.testing.assert_allclose(
        model.boundary.base_surface_pressure_pa,
        state.surface_pressure_pa,
    )
    np.testing.assert_allclose(
        model.boundary.surface_elevation_m * model.config.gravity_m_s2,
        state.surface_geopotential_m2_s2,
    )
    assert model.active.all()
    assert not model.boundary.land_mask.any()
    assert model.time_seconds == 0.0
    assert model.config.horizontal_diffusion_rate_s == 0.0
    assert model.config.vertical_mixing_rate_s == 0.0
    assert model.config.surface_drag_ocean_s == 0.0
    assert model.config.orographic_lift_scale == 0.0
    assert model.config.surface_pressure_coupling == 1.0
    assert np.count_nonzero(model.radiative_rate_s) == 0
    assert model._meridional_sponge_rate_s is not None
    equator = int(np.argmin(np.abs(model.grid.lat_deg)))
    assert np.all(model._meridional_sponge_rate_s[equator] == 0.0)
    assert np.isclose(
        float(np.max(model._meridional_sponge_rate_s)),
        1.0 / (3.0 * 3600.0),
    )

    tendencies = model._rhs(
        model.u,
        model.v,
        model.temperature_k,
        model.surface_pressure_anomaly_pa,
    )
    assert float(np.max(np.abs(tendencies.u))) < 1.0e-12
    assert float(np.max(np.abs(tendencies.temperature))) < 1.0e-12
    assert float(np.max(np.abs(tendencies.surface_pressure_anomaly))) < 1.0e-12
    # Remaining meridional acceleration is discrete pressure-gradient error,
    # not an imbalance in the analytic initial condition.
    assert float(np.max(np.abs(tendencies.v))) < 1.0e-4
