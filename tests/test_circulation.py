from __future__ import annotations

import numpy as np

from atmos20 import (
    AtmosphereModel,
    HeldSuarezSpec,
    ModelConfig,
    OrographicCirculationSpec,
    configure_held_suarez_circulation,
    configure_orographic_held_suarez_circulation,
    wind_belt_statistics,
)


def _model() -> AtmosphereModel:
    return AtmosphereModel(
        ModelConfig(
            dlon_deg=30.0,
            dlat_deg=10.0,
            lat_limit_deg=80.0,
            pressure_step_hpa=100,
            dt_seconds=300.0,
        )
    )


def test_held_suarez_forcing_matches_canonical_limits() -> None:
    model = _model()
    forcing = configure_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
    )
    equator = int(np.argmin(np.abs(model.grid.lat_deg)))
    poleward = int(np.argmax(model.grid.lat_deg))
    bottom = model.level_index(1000)
    upper = model.level_index(500)

    assert np.isclose(forcing.equilibrium_temperature_k[bottom, equator, 0], 315.0)
    assert forcing.equilibrium_temperature_k[bottom, poleward, 0] < 260.0
    assert np.min(forcing.equilibrium_temperature_k) >= 200.0
    assert np.isclose(
        forcing.newtonian_relaxation_rate_s[bottom, equator, 0],
        1.0 / (4.0 * 86_400.0),
    )
    assert np.isclose(
        forcing.newtonian_relaxation_rate_s[upper, equator, 0],
        1.0 / (40.0 * 86_400.0),
    )
    assert np.isclose(
        forcing.rayleigh_drag_rate_s[bottom, equator, 0],
        1.0 / 86_400.0,
    )
    assert forcing.rayleigh_drag_rate_s[upper, equator, 0] == 0.0


def test_configuration_is_flat_axisymmetric_aquaplanet() -> None:
    model = _model()
    forcing = configure_held_suarez_circulation(model)

    assert model.active.all()
    assert not model.boundary.land_mask.any()
    assert np.count_nonzero(model.boundary.surface_elevation_m) == 0
    assert np.all(model.boundary.base_surface_pressure_pa == 100_000.0)
    assert np.count_nonzero(model.boundary.seasonal_surface_pressure_anomaly_pa) == 0
    assert np.count_nonzero(model.u) == 0
    assert np.count_nonzero(model.v) == 0
    assert model.config.orographic_lift_scale == 0.0
    assert model.config.terrain_blocking_rate_s == 0.0
    assert model.config.mass_damping_rate_s == 0.0
    assert model.config.surface_pressure_coupling == 1.0

    # Every prescribed field is longitude independent.  Only the explicitly
    # documented small random temperature seed breaks zonal symmetry.
    assert np.max(np.ptp(forcing.equilibrium_temperature_k, axis=2)) == 0.0
    assert np.max(np.ptp(forcing.newtonian_relaxation_rate_s, axis=2)) == 0.0
    assert np.max(np.ptp(forcing.rayleigh_drag_rate_s, axis=2)) == 0.0
    weighted_noise = (
        model.temperature_k - HeldSuarezSpec().initial_temperature_k
    ) * model.grid.area_weight[None, :, :]
    np.testing.assert_allclose(
        np.sum(weighted_noise, axis=(1, 2)),
        0.0,
        atol=5.0e-12,
    )


def test_seasonal_control_moves_only_the_axisymmetric_thermal_equator() -> None:
    equinox = _model()
    summer = _model()
    eq = configure_held_suarez_circulation(
        equinox,
        HeldSuarezSpec(initial_temperature_noise_k=0.0, polar_sponge_width_rows=0.0),
    )
    js = configure_held_suarez_circulation(
        summer,
        HeldSuarezSpec(
            seasonal_heat_equator_deg=20.0,
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
    )
    bottom = summer.level_index(1000)
    eq_peak = equinox.grid.lat_deg[
        np.argmax(eq.equilibrium_temperature_k[bottom, :, 0])
    ]
    summer_peak = summer.grid.lat_deg[
        np.argmax(js.equilibrium_temperature_k[bottom, :, 0])
    ]
    assert abs(eq_peak) <= equinox.config.dlat_deg
    assert abs(summer_peak - 20.0) <= summer.config.dlat_deg
    assert np.max(np.ptp(js.equilibrium_temperature_k, axis=2)) == 0.0


def test_wind_belt_statistics_reports_directional_area_fractions() -> None:
    model = _model()
    configure_held_suarez_circulation(model)
    k = model.level_index(850)
    lat = model.grid.lat2d_deg
    model.u[k] = np.where(
        np.abs(lat) <= 20.0,
        -5.0,
        np.where(np.abs(lat) <= 60.0, 8.0, -2.0),
    )
    stats = wind_belt_statistics(model, 850.0)

    assert np.isclose(stats.tropical_mean_m_s, -5.0)
    assert np.isclose(stats.northern_midlatitude_mean_m_s, 8.0)
    assert np.isclose(stats.southern_midlatitude_mean_m_s, 8.0)
    assert np.isclose(stats.northern_polar_mean_m_s, -2.0)
    assert np.isclose(stats.southern_polar_mean_m_s, -2.0)
    assert stats.tropical_easterly_fraction == 1.0
    assert stats.northern_midlatitude_westerly_fraction == 1.0
    assert stats.southern_midlatitude_westerly_fraction == 1.0
    assert stats.northern_polar_easterly_fraction == 1.0
    assert stats.southern_polar_easterly_fraction == 1.0


def test_orographic_circulation_retains_etopo_and_masks_terrain() -> None:
    model = AtmosphereModel(
        ModelConfig(
            dlon_deg=5.0,
            dlat_deg=5.0,
            lat_limit_deg=77.5,
            pressure_step_hpa=100,
            dt_seconds=300.0,
            seasonal_phase=0.0,
        )
    )
    forcing = configure_orographic_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
        OrographicCirculationSpec(
            initial_temperature_noise_k=0.0,
        ),
    )
    tibet_y = int(np.argmin(np.abs(model.grid.lat_deg - 32.0)))
    tibet_x = int(
        np.argmin(
            np.abs(((model.grid.lon_deg - 87.0 + 180.0) % 360.0) - 180.0)
        )
    )

    assert model.boundary.surface_elevation_m[tibet_y, tibet_x] > 3_000.0
    assert not model.active[model.level_index(900), tibet_y, tibet_x]
    assert model.config.orographic_lift_scale > 0.0
    assert model.config.terrain_blocking_rate_s > 0.0
    assert np.max(model.boundary.terrain_slope) > 0.0
    assert np.max(model.surface_weight[model.lowest_layer]) > 0.8
    assert np.count_nonzero(model.boundary.seasonal_surface_pressure_anomaly_pa) == 0
    assert np.max(np.ptp(forcing.equilibrium_temperature_k[0], axis=1)) > 0.0


def test_orographic_resting_reference_is_hydrostatically_terrain_balanced() -> None:
    model = AtmosphereModel(
        ModelConfig(
            dlon_deg=10.0,
            dlat_deg=10.0,
            pressure_step_hpa=100,
            dt_seconds=300.0,
            seasonal_phase=0.0,
        )
    )
    configure_orographic_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
        OrographicCirculationSpec(
            initial_temperature_noise_k=0.0,
        ),
    )

    assert np.all(model.temperature_k[model.active] == 264.0)
    height_500 = model.compute_geopotential()[model.level_index(500)]
    assert float(np.ptp(height_500)) < 1.0e-8
