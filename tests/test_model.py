from __future__ import annotations

import numpy as np

from atmos20 import AtmosphereModel, ModelConfig


def small_model(**updates: float) -> AtmosphereModel:
    config = ModelConfig(dlon_deg=10.0, dlat_deg=10.0, dt_seconds=240.0, **updates)
    return AtmosphereModel(config)


def test_pressure_grid_is_twenty_50_hpa_levels() -> None:
    model = small_model()
    assert model.pressure_hpa.tolist() == list(range(1000, 49, -50))
    assert model.nz == 20
    assert np.all(np.diff(model.pressure_hpa) == -50)


def test_default_grid_uses_real_2p5_degree_etopo_relief() -> None:
    model = AtmosphereModel()
    assert model.grid.shape == (63, 144)

    def cell(lon: float, lat: float) -> tuple[int, int]:
        iy = int(np.argmin(np.abs(model.grid.lat_deg - lat)))
        ix = int(np.argmin(np.abs(((model.grid.lon_deg - lon + 180.0) % 360.0) - 180.0)))
        return iy, ix

    tibet = cell(87.0, 32.0)
    andes = cell(290.0, -20.0)
    pacific = cell(210.0, 0.0)
    assert model.boundary.surface_elevation_m[tibet] > 4_000.0
    assert model.boundary.surface_elevation_m[andes] > 1_500.0
    assert model.boundary.land_mask[tibet]
    assert not model.boundary.land_mask[pacific]
    assert model.boundary.surface_elevation_m[pacific] == 0.0


def test_tibetan_plateau_blocks_lower_pressure_levels() -> None:
    model = small_model()
    iy = int(np.argmin(np.abs(model.grid.lat_deg - 32.0)))
    ix = int(np.argmin(np.abs(((model.grid.lon_deg - 87.0 + 180.0) % 360.0) - 180.0)))
    elevation = model.boundary.surface_elevation_m[iy, ix]
    surface_pressure = model.boundary.base_surface_pressure_pa[iy, ix] / 100.0

    assert elevation > 3500.0
    assert surface_pressure < 700.0
    assert not model.active[model.level_index(850), iy, ix]
    assert model.active[model.level_index(500), iy, ix]


def test_terrain_changes_low_level_momentum_tendency() -> None:
    mountain = small_model(tibet_height_scale=1.0)
    flat = small_model(tibet_height_scale=0.0)

    mountain.u[mountain.active] = 12.0
    flat.u[flat.active] = 12.0

    rhs_m = mountain._rhs(
        mountain.u,
        mountain.v,
        mountain.temperature_k,
        mountain.surface_pressure_anomaly_pa,
    )
    rhs_f = flat._rhs(flat.u, flat.v, flat.temperature_k, flat.surface_pressure_anomaly_pa)

    # Compare a broad Himalayan/plateau-flank box rather than one fragile cell.
    region = (
        (mountain.grid.lat2d_deg >= 20)
        & (mountain.grid.lat2d_deg <= 45)
        & (mountain.grid.lon2d_deg >= 65)
        & (mountain.grid.lon2d_deg <= 115)
    )
    k = mountain.level_index(700)
    difference = np.abs(rhs_m.u[k] - rhs_f.u[k]) + np.abs(rhs_m.v[k] - rhs_f.v[k])
    assert float(np.nanmax(difference[region])) > 1.0e-5


def test_short_integration_stays_finite() -> None:
    model = small_model()
    model.step(24)
    assert np.isfinite(model.u).all()
    assert np.isfinite(model.v).all()
    assert np.isfinite(model.temperature_k).all()
    assert np.isfinite(model.surface_pressure_anomaly_pa).all()
    assert np.max(np.abs(model.u)) < 160.1
    assert np.max(np.abs(model.v)) < 160.1


def _wind_at(model: AtmosphereModel, level: int, lon: float, lat: float) -> tuple[float, float]:
    k = model.level_index(level)
    iy = int(np.argmin(np.abs(model.grid.lat_deg - lat)))
    ix = int(np.argmin(np.abs(((model.grid.lon_deg - lon + 180.0) % 360.0) - 180.0)))
    return float(model.u[k, iy, ix]), float(model.v[k, iy, ix])


def test_default_warm_start_produces_boreal_summer_monsoon() -> None:
    model = AtmosphereModel(ModelConfig(dlon_deg=5.0, dlat_deg=5.0, dt_seconds=600.0))
    model.advance_hours(3.0)

    arabian = _wind_at(model, 850, 65.0, 15.0)
    bay = _wind_at(model, 850, 88.0, 15.0)
    south_china_sea = _wind_at(model, 850, 112.0, 15.0)
    east_china_sea = _wind_at(model, 850, 125.0, 30.0)

    assert arabian[0] > 0.5 and arabian[1] > 0.5
    assert bay[0] > 0.2 and bay[1] > 0.5
    assert south_china_sea[1] > 0.5
    assert east_china_sea[1] > 0.5

    model.advance_hours(9.0)
    assert _wind_at(model, 850, 88.0, 15.0)[1] > 0.5
    assert _wind_at(model, 850, 112.0, 15.0)[1] > 0.5
    assert _wind_at(model, 850, 125.0, 30.0)[1] > 0.5


def test_winter_setting_reverses_low_level_asian_flow() -> None:
    winter = AtmosphereModel(
        ModelConfig(
            dlon_deg=5.0,
            dlat_deg=5.0,
            dt_seconds=600.0,
            seasonal_phase=-1.0,
        )
    )
    winter.advance_hours(3.0)

    assert _wind_at(winter, 850, 88.0, 15.0)[1] < -0.5
    assert _wind_at(winter, 850, 112.0, 15.0)[1] < -0.5
    # SSPRK3 removes the slight RK2 fast-mode amplification, so the northern
    # edge of the reversal develops a little more gradually during hour 3.
    assert _wind_at(winter, 850, 125.0, 30.0)[1] < -0.35

    winter.advance_hours(9.0)
    assert _wind_at(winter, 850, 88.0, 15.0)[1] < -0.5
    assert _wind_at(winter, 850, 112.0, 15.0)[1] < -0.5
    assert _wind_at(winter, 850, 125.0, 30.0)[1] < -0.5


def test_high_plateau_surface_temperature_uses_elevation_correction() -> None:
    model = AtmosphereModel()
    iy = int(np.argmin(np.abs(model.grid.lat_deg - 32.0)))
    ix = int(np.argmin(np.abs(((model.grid.lon_deg - 87.0 + 180.0) % 360.0) - 180.0)))

    assert model.boundary.surface_elevation_m[iy, ix] > 4_000.0
    assert model.boundary.surface_temperature_k[iy, ix] < 293.15
    weighted_mean = np.sum(
        model.boundary.seasonal_surface_pressure_anomaly_pa * model.grid.area_weight
    ) / np.sum(model.grid.area_weight)
    assert abs(float(weighted_mean)) < 1.0e-8
