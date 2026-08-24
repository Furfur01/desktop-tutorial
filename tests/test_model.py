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
