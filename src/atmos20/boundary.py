from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from .config import ModelConfig
from .grid import LatLonGrid


@dataclass(slots=True)
class BoundaryFields:
    land_mask: np.ndarray
    land_fraction: np.ndarray
    surface_temperature_k: np.ndarray
    surface_elevation_m: np.ndarray
    base_surface_pressure_pa: np.ndarray
    seasonal_surface_pressure_anomaly_pa: np.ndarray
    terrain_slope_x: np.ndarray
    terrain_slope_y: np.ndarray
    terrain_slope: np.ndarray


def _wrap_lon_delta(lon: np.ndarray, lon0: float) -> np.ndarray:
    return (lon - lon0 + 180.0) % 360.0 - 180.0


def gaussian(
    grid: LatLonGrid,
    lon0: float,
    lat0: float,
    sigma_lon: float,
    sigma_lat: float,
    amplitude: float,
) -> np.ndarray:
    dx = _wrap_lon_delta(grid.lon2d_deg, lon0) / sigma_lon
    dy = (grid.lat2d_deg - lat0) / sigma_lat
    return amplitude * np.exp(-0.5 * (dx * dx + dy * dy))


def _interpolate_real_topography(grid: LatLonGrid) -> tuple[np.ndarray, np.ndarray]:
    """Load packaged ETOPO 2022 relief and interpolate it to ``grid``.

    The package asset is a 1-degree area aggregate of 0.25-degree samples from
    the NOAA NCEI ETOPO 2022 ice-surface grid. Keeping this preprocessing out
    of the runtime makes the interactive model independent of GIS libraries.
    """

    exact_2p5 = np.isclose(grid.config.dlon_deg, 2.5) and np.isclose(
        grid.config.dlat_deg, 2.5
    )
    asset_name = "etopo_2022_2p5deg.npz" if exact_2p5 else "etopo_2022_1deg.npz"
    asset = files("atmos20").joinpath(f"data/{asset_name}")
    with asset.open("rb") as stream, np.load(stream) as data:
        source_elevation = data["elevation_m"].astype(float)
        source_land_fraction = data["land_fraction"].astype(float)
        source_lat = data["lat_deg"].astype(float)
        source_lon = data["lon_deg"].astype(float)

    if np.array_equal(grid.lat_deg, source_lat) and np.array_equal(grid.lon_deg, source_lon):
        return source_land_fraction, source_elevation

    source_spacing = float(source_lon[1] - source_lon[0])
    lon_position = np.mod(grid.lon_deg, 360.0) / source_spacing
    x0 = np.floor(lon_position).astype(int) % source_elevation.shape[1]
    x1 = (x0 + 1) % source_elevation.shape[1]
    tx = lon_position - np.floor(lon_position)

    lat_position = (
        np.clip(grid.lat_deg, source_lat[0], source_lat[-1]) - source_lat[0]
    ) / float(source_lat[1] - source_lat[0])
    y0 = np.floor(lat_position).astype(int)
    y1 = np.minimum(y0 + 1, source_elevation.shape[0] - 1)
    ty = lat_position - y0

    def interpolate(field: np.ndarray) -> np.ndarray:
        south = (1.0 - tx)[None, :] * field[y0[:, None], x0[None, :]]
        south += tx[None, :] * field[y0[:, None], x1[None, :]]
        north = (1.0 - tx)[None, :] * field[y1[:, None], x0[None, :]]
        north += tx[None, :] * field[y1[:, None], x1[None, :]]
        return (1.0 - ty)[:, None] * south + ty[:, None] * north

    return interpolate(source_land_fraction), interpolate(source_elevation)


def build_boundary(config: ModelConfig, grid: LatLonGrid) -> BoundaryFields:
    land_fraction, z = _interpolate_real_topography(grid)
    land = land_fraction >= 0.5
    lat = grid.lat2d_deg
    season = float(np.clip(config.seasonal_phase, -1.0, 1.0))

    # Preserve the Tibet experiment control on the observed ETOPO relief.
    # Apply it before the elevation-temperature correction so both boundary
    # fields describe the same scaled terrain.
    tibet_weight = gaussian(grid, 87, 32, 18, 9, 1.0)
    z *= 1.0 + (config.tibet_height_scale - 1.0) * tibet_weight
    z *= land
    z = np.clip(z, 0.0, 7000.0)

    # Prescribed ocean temperature. The warm belt is shifted north for boreal
    # summer; current anomalies are fixed, as requested.
    sst_c = np.clip(29.5 - 0.37 * np.abs(lat - 8.0 * season), -1.5, 30.5)
    current = config.ocean_current_scale
    sst_c += current * gaussian(grid, 143, 34, 18, 7, +3.3)   # Kuroshio
    sst_c += current * gaussian(grid, 300, 37, 18, 7, +3.1)   # Gulf Stream
    sst_c += current * gaussian(grid, 235, 29, 18, 8, -3.8)   # California Current
    sst_c += current * gaussian(grid, 346, 28, 16, 8, -3.2)   # Canary Current
    sst_c += current * gaussian(grid, 284, -20, 15, 11, -4.2) # Humboldt Current
    sst_c += current * gaussian(grid, 12, -23, 15, 10, -3.5)  # Benguela Current
    sst_c += current * gaussian(grid, 215, 0, 48, 7, -1.8)    # Pacific cold tongue
    sst_c += current * gaussian(grid, 75, 8, 28, 8, +1.1)     # Arabian/Indian warm pool

    # Land temperature uses the same seasonal latitude curve, then adds
    # continentality and named hot regions. It is a prescribed lower boundary,
    # so the atmosphere cannot cool it in this version.
    land_c = np.clip(27.0 - 0.43 * np.abs(lat - 24.0 * season), -24.0, 34.0)
    heat = config.land_heating_scale
    north_heat = 0.5 * (1.0 + season)
    south_heat = 0.5 * (1.0 - season)
    land_c += heat * north_heat * gaussian(grid, 25, 23, 28, 11, +10.0)  # Sahara
    land_c += heat * north_heat * gaussian(grid, 52, 27, 22, 10, +7.0)   # Arabia/Iran
    land_c += heat * north_heat * gaussian(grid, 79, 27, 25, 11, +7.5)   # India/Pakistan
    land_c += heat * north_heat * gaussian(grid, 96, 42, 34, 15, +8.0)   # central/east Asia
    land_c += heat * north_heat * gaussian(grid, 250, 34, 20, 11, +6.0)  # SW North America
    land_c += heat * south_heat * gaussian(grid, 292, -12, 22, 12, +6.0) # Brazil interior
    land_c += heat * south_heat * gaussian(grid, 134, -25, 20, 11, +8.0) # Australian interior

    # Surface air temperature follows terrain height. Without this correction
    # the ETOPO Tibetan Plateau was assigned nearly 40 °C and produced an
    # artificial low-level outflow that looked like an East Asian winter
    # monsoon despite the nominal summer boundary condition.
    land_c -= config.surface_lapse_rate_k_m * z

    surface_c = np.where(land, land_c, sst_c)
    surface_c = np.clip(surface_c, -28.0, 46.0)

    t_k = surface_c + 273.15
    p0 = 101_325.0
    ps = p0 * np.exp(
        -config.gravity_m_s2 * z
        / (config.gas_constant_dry_air * np.clip(t_k, 235.0, 315.0))
    )

    # A warm continental column must be paired with a thermal surface low.
    # Starting from zero pressure anomaly leaves only the raised warm-column
    # thickness gradient, which initially accelerates air out of Asia. This
    # balanced warm start creates the expected summer inflow within hours
    # instead of requiring several simulated days of spin-up.
    zonal_surface_c = np.mean(surface_c, axis=1, keepdims=True)
    pressure_target = (
        -config.thermal_low_pressure_pa_per_k
        * (surface_c - zonal_surface_c)
        * land_fraction
    )
    pressure_target += season * gaussian(grid, 80, 25, 30, 15, -700.0)
    pressure_target += season * gaussian(grid, 110, 35, 35, 18, -500.0)
    pressure_target += season * gaussian(grid, 150, 30, 35, 16, +550.0)
    pressure_target += season * gaussian(grid, 65, -30, 35, 16, +500.0)
    pressure_target += season * gaussian(grid, 105, 16, 42, 9, -300.0)
    pressure_mean = np.sum(pressure_target * grid.area_weight) / np.sum(grid.area_weight)
    pressure_target = np.clip(pressure_target - pressure_mean, -1_600.0, 1_600.0)
    pressure_mean = np.sum(pressure_target * grid.area_weight) / np.sum(grid.area_weight)
    pressure_target -= pressure_mean

    dzdx = grid.grad_x(z)
    dzdy = grid.grad_y(z)
    slope = np.sqrt(dzdx * dzdx + dzdy * dzdy)

    return BoundaryFields(
        land_mask=land,
        land_fraction=land_fraction,
        surface_temperature_k=t_k,
        surface_elevation_m=z,
        base_surface_pressure_pa=ps,
        seasonal_surface_pressure_anomaly_pa=pressure_target,
        terrain_slope_x=dzdx,
        terrain_slope_y=dzdy,
        terrain_slope=slope,
    )
