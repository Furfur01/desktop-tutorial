from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from .config import ModelConfig
from .grid import LatLonGrid


# Deliberately lightweight coastlines/land polygons. They are coarse enough for
# the dynamics grid and also double as outlines in the frontend, avoiding a
# heavy GIS dependency.
CONTINENT_POLYGONS: dict[str, list[tuple[float, float]]] = {
    "eurasia": [
        (-12, 35), (-6, 48), (2, 58), (18, 68), (45, 72), (80, 76),
        (120, 72), (150, 65), (178, 54), (170, 40), (145, 27), (122, 18),
        (110, 8), (97, 5), (82, 10), (72, 20), (58, 24), (45, 32),
        (33, 38), (20, 35), (7, 36), (-12, 35),
    ],
    "arabia_india_seasia": [
        (34, 34), (51, 31), (58, 25), (68, 24), (76, 8), (84, 7),
        (91, 22), (105, 22), (116, 12), (125, 4), (112, -8), (99, -5),
        (92, 5), (82, 7), (73, 18), (58, 17), (47, 12), (39, 17), (34, 34),
    ],
    "africa": [
        (-17, 36), (9, 37), (31, 31), (42, 12), (51, 3), (43, -15),
        (31, -34), (18, -35), (8, -23), (-6, -5), (-17, 14), (-17, 36),
    ],
    "north_america": [
        (-168, 69), (-145, 72), (-120, 73), (-95, 72), (-66, 58),
        (-55, 47), (-72, 27), (-82, 20), (-97, 15), (-112, 24),
        (-126, 39), (-151, 56), (-168, 69),
    ],
    "central_america": [
        (-111, 25), (-98, 25), (-88, 18), (-78, 9), (-83, 6),
        (-92, 14), (-105, 19), (-111, 25),
    ],
    "south_america": [
        (-81, 12), (-67, 13), (-50, 5), (-35, -7), (-43, -23),
        (-54, -38), (-68, -55), (-76, -42), (-80, -18), (-81, 12),
    ],
    "australia": [
        (112, -11), (132, -10), (153, -24), (146, -39), (126, -36),
        (113, -25), (112, -11),
    ],
    "greenland": [
        (-73, 59), (-48, 59), (-20, 72), (-34, 83), (-58, 82),
        (-73, 70), (-73, 59),
    ],
    "madagascar": [(43, -12), (50, -13), (51, -25), (46, -27), (43, -12)],
    "japan": [(129, 31), (143, 31), (146, 44), (137, 46), (129, 31)],
}


@dataclass(slots=True)
class BoundaryFields:
    land_mask: np.ndarray
    surface_temperature_k: np.ndarray
    surface_elevation_m: np.ndarray
    base_surface_pressure_pa: np.ndarray
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


def build_land_mask(grid: LatLonGrid) -> np.ndarray:
    points = np.column_stack(
        (
            ((grid.lon2d_deg + 180.0) % 360.0 - 180.0).ravel(),
            grid.lat2d_deg.ravel(),
        )
    )
    land = np.zeros(points.shape[0], dtype=bool)
    for polygon in CONTINENT_POLYGONS.values():
        land |= Path(np.asarray(polygon, dtype=float)).contains_points(points)
    return land.reshape(grid.shape)


def build_boundary(config: ModelConfig, grid: LatLonGrid) -> BoundaryFields:
    land = build_land_mask(grid)
    lat = grid.lat2d_deg

    # Prescribed ocean temperature. The warm belt is shifted north for boreal
    # summer; current anomalies are fixed, as requested.
    sst_c = np.clip(29.5 - 0.37 * np.abs(lat - 8.0), -1.5, 30.5)
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
    land_c = np.clip(27.0 - 0.43 * np.abs(lat - 24.0), -24.0, 34.0)
    heat = config.land_heating_scale
    land_c += heat * gaussian(grid, 25, 23, 28, 11, +10.0)  # Sahara
    land_c += heat * gaussian(grid, 52, 27, 22, 10, +7.0)   # Arabia/Iran
    land_c += heat * gaussian(grid, 79, 27, 25, 11, +7.5)   # India/Pakistan
    land_c += heat * gaussian(grid, 96, 42, 34, 15, +8.0)   # central/east Asia
    land_c += heat * gaussian(grid, 250, 34, 20, 11, +6.0)  # SW North America
    land_c += heat * gaussian(grid, 292, -12, 22, 12, +3.0) # Brazil interior
    land_c += heat * gaussian(grid, 134, -25, 20, 11, +4.0) # Australian interior

    surface_c = np.where(land, land_c, sst_c)
    surface_c = np.clip(surface_c, -28.0, 46.0)

    # Idealized topography. Tibet is intentionally broad and high enough to
    # remove the lower eight-to-nine 50 hPa layers from the local atmosphere.
    z = np.zeros(grid.shape, dtype=float)
    tibet = config.tibet_height_scale
    z += tibet * gaussian(grid, 87, 32, 16, 7.0, 3600.0)
    z += tibet * gaussian(grid, 92, 34, 11, 6.0, 1800.0)
    z += tibet * gaussian(grid, 80, 29, 20, 2.7, 1050.0)  # Himalaya ridge
    z += gaussian(grid, 54, 32, 12, 7, 1450.0)            # Iranian Plateau
    z += gaussian(grid, 101, 46, 12, 7, 1500.0)           # Altai/Mongolia
    z += gaussian(grid, 250, 42, 10, 18, 2450.0)          # Rockies
    z += gaussian(grid, 287, -18, 6.5, 24, 3900.0)        # Andes
    z += gaussian(grid, 37, 8, 7, 10, 1900.0)             # Ethiopian Highlands
    z += gaussian(grid, 10, 46, 8, 4, 950.0)              # Alps
    z += gaussian(grid, 145, -5, 6, 7, 1300.0)            # New Guinea
    z *= land
    z = np.clip(z, 0.0, 5800.0)

    t_k = surface_c + 273.15
    p0 = 101_325.0
    ps = p0 * np.exp(
        -config.gravity_m_s2 * z
        / (config.gas_constant_dry_air * np.clip(t_k, 235.0, 315.0))
    )

    dzdx = grid.grad_x(z)
    dzdy = grid.grad_y(z)
    slope = np.sqrt(dzdx * dzdx + dzdy * dzdy)

    return BoundaryFields(
        land_mask=land,
        surface_temperature_k=t_k,
        surface_elevation_m=z,
        base_surface_pressure_pa=ps,
        terrain_slope_x=dzdx,
        terrain_slope_y=dzdy,
        terrain_slope=slope,
    )
