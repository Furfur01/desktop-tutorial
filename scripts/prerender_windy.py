"""Compute Atmos20 on the backend and render browser-safe media assets.

The browser receives only WebP files and a JSON manifest. Atmospheric state,
scalar grids, and particle positions never leave this module.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from scipy.ndimage import gaussian_filter, maximum_filter

from atmos20 import (
    AtmosphereModel,
    ModelConfig,
    OrographicCirculationSpec,
    ThreeCellAtmosphereModel,
    ThreeCellSpec,
)
from atmos20.baroclinic import DryBaroclinicWaveSpec, configure_baroclinic_wave_model
from atmos20.circulation import (
    HeldSuarezSpec,
    configure_orographic_held_suarez_circulation,
)
from atmos20.fronts import FrontDiagnostics, FrontType, diagnose_fronts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "web" / "assets" / "prerender"
COUNTRIES_PATH = ROOT / "data" / "natural_earth" / "ne_110m_admin_0_countries.geojson"
ProgressCallback = Callable[[float, str, str], None]


@dataclass(frozen=True, slots=True)
class Viewport:
    """A rectangular equirectangular viewport in display longitudes."""

    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    width: int = 1280
    height: int = 640

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def is_global(self) -> bool:
        return self.lon_span >= 359.999

    def normalize_lon(self, lon: float) -> float:
        return (lon - self.lon_min) % 360.0 + self.lon_min


WORLD_VIEWPORT = Viewport("world", 0.0, 360.0, -90.0, 90.0)
VIEWPORTS: dict[str, Viewport] = {"world": WORLD_VIEWPORT}


@dataclass(frozen=True, slots=True)
class RenderSettings:
    scenario: str = "circulation"
    resolution: float = 5.0
    level: int = 900
    region: str = "world"
    season: str = "equinox"
    spinup_hours: float = 120.0
    analysis_hours: float = 120.0
    frames: int = 48
    fps: int = 12
    particles: int = 2400
    flow_speed: float = 1.0
    trail: float = 0.94
    tibet_scale: float = 1.0
    land_heating_scale: float = 1.0
    ocean_current_scale: float = 1.0
    jet_strength: float = 35.0
    perturbation_amplitude: float = 1.0
    hemisphere: str = "north"
    equator_to_pole_contrast_k: float = 60.0
    surface_drag_days: float = 1.0
    seasonal_heat_equator_deg: float | None = None

    @property
    def timestep_seconds(self) -> float:
        if self.scenario == "baroclinic":
            if self.resolution <= 1.0:
                return 45.0
            if self.resolution <= 2.5:
                return 120.0
            return 240.0
        if self.resolution <= 1.0:
            return 45.0
        if self.resolution <= 2.5:
            return 120.0
        return 300.0

    @property
    def seasonal_phase(self) -> float:
        return {"winter": -1.0, "equinox": 0.0, "summer": 1.0}[self.season]

    @property
    def effective_seasonal_heat_equator_deg(self) -> float:
        """Return the explicit thermal-equator latitude or an Earth-like season."""

        if self.seasonal_heat_equator_deg is not None:
            return float(self.seasonal_heat_equator_deg)
        return 23.44 * self.seasonal_phase


PALETTES = {
    "wind": ["#3d50a3", "#3374b5", "#269ba9", "#36b875", "#8dcc55", "#e2d653", "#e5974e", "#b64173"],
    "zonalWind": ["#432d88", "#355db2", "#2f9bc1", "#d8dfd4", "#e6c357", "#df8248", "#a93656"],
    "temperature": ["#4545a5", "#347cc2", "#42b7b1", "#c3d779", "#f4c85f", "#e87545", "#b93455"],
    "pressure": ["#513392", "#3d70ba", "#43a9a4", "#d4d18e", "#e39b4e", "#b84961"],
    "omega": ["#67339b", "#397dcc", "#47c4c2", "#d6d6b1", "#e99952", "#bc3f62"],
    "geopotential": ["#343d82", "#3769a6", "#399a9d", "#75b567", "#c7c65b", "#d48b54"],
    "terrain": ["#284f85", "#367d75", "#66a762", "#aabb69", "#b9915e", "#8a684f", "#e1ddd0"],
    "fronts": ["#38439b", "#3478bc", "#47b5b0", "#d4d684", "#efb34e", "#cf5262"],
}

LAYER_INFO = {
    "wind": ("风速", "WIND", "m/s"),
    "temperature": ("温度", "TEMPERATURE", "°C"),
    "pressure": ("气压", "SURFACE PRESSURE ANOMALY", "hPa"),
    "omega": ("垂直运动", "OMEGA", "Pa/s"),
    "geopotential": ("位势高度", "GEOPOTENTIAL", "m"),
    "terrain": ("真实地形", "RELIEF", "m"),
}

CIRCULATION_LAYER_INFO = {
    "wind": LAYER_INFO["wind"],
    "zonalWind": ("纬向风", "ZONAL WIND (EASTERLY / WESTERLY)", "m/s"),
    "temperature": LAYER_INFO["temperature"],
    "pressure": LAYER_INFO["pressure"],
    "omega": LAYER_INFO["omega"],
    "geopotential": LAYER_INFO["geopotential"],
    "terrain": LAYER_INFO["terrain"],
}

FRONT_LAYER_INFO = {
    "fronts": ("锋面", "OBJECTIVE FRONTS", "K"),
}


@dataclass(frozen=True, slots=True)
class DisplaySurface:
    """High-resolution rendering copy of the model's ETOPO lower boundary."""

    land_fraction: np.ndarray
    elevation_m: np.ndarray
    lon_deg: np.ndarray | None = None
    lat_deg: np.ndarray | None = None


def load_display_surface(tibet_scale: float = 1.0) -> DisplaySurface:
    """Load the packaged 1-degree relief without downsampling to the model grid."""

    source = ROOT / "src" / "atmos20" / "data" / "etopo_2022_1deg.npz"
    with np.load(source) as data:
        lon = data["lon_deg"].astype(float)
        lat = data["lat_deg"].astype(float)
        elevation = data["elevation_m"].astype(float)
        land_fraction = data["land_fraction"].astype(float)
    lon2d, lat2d = np.meshgrid(lon, lat)
    lon_delta = (lon2d - 87.0 + 180.0) % 360.0 - 180.0
    tibet_weight = np.exp(
        -0.5 * ((lon_delta / 18.0) ** 2 + ((lat2d - 32.0) / 9.0) ** 2)
    )
    elevation *= 1.0 + (float(tibet_scale) - 1.0) * tibet_weight
    elevation = np.clip(elevation, 0.0, 7000.0)
    return DisplaySurface(
        land_fraction=land_fraction,
        elevation_m=elevation,
        lon_deg=lon,
        lat_deg=lat,
    )


@dataclass(frozen=True, slots=True)
class RenderSnapshot:
    """Two-dimensional model output retained for one playback time."""

    time_hours: float
    u_m_s: np.ndarray
    v_m_s: np.ndarray
    temperature_k: np.ndarray
    potential_temperature_k: np.ndarray
    pressure_anomaly_hpa: np.ndarray
    omega_pa_s: np.ndarray
    geopotential_height_m: np.ndarray
    active: np.ndarray
    fronts: FrontDiagnostics

    def field(self, layer: str) -> tuple[np.ndarray, np.ndarray]:
        if layer == "wind":
            return np.hypot(self.u_m_s, self.v_m_s), self.active
        if layer == "zonalWind":
            return self.u_m_s, self.active
        if layer == "temperature":
            return self.temperature_k - 273.15, self.active
        if layer == "pressure":
            return self.pressure_anomaly_hpa, np.ones_like(self.active)
        if layer == "omega":
            return self.omega_pa_s, self.active
        if layer == "geopotential":
            return self.geopotential_height_m, self.active
        if layer == "fronts":
            return self.potential_temperature_k, self.active
        raise KeyError(layer)


def _noop_progress(_progress: float, _stage: str, _message: str) -> None:
    return


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray([int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=float)


def colourize(
    values: np.ndarray,
    lower: float,
    upper: float,
    palette: list[str],
    stops: list[float] | None = None,
) -> np.ndarray:
    if stops is None:
        normalized = np.clip((values - lower) / max(upper - lower, 1.0e-9), 0.0, 0.99999)
    else:
        normalized = np.interp(values, stops, np.linspace(0.0, 0.99999, len(stops)))
    position = normalized * (len(palette) - 1)
    index = np.floor(position).astype(int)
    fraction = (position - index)[..., None]
    colours = np.stack([hex_rgb(item) for item in palette])
    rgb = colours[index] * (1.0 - fraction) + colours[np.minimum(index + 1, len(palette) - 1)] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def grid_sample(field: np.ndarray, lon: np.ndarray, lat: np.ndarray, model: AtmosphereModel) -> np.ndarray:
    nx = model.nx
    ny = model.ny
    x = np.mod(lon, 360.0) / model.config.dlon_deg
    # The dynamical grid deliberately stops short of the singular poles. Clamp
    # the 2:1 display texture to the nearest physical model row so the polar
    # caps are stable instead of accidentally interpolating the first two rows.
    y = np.clip(
        (lat - model.grid.lat_deg[0]) / model.config.dlat_deg,
        0.0,
        ny - 1.0,
    )
    x0 = np.floor(x).astype(int) % nx
    x1 = (x0 + 1) % nx
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, ny - 1)
    tx = x - np.floor(x)
    ty = y - np.floor(y)
    south = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    north = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return south * (1.0 - ty) + north * ty


def regular_grid_sample(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    source_lon: np.ndarray,
    source_lat: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a periodic regular longitude/latitude grid."""

    lon_spacing = float(source_lon[1] - source_lon[0])
    lat_spacing = float(source_lat[1] - source_lat[0])
    x = np.mod(lon - float(source_lon[0]), 360.0) / lon_spacing
    y = np.clip(
        (lat - float(source_lat[0])) / lat_spacing,
        0.0,
        len(source_lat) - 1.0,
    )
    x0 = np.floor(x).astype(int) % len(source_lon)
    x1 = (x0 + 1) % len(source_lon)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, len(source_lat) - 1)
    tx = x - np.floor(x)
    ty = y - np.floor(y)
    south = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    north = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return south * (1.0 - ty) + north * ty


def sample_display_surface(
    surface: DisplaySurface,
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    model: AtmosphereModel,
) -> np.ndarray:
    if surface.lon_deg is None or surface.lat_deg is None:
        return grid_sample(field, lon, lat, model)
    return regular_grid_sample(
        field,
        lon,
        lat,
        surface.lon_deg,
        surface.lat_deg,
    )


def pixel_coordinates(viewport: Viewport) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(viewport.width, dtype=float) + 0.5) / viewport.width
    y = (np.arange(viewport.height, dtype=float) + 0.5) / viewport.height
    lon = viewport.lon_min + x * viewport.lon_span
    lat = viewport.lat_max - y * viewport.lat_span
    return np.meshgrid(lon, lat)


def project(lon: float, lat: float, viewport: Viewport) -> tuple[float, float]:
    display_lon = viewport.normalize_lon(lon)
    return (
        (display_lon - viewport.lon_min) / viewport.lon_span * viewport.width,
        (viewport.lat_max - lat) / viewport.lat_span * viewport.height,
    )


def iter_rings(geometry: dict[str, object]):
    coordinates = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        yield from coordinates
    elif geometry["type"] == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def draw_boundaries(image: Image.Image, viewport: Viewport) -> None:
    if not COUNTRIES_PATH.exists():
        return
    geojson = json.loads(COUNTRIES_PATH.read_text(encoding="utf-8"))
    draw = ImageDraw.Draw(image, "RGBA")
    for feature in geojson["features"]:
        for ring in iter_rings(feature["geometry"]):
            segment: list[tuple[float, float]] = []
            previous_x: float | None = None
            for lon, lat, *_ in ring:
                x, y = project(float(lon), float(lat), viewport)
                if previous_x is not None and abs(x - previous_x) > viewport.width * 0.45:
                    if len(segment) > 1:
                        draw.line(segment, fill=(18, 37, 58, 190), width=1, joint="curve")
                    segment = []
                segment.append((x, y))
                previous_x = x
            if len(segment) > 1:
                draw.line(segment, fill=(18, 37, 58, 190), width=1, joint="curve")


def terrain_filled_wind(model: AtmosphereModel, level_index: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.where(model.active[level_index], model.u[level_index], model._gather_lowest(model.u))
    v = np.where(model.active[level_index], model.v[level_index], model._gather_lowest(model.v))
    mode = ("nearest", "wrap")
    return gaussian_filter(u, sigma=0.9, mode=mode), gaussian_filter(v, sigma=0.9, mode=mode)


def layer_field(model: AtmosphereModel, layer: str, level: int) -> tuple[np.ndarray, np.ndarray]:
    level_index = model.level_index(level)
    if layer == "zonalWind":
        u, _ = terrain_filled_wind(model, level_index)
        return u, model.active[level_index]
    if layer == "temperature":
        return model.temperature_k[level_index] - 273.15, model.active[level_index]
    if layer == "pressure":
        return model.sea_level_pressure_anomaly_hpa(), np.ones(model.grid.shape, dtype=bool)
    if layer == "omega":
        return model.last_omega_pa_s[level_index], model.active[level_index]
    if layer == "geopotential":
        return model.geopotential_height_m()[level_index], model.active[level_index]
    if layer == "terrain":
        return model.boundary.surface_elevation_m, np.ones(model.grid.shape, dtype=bool)
    u, v = terrain_filled_wind(model, level_index)
    return np.hypot(u, v), np.ones(model.grid.shape, dtype=bool)


def layer_limits(field: np.ndarray, active: np.ndarray, layer: str) -> tuple[float, float]:
    finite = field[active & np.isfinite(field)]
    if finite.size == 0:
        return 0.0, 1.0
    if layer == "wind":
        return 0.0, 35.0
    if layer == "zonalWind":
        extent = min(
            70.0,
            max(10.0, float(np.percentile(np.abs(finite), 99))),
        )
        return -extent, extent
    if layer == "pressure":
        return -12.0, 12.0
    if layer == "omega":
        return -0.5, 0.5
    if layer == "terrain":
        return 0.0, max(4_800.0, float(np.percentile(finite, 99)))
    if layer == "fronts":
        lower, upper = np.percentile(finite, [1, 99])
        return float(lower), float(upper)
    lower, upper = np.percentile(finite, [1, 99])
    return min(0.0, float(lower)), max(float(upper), 1.0)


def render_field_background(
    model: AtmosphereModel,
    field: np.ndarray,
    active: np.ndarray,
    layer: str,
    viewport: Viewport,
    *,
    limits: tuple[float, float] | None = None,
    display_surface: DisplaySurface | None = None,
    smooth_polar_caps: bool = False,
) -> tuple[Image.Image, float, float]:
    lon, lat = pixel_coordinates(viewport)
    lower, upper = limits or layer_limits(field, active, layer)
    sampled = grid_sample(field, lon, lat, model)
    sampled_active = grid_sample(active.astype(float), lon, lat, model) > 0.5
    if smooth_polar_caps:
        # Dynamics stop short of the longitude singularity.  Blend the last
        # resolved rows into their zonal mean across the outer buffer instead
        # of stretching any residual boundary-grid structure to the poles.
        configured_start = model.meridional_sponge_start_latitude_deg
        start = (
            float(configured_start)
            if configured_start is not None
            else float(model.grid.lat_deg[-1] - model.config.dlat_deg)
        )
        limit = float(model.grid.lat_deg[-1])
        fraction = np.clip(
            (np.abs(lat) - start) / max(limit - start, 1.0e-6),
            0.0,
            1.0,
        )
        polar_blend = np.sin(0.5 * np.pi * fraction) ** 2
        north_values = field[-1][active[-1] & np.isfinite(field[-1])]
        south_values = field[0][active[0] & np.isfinite(field[0])]
        north_cap = float(np.mean(north_values)) if north_values.size else 0.0
        south_cap = float(np.mean(south_values)) if south_values.size else 0.0
        cap = np.where(lat >= 0.0, north_cap, south_cap)
        sampled = sampled * (1.0 - polar_blend) + cap * polar_blend
    wind_stops = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
    weather_rgb = colourize(
        sampled,
        lower,
        upper,
        PALETTES[layer],
        stops=wind_stops if layer == "wind" else None,
    ).astype(float)

    surface = display_surface or DisplaySurface(
        land_fraction=model.boundary.land_fraction,
        elevation_m=model.boundary.surface_elevation_m,
    )
    elevation = sample_display_surface(
        surface,
        surface.elevation_m,
        lon,
        lat,
        model,
    )
    land = sample_display_surface(
        surface,
        surface.land_fraction,
        lon,
        lat,
        model,
    ) > 0.42
    smoothed_elevation = gaussian_filter(
        elevation,
        sigma=1.1,
        mode=("nearest", "wrap"),
    )
    gradient_y, gradient_x = np.gradient(smoothed_elevation)
    gradient_scale = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y + 180.0**2
    )
    illumination = np.clip(
        0.82 + 0.28 * (-0.72 * gradient_x + 0.50 * gradient_y) / gradient_scale,
        0.56,
        1.08,
    )
    terrain_rgb = colourize(
        elevation,
        0.0,
        5600.0,
        PALETTES["terrain"],
        stops=[0.0, 120.0, 500.0, 1200.0, 2400.0, 3800.0, 5600.0],
    ).astype(float)
    terrain_rgb *= illumination[..., None]
    base = np.empty_like(weather_rgb)
    base[~land] = np.asarray([43.0, 82.0, 111.0])
    base[land] = terrain_rgb[land]
    if layer == "terrain":
        rgb = base
    else:
        blend = 0.74 if layer == "fronts" else 0.70
        rgb = blend * weather_rgb + (1.0 - blend) * base
        # Preserve readable mountain structure even beneath saturated weather
        # colours. This is true shaded relief, not a model-generated field.
        relief = np.clip(elevation / 4200.0, 0.0, 1.0) * land
        rgb *= (1.0 - 0.10 * relief[..., None])
    if layer not in {"terrain", "pressure"}:
        ground = ~sampled_active
        rgb[ground] = 0.94 * base[ground] + 0.06 * weather_rgb[ground]

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    draw_boundaries(image, viewport)
    return image, lower, upper


def render_background(
    model: AtmosphereModel,
    layer: str,
    level: int,
    viewport: Viewport,
) -> tuple[Image.Image, float, float]:
    field, active = layer_field(model, layer, level)
    return render_field_background(model, field, active, layer, viewport)


def velocity(model: AtmosphereModel, level: int, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    level_index = model.level_index(level)
    u, v = terrain_filled_wind(model, level_index)
    return grid_sample(u, lon, lat, model), grid_sample(v, lon, lat, model)


def _front_gradient_threshold(resolution: float) -> float:
    # The 5-degree analytic basic state reaches about 1.76 K / 100 km without
    # containing a synoptic front.  Requiring a stronger gradient prevents the
    # background baroclinic zone from being mislabeled before wave growth.
    if resolution >= 5.0:
        return 1.8
    if resolution >= 2.5:
        return 1.5
    return 1.2


def experiment_diagnostic_mask(
    model: AtmosphereModel,
    hemisphere: str,
) -> np.ndarray:
    """Return the mid-latitude domain outside the polar boundary sponge."""

    latitude = model.grid.lat2d_deg
    selected = latitude >= 0.0 if hemisphere == "north" else latitude <= 0.0
    sponge_rate = model._meridional_sponge_rate_s
    outside_sponge = (
        np.ones(model.grid.shape, dtype=bool)
        if sponge_rate is None
        else sponge_rate <= np.finfo(float).eps
    )
    return selected & (np.abs(latitude) >= 20.0) & outside_sponge


def experiment_front_reporting_mask(
    model: AtmosphereModel,
    hemisphere: str,
) -> np.ndarray:
    """Front-symbol domain with one undamped row kept as a stencil guard."""

    diagnostic = experiment_diagnostic_mask(model, hemisphere)
    if not np.any(diagnostic):
        return diagnostic
    outer = float(np.max(np.abs(model.grid.lat2d_deg[diagnostic])))
    return diagnostic & (
        np.abs(model.grid.lat2d_deg) <= outer - model.config.dlat_deg + 1.0e-9
    )


def capture_snapshot(
    model: AtmosphereModel,
    level: int,
    *,
    hemisphere: str,
    previous_theta_k: np.ndarray | None,
    previous_time_hours: float | None,
) -> RenderSnapshot:
    level_index = model.level_index(level)
    u, v = terrain_filled_wind(model, level_index)
    temperature = model.temperature_k[level_index].copy()
    pressure_hpa = float(model.pressure_hpa[level_index])
    theta = temperature * (1000.0 / pressure_hpa) ** model.config.kappa
    time_hours = model.time_seconds / 3600.0
    delta_seconds = None
    if previous_theta_k is not None and previous_time_hours is not None:
        delta_seconds = (time_hours - previous_time_hours) * 3600.0
    diagnostic_mask = experiment_diagnostic_mask(model, hemisphere)
    # One-cell halo lets the gradient/TFP stencils at the diagnostic edge use
    # neighbouring values; final symbols and metrics are still clipped to the
    # completely undamped domain.
    front_computation_mask = maximum_filter(
        diagnostic_mask.astype(np.uint8),
        size=(3, 1),
    ) > 0
    fronts = diagnose_fronts(
        theta,
        u,
        v,
        grid=model.grid,
        previous_thermal_field_k=previous_theta_k,
        time_delta_seconds=delta_seconds,
        valid_mask=model.active[level_index] & front_computation_mask,
        min_gradient_k_per_100km=_front_gradient_threshold(model.config.dlon_deg),
        min_normal_motion_m_s=0.6,
        smoothing_passes=1,
    )
    geopotential = model.compute_geopotential()[level_index] / model.config.gravity_m_s2
    # Diagnose omega from the same accepted RK state as every other field in
    # this snapshot. ``last_omega_pa_s`` belongs to the last SSPRK3 stage.
    omega = model._diagnose_omega_and_mass_tendency(
        model.u,
        model.v,
        model.temperature_k,
        model.surface_pressure_anomaly_pa,
    )[0][level_index]
    return RenderSnapshot(
        time_hours=time_hours,
        u_m_s=u.copy(),
        v_m_s=v.copy(),
        temperature_k=temperature,
        potential_temperature_k=theta.copy(),
        pressure_anomaly_hpa=model.sea_level_pressure_anomaly_hpa().copy(),
        omega_pa_s=omega.copy(),
        geopotential_height_m=geopotential.copy(),
        active=model.active[level_index].copy(),
        fronts=fronts,
    )


def simulate_baroclinic_sequence(
    settings: RenderSettings,
    report: ProgressCallback,
) -> tuple[AtmosphereModel, DisplaySurface, list[RenderSnapshot]]:
    """Integrate a published dry baroclinic-wave experiment for playback."""

    diffusion_rate = 2.0e-6 * (settings.resolution / 5.0) ** 2
    # A 950 hPa bottom reduces the unresolved surface-to-lowest-level gap in
    # the coarse default run while remaining safely above its ~990 hPa mature
    # low. Finer runs retain 900 hPa because their cyclone may deepen further.
    pressure_bottom_hpa = 950 if settings.resolution >= 5.0 else 900
    model = AtmosphereModel(
        ModelConfig(
            dlon_deg=settings.resolution,
            dlat_deg=settings.resolution,
            dt_seconds=settings.timestep_seconds,
            pressure_bottom_hpa=pressure_bottom_hpa,
            advection_scheme="tvd",
            horizontal_diffusion_rate_s=diffusion_rate,
            vertical_mixing_rate_s=1.0e-7,
            surface_pressure_anomaly_limit_pa=12_000.0,
            newtonian_relaxation_rate_scale=0.0,
            tibet_height_scale=settings.tibet_scale,
            land_heating_scale=settings.land_heating_scale,
            ocean_current_scale=settings.ocean_current_scale,
            seasonal_phase=0.0,
        )
    )
    display_surface = load_display_surface(settings.tibet_scale)
    configure_baroclinic_wave_model(
        model,
        spec=DryBaroclinicWaveSpec(
            jet_max_m_s=settings.jet_strength,
            perturbation_amplitude_m_s=settings.perturbation_amplitude,
            perturbation_hemisphere=settings.hemisphere,
        ),
        add_perturbation=settings.perturbation_amplitude > 0.0,
    )
    # configure_baroclinic_wave_model removes all project-specific forcing.
    # Restore only weak, resolution-scaled numerical mixing that suppresses
    # two-grid noise without providing an energy source or a prescribed low.
    model.config = model.config.with_updates(
        advection_scheme="tvd",
        horizontal_diffusion_rate_s=diffusion_rate,
        vertical_mixing_rate_s=1.0e-7,
        surface_pressure_anomaly_limit_pa=12_000.0,
    )
    model.grid.config = model.config

    total_steps = max(1, int(round(settings.spinup_hours * 3600.0 / settings.timestep_seconds)))
    frame_count = max(2, int(settings.frames))
    playback_fraction = np.linspace(0.0, 1.0, frame_count)
    # The published life cycle spends many days in weak linear growth and then
    # develops rapidly.  Keep the full hour-0 to day-10 sequence, but sample
    # later model time more densely so the resolved cyclogenesis and fronts do
    # not flash by in only the last few playback frames.
    model_time_fraction = 1.0 - (1.0 - playback_fraction) ** 2
    targets = np.rint(total_steps * model_time_fraction).astype(int)
    snapshots: list[RenderSnapshot] = []
    current_step = 0
    level_index = model.level_index(settings.level)
    theta_factor = (1000.0 / float(model.pressure_hpa[level_index])) ** model.config.kappa
    diagnostic_history: list[tuple[float, np.ndarray]] = [
        (0.0, model.temperature_k[level_index].copy() * theta_factor)
    ]
    history_stride_steps = max(
        1,
        int(round(3.0 * 3600.0 / settings.timestep_seconds)),
    )
    next_history_step = history_stride_steps
    classification_lag_hours = 6.0
    for frame_index, target_step in enumerate(targets):
        while current_step < target_step:
            step_limit = min(int(target_step), next_history_step)
            chunk = min(24, step_limit - current_step)
            model.step(chunk)
            current_step += chunk
            if current_step == next_history_step:
                diagnostic_history.append(
                    (
                        model.time_seconds / 3600.0,
                        model.temperature_k[level_index].copy() * theta_factor,
                    )
                )
                next_history_step += history_stride_steps
            progress = current_step / total_steps
            report(
                0.03 + 0.57 * progress,
                "model",
                f"干斜压积分 {model.time_seconds / 86_400.0:.2f}/{settings.spinup_hours / 24.0:.2f} d",
            )
        current_time_hours = model.time_seconds / 3600.0
        desired_reference_time = current_time_hours - classification_lag_hours
        reference = None
        if desired_reference_time >= 0.0:
            candidates = [
                item for item in diagnostic_history
                if item[0] < current_time_hours - 1.0e-9
            ]
            if candidates:
                reference = min(
                    candidates,
                    key=lambda item: abs(item[0] - desired_reference_time),
                )
        snapshot = capture_snapshot(
            model,
            settings.level,
            hemisphere=settings.hemisphere,
            previous_theta_k=None if reference is None else reference[1],
            previous_time_hours=None if reference is None else reference[0],
        )
        snapshots.append(snapshot)
        report(
            0.03 + 0.57 * current_step / total_steps,
            "model",
            f"已保存模式时次 {frame_index + 1}/{frame_count}",
        )
    return model, display_surface, snapshots


def _empty_front_diagnostics(shape: tuple[int, int]) -> FrontDiagnostics:
    """Return a neutral front payload for circulation snapshots.

    Front diagnosis belongs to the baroclinic-wave experiment.  Keeping the
    same snapshot type lets both experiments share the animation pipeline
    without spending circulation spin-up time on an unused diagnostic.
    """

    scalar = np.zeros(shape, dtype=float)
    boolean = np.zeros(shape, dtype=bool)
    integer = np.zeros(shape, dtype=np.int8)
    return FrontDiagnostics(
        gradient_x_k_m=scalar,
        gradient_y_k_m=scalar,
        gradient_magnitude_k_per_100km=scalar,
        thermal_front_parameter_k_per_100km2=scalar,
        kinematic_frontogenesis_k_per_100km_per_3h=scalar,
        front_zone=boolean,
        front_line=boolean,
        front_type=integer,
        classification_confidence=scalar,
        normal_motion_m_s=scalar,
        classification_tendency_k_s=scalar,
        classification_method="not computed for circulation experiment",
    )


def capture_circulation_snapshot(
    model: AtmosphereModel,
    level: int,
) -> RenderSnapshot:
    """Capture every rendered circulation field from one accepted RK state."""

    level_index = model.level_index(level)
    # Preserve the actual prognostic wind and the terrain mask.  The same
    # accepted state drives the colour layer, particles, and quality gate.
    u = model.u[level_index].copy()
    v = model.v[level_index].copy()
    temperature = model.temperature_k[level_index].copy()
    pressure_hpa = float(model.pressure_hpa[level_index])
    theta = temperature * (1000.0 / pressure_hpa) ** model.config.kappa
    omega = model._diagnose_omega_and_mass_tendency(
        model.u,
        model.v,
        model.temperature_k,
        model.surface_pressure_anomaly_pa,
    )[0][level_index]
    geopotential = model.compute_geopotential()[level_index] / model.config.gravity_m_s2
    return RenderSnapshot(
        time_hours=model.time_seconds / 3600.0,
        u_m_s=u.copy(),
        v_m_s=v.copy(),
        temperature_k=temperature,
        potential_temperature_k=theta.copy(),
        pressure_anomaly_hpa=model.sea_level_pressure_anomaly_hpa().copy(),
        omega_pa_s=omega.copy(),
        geopotential_height_m=geopotential.copy(),
        active=model.active[level_index].copy(),
        fronts=_empty_front_diagnostics(model.grid.shape),
    )


def simulate_circulation_sequence(
    settings: RenderSettings,
    report: ProgressCallback,
) -> tuple[ThreeCellAtmosphereModel, DisplaySurface, list[RenderSnapshot], float]:
    """Spin up a Held--Suarez-like dry circulation, then sample a time window."""

    if settings.analysis_hours <= 0.0:
        raise ValueError("analysis_hours must be positive for circulation playback")
    if settings.spinup_hours < 0.0:
        raise ValueError("spinup_hours cannot be negative")

    # ``horizontal_diffusion_rate_s`` multiplies an index-space Laplacian.
    # Scaling as inverse grid spacing squared keeps the physical diffusivity
    # approximately resolution independent.
    diffusion_rate = 3.0e-6 * (5.0 / settings.resolution) ** 2
    seasonal_phase = float(
        np.clip(
            settings.effective_seasonal_heat_equator_deg / 23.44,
            -1.0,
            1.0,
        )
    )
    model = ThreeCellAtmosphereModel(
        ModelConfig(
            dlon_deg=settings.resolution,
            dlat_deg=settings.resolution,
            lat_limit_deg=77.5,
            dt_seconds=settings.timestep_seconds,
            advection_scheme="tvd",
            horizontal_diffusion_rate_s=diffusion_rate,
            divergence_damping_m2_s=(
                2.0e6 * (settings.resolution / 5.0) ** 2
            ),
            vertical_mixing_rate_s=2.0e-6,
            surface_pressure_anomaly_limit_pa=12_000.0,
            tibet_height_scale=settings.tibet_scale,
            land_heating_scale=settings.land_heating_scale,
            ocean_current_scale=settings.ocean_current_scale,
            seasonal_phase=seasonal_phase,
        )
    )
    display_surface = load_display_surface(settings.tibet_scale)
    configure_orographic_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            equator_to_pole_contrast_k=settings.equator_to_pole_contrast_k,
            surface_drag_days=settings.surface_drag_days,
            seasonal_heat_equator_deg=(
                settings.effective_seasonal_heat_equator_deg
            ),
            # Two full grid rows absorb the artificial solid-wall reflection
            # before it can create a one-cell polar wind spike.  The wind-belt
            # gate still samples the undamped 65--67.5 degree polar flank.
            polar_sponge_width_rows=10.0 / settings.resolution,
            polar_sponge_e_folding_hours=1.0,
            random_seed=model.config.random_seed,
        ),
        OrographicCirculationSpec(),
    )
    # The closure prescribes only a mass-neutral zonal-mean meridional target;
    # zonal wind still develops through the core's Coriolis and drag terms.
    model.set_three_cell_closure(
        ThreeCellSpec(
            equator_to_pole_contrast_k=settings.equator_to_pole_contrast_k,
            thermal_equator_deg=settings.effective_seasonal_heat_equator_deg,
        )
    )
    model.grid.config = model.config

    spinup_steps = max(
        0,
        int(round(settings.spinup_hours * 3600.0 / settings.timestep_seconds)),
    )
    completed = 0
    while completed < spinup_steps:
        chunk = min(24, spinup_steps - completed)
        model.step(chunk)
        completed += chunk
        report(
            0.02 + 0.40 * completed / max(1, spinup_steps),
            "spinup",
            (
                f"Held-Suarez 环流自旋 "
                f"{model.time_seconds / 86_400.0:.2f}/{settings.spinup_hours / 24.0:.2f} d"
            ),
        )
    if spinup_steps == 0:
        report(0.42, "spinup", "跳过自旋，直接进入连续分析窗")

    analysis_start_hours = model.time_seconds / 3600.0
    analysis_steps = max(
        1,
        int(round(settings.analysis_hours * 3600.0 / settings.timestep_seconds)),
    )
    frame_count = max(2, int(settings.frames))
    targets = np.rint(np.linspace(0, analysis_steps, frame_count)).astype(int)
    snapshots: list[RenderSnapshot] = []
    completed = 0
    for frame_index, target in enumerate(targets):
        while completed < int(target):
            chunk = min(24, int(target) - completed)
            model.step(chunk)
            completed += chunk
            report(
                0.42 + 0.20 * completed / analysis_steps,
                "analysis",
                (
                    f"连续分析窗 {completed * settings.timestep_seconds / 3600.0:.1f}"
                    f"/{settings.analysis_hours:.1f} h"
                ),
            )
        snapshots.append(capture_circulation_snapshot(model, settings.level))
        report(
            0.42 + 0.20 * completed / analysis_steps,
            "analysis",
            f"已保存连续模式时次 {frame_index + 1}/{frame_count}",
        )
    return model, display_surface, snapshots, analysis_start_hours


def circulation_wind_belt_statistics(
    model: AtmosphereModel,
    snapshots: list[RenderSnapshot],
    level_hpa: float,
) -> dict[str, object]:
    """Compute equal-time, area-weighted wind-belt means over the analysis window."""

    latitude = model.grid.lat2d_deg
    area = model.grid.area_weight

    def summarize(low: float, high: float, *, easterly: bool) -> tuple[float, float]:
        latitude_mask = (latitude >= low) & (latitude <= high)
        denominator = 0.0
        mean = 0.0
        desired = 0.0
        for snapshot in snapshots:
            weights = np.where(latitude_mask & snapshot.active, area, 0.0)
            denominator += float(np.sum(weights))
            mean += float(np.sum(snapshot.u_m_s * weights))
            desired += float(
                np.sum(
                    weights
                    * (
                        snapshot.u_m_s < 0.0
                        if easterly
                        else snapshot.u_m_s > 0.0
                    )
                )
            )
        if denominator <= 0.0:
            return float("nan"), float("nan")
        return mean / denominator, desired / denominator

    tropical, tropical_fraction = summarize(-20.0, 20.0, easterly=True)
    north_mid, north_mid_fraction = summarize(30.0, 60.0, easterly=False)
    south_mid, south_mid_fraction = summarize(-60.0, -30.0, easterly=False)
    north_polar, north_polar_fraction = summarize(65.0, 80.0, easterly=True)
    south_polar, south_polar_fraction = summarize(-80.0, -65.0, easterly=True)
    window_mean_passed = bool(
        tropical < 0.0
        and north_mid > 0.0
        and south_mid > 0.0
        and north_polar < 0.0
        and south_polar < 0.0
        and tropical_fraction >= 0.5
        and north_mid_fraction >= 0.5
        and south_mid_fraction >= 0.5
        and north_polar_fraction >= 0.5
        and south_polar_fraction >= 0.5
    )

    def one_time_mean(snapshot: RenderSnapshot, low: float, high: float) -> float:
        weights = np.where(
            (latitude >= low) & (latitude <= high) & snapshot.active,
            area,
            0.0,
        )
        return float(np.sum(snapshot.u_m_s * weights) / np.sum(weights))

    def high_wavenumber_metrics(snapshot: RenderSnapshot) -> tuple[float, float]:
        """Return tropical k>=8 RMS and its share of zonal-eddy energy."""

        selected_rows = np.flatnonzero(np.abs(model.grid.lat_deg) <= 10.0)
        high_variances: list[float] = []
        total_variances: list[float] = []
        for row in selected_rows:
            valid = snapshot.active[row]
            if np.count_nonzero(valid) < max(8, model.nx // 2):
                continue
            values = snapshot.u_m_s[row].copy()
            row_mean = float(np.mean(values[valid]))
            anomaly = np.where(valid, values - row_mean, 0.0)
            spectrum = np.fft.rfft(anomaly)
            spectrum[:8] = 0.0
            high_pass = np.fft.irfft(spectrum, n=model.nx)
            high_variances.append(float(np.mean(high_pass[valid] ** 2)))
            total_variances.append(float(np.mean(anomaly[valid] ** 2)))
        if not high_variances:
            return 0.0, 0.0
        high_variance = float(np.mean(high_variances))
        total_variance = float(np.mean(total_variances))
        fraction = high_variance / max(total_variance, 1.0e-12)
        return float(np.sqrt(high_variance)), float(np.clip(fraction, 0.0, 1.0))

    sampled_time_passes = [
        (
            one_time_mean(snapshot, -20.0, 20.0) < 0.0
            and one_time_mean(snapshot, 30.0, 60.0) > 0.0
            and one_time_mean(snapshot, -60.0, -30.0) > 0.0
            and one_time_mean(snapshot, 65.0, 80.0) < 0.0
            and one_time_mean(snapshot, -80.0, -65.0) < 0.0
        )
        for snapshot in snapshots
    ]
    every_sample_passed = bool(sampled_time_passes and all(sampled_time_passes))
    level_max_each_sample = [
        float(np.max(np.hypot(snapshot.u_m_s, snapshot.v_m_s)[snapshot.active]))
        for snapshot in snapshots
    ]
    high_wave_metrics = [high_wavenumber_metrics(snapshot) for snapshot in snapshots]
    high_wave_rms = [item[0] for item in high_wave_metrics]
    high_wave_fraction = [item[1] for item in high_wave_metrics]
    high_wave_growth_ok = bool(
        high_wave_rms
        and high_wave_rms[-1] <= max(0.75, 2.0 * high_wave_rms[0])
    )
    high_wave_stable = bool(
        high_wave_rms
        and max(high_wave_rms) <= 1.5
        and high_wave_growth_ok
    )
    final_column_wind = model.wind_speed_m_s()
    # The compact top is a simplified 50--150 hPa sponge-free layer and can
    # carry a strong idealized stratospheric jet.  The product and its wind
    # belts are tropospheric, so reject instability over p >= 200 hPa while
    # reporting no claim about quantitative stratospheric winds.
    troposphere = (
        model.active
        & (model.pressure_hpa[:, None, None] >= 200.0)
    )
    final_tropospheric_max = float(np.max(final_column_wind[troposphere]))
    numerically_stable = bool(
        np.all(np.isfinite(level_max_each_sample))
        and np.isfinite(final_tropospheric_max)
        and max(level_max_each_sample, default=float("inf")) <= 60.0
        and final_tropospheric_max <= 100.0
        and high_wave_stable
    )
    passed = window_mean_passed and every_sample_passed and numerically_stable
    return {
        "pressure_hpa": float(model.pressure_hpa[model.level_index(level_hpa)]),
        "time_samples": len(snapshots),
        "bands_deg": {
            "tropical": [-20.0, 20.0],
            "northern_midlatitude": [30.0, 60.0],
            "southern_midlatitude": [-60.0, -30.0],
            "northern_polar": [65.0, 80.0],
            "southern_polar": [-80.0, -65.0],
        },
        "tropical_mean_m_s": round(tropical, 4),
        "northern_midlatitude_mean_m_s": round(north_mid, 4),
        "southern_midlatitude_mean_m_s": round(south_mid, 4),
        "northern_polar_mean_m_s": round(north_polar, 4),
        "southern_polar_mean_m_s": round(south_polar, 4),
        "tropical_easterly_fraction": round(tropical_fraction, 4),
        "northern_midlatitude_westerly_fraction": round(north_mid_fraction, 4),
        "southern_midlatitude_westerly_fraction": round(south_mid_fraction, 4),
        "northern_polar_easterly_fraction": round(north_polar_fraction, 4),
        "southern_polar_easterly_fraction": round(south_polar_fraction, 4),
        "gate": {
            "criteria": (
                "analysis-window tropical mean u < 0 and both 30-60 degree "
                "hemisphere means u > 0, with each matching-direction area fraction >= 0.5; "
                "both 65-80 degree polar means u < 0 with easterly fractions >= 0.5; "
                "all five mean directions must hold at every sampled model time; "
                "sampled-level wind <= 60 m/s and final p>=200 hPa wind <= 100 m/s; "
                "tropical zonal k>=8 RMS <= 1.5 m/s without late-window growth"
            ),
            "window_mean_passed": window_mean_passed,
            "every_sampled_time_passed": every_sample_passed,
            "failed_sample_count": sampled_time_passes.count(False),
            "polar_bands_are_report_only": False,
            "maximum_level_wind_m_s": round(max(level_max_each_sample), 4),
            "maximum_final_tropospheric_wind_m_s": round(final_tropospheric_max, 4),
            "level_wind_limit_m_s": 60.0,
            "tropospheric_wind_limit_m_s": 100.0,
            "tropical_high_wavenumber_start_rms_m_s": round(high_wave_rms[0], 5),
            "tropical_high_wavenumber_end_rms_m_s": round(high_wave_rms[-1], 5),
            "tropical_high_wavenumber_max_rms_m_s": round(max(high_wave_rms), 5),
            "tropical_high_wavenumber_max_energy_fraction": round(
                max(high_wave_fraction),
                5,
            ),
            "tropical_high_wavenumber_limit_m_s": 1.5,
            "tropical_high_wavenumber_growth_passed": high_wave_growth_ok,
            "tropical_high_wavenumber_passed": high_wave_stable,
            "numerically_stable": numerically_stable,
            "passed": passed,
        },
    }


def sequence_layer_limits(
    snapshots: list[RenderSnapshot],
    layer: str,
) -> tuple[float, float]:
    if layer == "wind":
        return 0.0, 45.0
    values: list[np.ndarray] = []
    for snapshot in snapshots:
        field, active = snapshot.field(layer)
        finite = field[active & np.isfinite(field)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    combined = np.concatenate(values)
    if layer == "zonalWind":
        extent = float(np.percentile(np.abs(combined), 99.0))
        extent = min(70.0, max(10.0, extent))
        return -extent, extent
    if layer in {"pressure", "omega"}:
        extent = float(np.percentile(np.abs(combined), 99.0))
        floor = 12.0 if layer == "pressure" else 0.05
        ceiling = 100.0 if layer == "pressure" else 1.0
        extent = min(ceiling, max(floor, extent))
        return -extent, extent
    lower, upper = np.percentile(combined, [1.0, 99.0])
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return float(lower), float(upper)


def draw_front_overlay(
    image: Image.Image,
    model: AtmosphereModel,
    snapshot: RenderSnapshot,
    viewport: Viewport,
    hemisphere: str,
) -> Image.Image:
    diagnostics = snapshot.fronts
    diagnostic_mask = experiment_diagnostic_mask(model, hemisphere)
    front_mask = experiment_front_reporting_mask(model, hemisphere)
    line = (
        diagnostics.front_line
        & front_mask
        & (diagnostics.classification_confidence >= 0.44)
    )
    result = image.convert("RGBA")
    draw = ImageDraw.Draw(result, "RGBA")
    colours = {
        FrontType.COLD: (54, 165, 255, 255),
        FrontType.WARM: (244, 74, 91, 255),
        FrontType.STATIONARY: (242, 214, 92, 255),
    }
    cell_pixels = viewport.width * model.config.dlon_deg / viewport.lon_span
    half_segment = max(5.0, min(11.0, 0.58 * cell_pixels))
    symbol_size = max(4.0, min(8.0, 0.38 * cell_pixels))
    line_width = 3 if model.config.dlon_deg >= 5.0 else 2
    symbol_stride = 1 if model.config.dlon_deg >= 5.0 else (2 if model.config.dlon_deg >= 2.5 else 3)
    for front_type, colour in colours.items():
        native = line & (diagnostics.front_type == int(front_type))
        for point_index, (iy, ix) in enumerate(np.argwhere(native)):
            point_lon = float(model.grid.lon2d_deg[iy, ix])
            point_lat = float(model.grid.lat2d_deg[iy, ix])
            centre_x, centre_y = project(point_lon, point_lat, viewport)
            grad_x = float(diagnostics.gradient_x_k_m[iy, ix])
            grad_y = float(diagnostics.gradient_y_k_m[iy, ix])
            magnitude = np.hypot(grad_x, grad_y)
            if magnitude <= 1.0e-15:
                continue

            normal_east = grad_x / magnitude
            normal_north = grad_y / magnitude
            cos_lat = max(0.2, float(np.cos(np.deg2rad(point_lat))))
            normal_screen = np.asarray(
                [normal_east / cos_lat, -normal_north],
                dtype=float,
            )
            normal_screen /= max(float(np.hypot(*normal_screen)), 1.0e-12)
            tangent_screen = np.asarray(
                [-normal_north / cos_lat, -normal_east],
                dtype=float,
            )
            tangent_screen /= max(float(np.hypot(*tangent_screen)), 1.0e-12)

            start = (
                centre_x - half_segment * tangent_screen[0],
                centre_y - half_segment * tangent_screen[1],
            )
            end = (
                centre_x + half_segment * tangent_screen[0],
                centre_y + half_segment * tangent_screen[1],
            )
            draw.line((start, end), fill=colour, width=line_width)
            if point_index % symbol_stride:
                continue

            if front_type == FrontType.COLD:
                base_a = (
                    centre_x - 0.62 * symbol_size * tangent_screen[0],
                    centre_y - 0.62 * symbol_size * tangent_screen[1],
                )
                base_b = (
                    centre_x + 0.62 * symbol_size * tangent_screen[0],
                    centre_y + 0.62 * symbol_size * tangent_screen[1],
                )
                apex = (
                    centre_x + symbol_size * normal_screen[0],
                    centre_y + symbol_size * normal_screen[1],
                )
                draw.polygon((base_a, base_b, apex), fill=colour)
            elif front_type == FrontType.WARM:
                movement = -normal_screen
                angle = float(np.rad2deg(np.arctan2(movement[1], movement[0])))
                box = (
                    centre_x - symbol_size,
                    centre_y - symbol_size,
                    centre_x + symbol_size,
                    centre_y + symbol_size,
                )
                draw.pieslice(
                    box,
                    start=angle - 90.0,
                    end=angle + 90.0,
                    fill=colour,
                )
            else:
                radius = 0.55 * symbol_size
                draw.ellipse(
                    (
                        centre_x - radius,
                        centre_y - radius,
                        centre_x + radius,
                        centre_y + radius,
                    ),
                    fill=colour,
                )

    # A low marker is a diagnosis of the pressure minimum, not an inserted
    # cyclone location.  Suppress it until a meaningful closed anomaly grows.
    candidate = diagnostic_mask
    pressure = np.where(candidate, snapshot.pressure_anomaly_hpa, np.inf)
    minimum_index = np.unravel_index(int(np.argmin(pressure)), pressure.shape)
    minimum = float(pressure[minimum_index])
    if np.isfinite(minimum) and minimum <= -5.0:
        marker_lon = float(model.grid.lon2d_deg[minimum_index])
        marker_lat = float(model.grid.lat2d_deg[minimum_index])
        x, y = project(marker_lon, marker_lat, viewport)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=(9, 23, 34, 190), outline=(255, 255, 255, 220), width=1)
        draw.text((x - 3, y - 6), "L", fill=(255, 255, 255, 255))
    return result


def save_webp_animation(
    frames: list[Image.Image],
    output: Path,
    fps: int,
    *,
    quality: int,
) -> None:
    if not frames:
        raise ValueError("cannot encode an empty animation")
    # Pillow/WebP coalesces visually identical consecutive images into one
    # longer frame.  That is normally a useful optimisation, but here every
    # image has a corresponding model timestamp and must remain aligned with
    # the particle animation and diagnostics timeline.  Encode the frame index
    # in the alpha channel of two south-pole pixels.  WebGL ignores texture
    # alpha, and the change is imperceptible in the two-dimensional fallback.
    converted: list[Image.Image] = []
    for index, frame in enumerate(frames):
        rgba = frame.convert("RGBA")
        y = max(0, rgba.height - 1)
        for x, digit in (
            (0, index % 250),
            (min(1, rgba.width - 1), (index // 250) % 250),
        ):
            red, green, blue, _ = rgba.getpixel((x, y))
            rgba.putpixel((x, y), (red, green, blue, 255 - digit))
        converted.append(rgba)
    converted[0].save(
        output,
        save_all=len(converted) > 1,
        append_images=converted[1:],
        duration=round(1000 / fps),
        loop=0,
        format="WEBP",
        lossless=False,
        quality=quality,
        method=4,
        minimize_size=True,
        exact=True,
    )


def new_particles(
    rng: np.random.Generator,
    count: int,
    viewport: Viewport,
    lat_bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon = rng.uniform(viewport.lon_min, viewport.lon_max, count)
    lat_min, lat_max = lat_bounds or (viewport.lat_min, viewport.lat_max)
    lat_min = max(viewport.lat_min, float(lat_min))
    lat_max = min(viewport.lat_max, float(lat_max))
    sin_min = np.sin(np.deg2rad(lat_min))
    sin_max = np.sin(np.deg2rad(lat_max))
    lat = np.rad2deg(np.arcsin(rng.uniform(sin_min, sin_max, count)))
    age = rng.integers(0, 100, count)
    life = rng.integers(68, 145, count)
    return lon, lat, age, life


def render_particles(
    model: AtmosphereModel,
    level: int,
    output: Path,
    viewport: Viewport,
    frames: int,
    fps: int,
    particle_count: int,
    flow_speed: float,
    trail_decay: float,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    report = progress or (lambda _progress, _message: None)
    rng = np.random.default_rng(20_260_825 + level)
    latitude_bounds = (float(model.grid.lat_deg[0]), float(model.grid.lat_deg[-1]))
    lon, lat, age, life = new_particles(
        rng,
        particle_count,
        viewport,
        latitude_bounds,
    )
    trail_image = Image.new("L", (viewport.width, viewport.height), 0)
    rendered: list[Image.Image] = []
    warmup_frames = min(24, max(6, frames // 2))
    total_iterations = frames + warmup_frames
    fade_table = [int(value * trail_decay) for value in range(256)]
    level_index = model.level_index(level)
    u_field, v_field = terrain_filled_wind(model, level_index)

    for frame_index in range(total_iterations):
        trail_image = trail_image.point(fade_table)
        u = grid_sample(u_field, lon, lat, model)
        v = grid_sample(v_field, lon, lat, model)
        speed = np.hypot(u, v)
        previous_lon = lon.copy()
        previous_lat = lat.copy()
        visual_seconds = 6_000.0 * flow_speed
        lon = lon + u * visual_seconds / (111_320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.18))
        lat = lat + v * visual_seconds / 111_320.0
        if viewport.is_global:
            lon = (lon - viewport.lon_min) % 360.0 + viewport.lon_min
        age += 1

        expired = (
            (~np.isfinite(speed))
            | (speed < 0.025)
            | (age > life)
            | (lat < latitude_bounds[0])
            | (lat > latitude_bounds[1])
            | (lon < viewport.lon_min)
            | (lon > viewport.lon_max)
        )
        replacement = int(expired.sum())
        if replacement:
            new_lon, new_lat, _new_age, new_life = new_particles(
                rng,
                replacement,
                viewport,
                latitude_bounds,
            )
            lon[expired], lat[expired], age[expired], life[expired] = new_lon, new_lat, 0, new_life

        overlay = Image.new("L", (viewport.width, viewport.height), 0)
        draw = ImageDraw.Draw(overlay)
        for index in np.flatnonzero(~expired):
            x0, y0 = project(float(previous_lon[index]), float(previous_lat[index]), viewport)
            x1, y1 = project(float(lon[index]), float(lat[index]), viewport)
            if abs(x1 - x0) > viewport.width * 0.25:
                continue
            strength = min(1.0, 0.28 + float(speed[index]) / 22.0)
            draw.line((x0, y0, x1, y1), fill=int(84 + 116 * strength), width=1)
        trail_image = ImageChops.lighter(trail_image, overlay)
        if frame_index >= warmup_frames:
            rendered.append(trail_image.convert("RGB"))
        report((frame_index + 1) / total_iterations, f"粒子帧 {frame_index + 1}/{total_iterations}")

    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / fps),
        loop=0,
        format="WEBP",
        lossless=False,
        quality=76,
        method=4,
        minimize_size=True,
    )


def render_particle_sequence(
    model: AtmosphereModel,
    snapshots: list[RenderSnapshot],
    output: Path,
    viewport: Viewport,
    fps: int,
    particle_count: int,
    flow_speed: float,
    trail_decay: float,
    progress: Callable[[float, str], None] | None = None,
) -> None:
    """Render particles against successive model states, not one frozen wind."""

    if not snapshots:
        raise ValueError("particle sequence requires at least one model snapshot")
    report = progress or (lambda _progress, _message: None)
    rng = np.random.default_rng(20_260_826 + int(snapshots[0].time_hours))
    latitude_bounds = (float(model.grid.lat_deg[0]), float(model.grid.lat_deg[-1]))
    lon, lat, age, life = new_particles(
        rng,
        particle_count,
        viewport,
        latitude_bounds,
    )
    trail_image = Image.new("L", (viewport.width, viewport.height), 0)
    fade_table = [int(value * trail_decay) for value in range(256)]
    rendered: list[Image.Image] = []
    warmup = min(12, max(4, len(snapshots) // 3))
    sequence: list[RenderSnapshot] = [snapshots[0]] * warmup + snapshots
    for iteration, snapshot in enumerate(sequence):
        trail_image = trail_image.point(fade_table)
        active_before = grid_sample(
            snapshot.active.astype(float),
            lon,
            lat,
            model,
        ) > 0.75
        u = grid_sample(snapshot.u_m_s, lon, lat, model)
        v = grid_sample(snapshot.v_m_s, lon, lat, model)
        speed = np.hypot(u, v)
        previous_lon = lon.copy()
        previous_lat = lat.copy()
        visual_seconds = 6_000.0 * flow_speed
        lon = lon + u * visual_seconds / (
            111_320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.18)
        )
        lat = lat + v * visual_seconds / 111_320.0
        if viewport.is_global:
            lon = (lon - viewport.lon_min) % 360.0 + viewport.lon_min
        age += 1
        active_after = grid_sample(
            snapshot.active.astype(float),
            lon,
            lat,
            model,
        ) > 0.75
        expired = (
            (~np.isfinite(speed))
            | (speed < 0.025)
            | (~active_before)
            | (~active_after)
            | (age > life)
            | (lat < latitude_bounds[0])
            | (lat > latitude_bounds[1])
            | (lon < viewport.lon_min)
            | (lon > viewport.lon_max)
        )
        replacement = int(expired.sum())
        if replacement:
            new_lon, new_lat, _new_age, new_life = new_particles(
                rng,
                replacement,
                viewport,
                latitude_bounds,
            )
            lon[expired], lat[expired], age[expired], life[expired] = (
                new_lon,
                new_lat,
                0,
                new_life,
            )

        overlay = Image.new("L", (viewport.width, viewport.height), 0)
        draw = ImageDraw.Draw(overlay)
        for index in np.flatnonzero(~expired):
            x0, y0 = project(float(previous_lon[index]), float(previous_lat[index]), viewport)
            x1, y1 = project(float(lon[index]), float(lat[index]), viewport)
            if abs(x1 - x0) > viewport.width * 0.25:
                continue
            strength = min(1.0, 0.28 + float(speed[index]) / 36.0)
            draw.line((x0, y0, x1, y1), fill=int(78 + 122 * strength), width=1)
        trail_image = ImageChops.lighter(trail_image, overlay)
        if iteration >= warmup:
            rendered.append(trail_image.convert("RGB"))
        report(
            (iteration + 1) / len(sequence),
            f"连续风场粒子帧 {iteration + 1}/{len(sequence)}",
        )
    save_webp_animation(rendered, output, fps, quality=76)


def _settings_for_manifest(settings: RenderSettings) -> dict[str, object]:
    values = asdict(settings)
    common = {
        "scenario": values["scenario"],
        "resolution": values["resolution"],
        "level": values["level"],
        "region": "world",
        "spinupHours": values["spinup_hours"],
        "frames": values["frames"],
        "fps": values["fps"],
        "particles": values["particles"],
        "flowSpeed": values["flow_speed"],
        "trail": values["trail"],
    }
    if values["scenario"] == "baroclinic":
        return {
            **common,
            "jetStrength": values["jet_strength"],
            "perturbationAmplitude": values["perturbation_amplitude"],
            "hemisphere": values["hemisphere"],
        }
    return {
        **common,
        "season": values["season"],
        "analysisHours": values["analysis_hours"],
        "tibetScale": values["tibet_scale"],
        "landHeatingScale": values["land_heating_scale"],
        "oceanCurrentScale": values["ocean_current_scale"],
        "equatorToPoleContrastK": values["equator_to_pole_contrast_k"],
        "surfaceDragDays": values["surface_drag_days"],
        "seasonalHeatEquatorDeg": settings.effective_seasonal_heat_equator_deg,
    }


def render_baroclinic_assets(
    settings: RenderSettings,
    output: Path,
    *,
    asset_base: str,
    report: ProgressCallback,
    viewport: Viewport,
) -> dict[str, object]:
    model, display_surface, snapshots = simulate_baroclinic_sequence(settings, report)
    diagnostic_mask = experiment_diagnostic_mask(model, settings.hemisphere)
    diagnostic_latitude_limit = float(
        np.max(np.abs(model.grid.lat2d_deg[diagnostic_mask]))
    )
    front_reporting_mask = experiment_front_reporting_mask(
        model,
        settings.hemisphere,
    )
    front_latitude_limit = float(
        np.max(np.abs(model.grid.lat2d_deg[front_reporting_mask]))
    )

    particle_name = f"particles_{settings.level}.webp"
    render_particle_sequence(
        model,
        snapshots,
        output / particle_name,
        viewport,
        settings.fps,
        settings.particles,
        settings.flow_speed,
        settings.trail,
        progress=lambda value, message: report(0.60 + 0.14 * value, "particles", message),
    )

    experiment_layers = {
        key: value for key, value in LAYER_INFO.items() if key != "terrain"
    }
    experiment_layers = {"fronts": FRONT_LAYER_INFO["fronts"], **experiment_layers}
    manifest: dict[str, object] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Atmos20 dry Jablonowski-Williamson baroclinic-wave mechanism experiment; "
            "fronts objectively diagnosed from model output; Natural Earth 1:110m context"
        ),
        "schemaVersion": 4,
        "assetBase": asset_base.rstrip("/") + "/",
        "modelGridDegrees": settings.resolution,
        "width": viewport.width,
        "height": viewport.height,
        "frames": len(snapshots),
        "fps": settings.fps,
        "durationSeconds": len(snapshots) / settings.fps,
        "simulationHours": settings.spinup_hours,
        "levels": [settings.level],
        "defaultLevel": settings.level,
        "defaultLayer": "fronts",
        "viewport": asdict(viewport),
        "settings": _settings_for_manifest(settings),
        "experiment": {
            "kind": "dry_baroclinic_wave",
            "initialCondition": (
                "Jablonowski-Williamson 2006 analytic balanced state sampled "
                "on the model pressure grid"
            ),
            "trigger": "localized Gaussian zonal-wind perturbation only",
            "prescribedCyclone": False,
            "prescribedFronts": False,
            "moistPhysics": False,
            "orographicLift": False,
            "verticalGrid": (
                f"{model.nz} fixed pressure levels, "
                f"{model.config.pressure_bottom_hpa}-{model.config.pressure_top_hpa} hPa, "
                f"{model.config.pressure_step_hpa} hPa spacing"
            ),
            "numerics": (
                "SSPRK3; MC-limited second-order TVD horizontal advection; "
                "weak resolution-scaled horizontal diffusion and vertical mixing"
            ),
            "benchmarkScope": (
                "mechanism demonstration on a truncated latitude-pressure grid; "
                "not a quantitative reference-solution reproduction"
            ),
            "diagnosticLatitudeBandDeg": [20.0, diagnostic_latitude_limit],
            "frontReportingLatitudeBandDeg": [20.0, front_latitude_limit],
            "timelineWindScope": f"{settings.level} hPa in selected undamped hemisphere band",
            "polarBoundary": (
                "balanced-state sponge from "
                f"{model.meridional_sponge_start_latitude_deg:g} to "
                f"{model.config.lat_limit_deg:g} degrees"
            ),
            "polarCapDisplay": "zonal-mean extrapolation outside the dynamical grid",
            "timeSampling": "quadratic ease-out toward mature lifecycle",
            "frontClassificationLagHours": 6.0,
            "frontDiagnosticHistoryHours": 3.0,
            "reference": "https://doi.org/10.1256/qj.06.12",
        },
        "frontLegend": {
            "cold": "#36a5ff",
            "warm": "#f44a5b",
            "stationary": "#f2d65c",
        },
        "timeline": [],
        "layers": {},
        "particles": {str(settings.level): particle_name},
    }

    for snapshot in snapshots:
        line = (
            snapshot.fronts.front_line
            & front_reporting_mask
            & (snapshot.fronts.classification_confidence >= 0.44)
        )
        pressure_values = snapshot.pressure_anomaly_hpa[diagnostic_mask]
        wind_values = np.hypot(snapshot.u_m_s, snapshot.v_m_s)[
            diagnostic_mask & snapshot.active
        ]
        gradient_values = snapshot.fronts.gradient_magnitude_k_per_100km[
            diagnostic_mask
        ]
        manifest["timeline"].append(
            {
                "forecastHour": round(snapshot.time_hours, 3),
                "surfacePressureMinHpa": round(
                    1000.0 + float(np.min(pressure_values)), 2
                ),
                "surfacePressureMaxHpa": round(
                    1000.0 + float(np.max(pressure_values)), 2
                ),
                "maxWindMps": round(float(np.max(wind_values)), 2),
                "maxThetaGradientKPer100km": round(
                    float(np.max(gradient_values)), 3
                ),
                "coldFrontCells": int(
                    np.count_nonzero(line & (snapshot.fronts.front_type == int(FrontType.COLD)))
                ),
                "warmFrontCells": int(
                    np.count_nonzero(line & (snapshot.fronts.front_type == int(FrontType.WARM)))
                ),
            }
        )

    layer_count = len(experiment_layers)
    for layer_index, (layer, (label, english, unit)) in enumerate(experiment_layers.items()):
        limits = sequence_layer_limits(snapshots, layer)
        rendered: list[Image.Image] = []
        for frame_index, snapshot in enumerate(snapshots):
            field, active = snapshot.field(layer)
            background, _, _ = render_field_background(
                model,
                field,
                active,
                layer,
                viewport,
                limits=limits,
                display_surface=display_surface,
                smooth_polar_caps=True,
            )
            if layer in {"fronts", "wind", "temperature", "pressure"}:
                background = draw_front_overlay(
                    background,
                    model,
                    snapshot,
                    viewport,
                    settings.hemisphere,
                )
            rendered.append(background)
            completed = (layer_index + (frame_index + 1) / len(snapshots)) / layer_count
            report(
                0.74 + 0.24 * completed,
                "layers",
                f"{label}时次 {frame_index + 1}/{len(snapshots)}",
            )
        name = f"{layer}_{settings.level}.webp"
        save_webp_animation(rendered, output / name, settings.fps, quality=84)
        manifest["layers"][layer] = {
            "label": label,
            "english": english,
            "unit": unit,
            "animated": True,
            "assets": {str(settings.level): name},
            "ranges": {str(settings.level): [limits[0], limits[1]]},
        }

    report(0.99, "manifest", "正在写入物理实验播放清单")
    manifest_path = output / "manifest.json"
    temporary_path = output / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    report(1.0, "complete", "干温带气旋实验已完成，可以播放")
    return manifest


def render_circulation_assets(
    settings: RenderSettings,
    output: Path,
    *,
    asset_base: str,
    report: ProgressCallback,
    viewport: Viewport,
) -> dict[str, object]:
    """Render one synchronized Held--Suarez analysis window."""

    model, display_surface, snapshots, analysis_start_hours = (
        simulate_circulation_sequence(settings, report)
    )
    if not isinstance(model, ThreeCellAtmosphereModel):
        raise TypeError("circulation renderer requires ThreeCellAtmosphereModel")
    if model.three_cell_closure is None:
        raise RuntimeError("three-cell circulation closure was not installed")
    closure_spec = model.three_cell_closure.spec
    wind_statistics = circulation_wind_belt_statistics(
        model,
        snapshots,
        settings.level,
    )
    wind_statistics["time_window_model_hours"] = [
        round(snapshots[0].time_hours, 3),
        round(snapshots[-1].time_hours, 3),
    ]
    wind_statistics["time_window_duration_hours"] = round(
        snapshots[-1].time_hours - snapshots[0].time_hours,
        3,
    )
    gate_passed = bool(wind_statistics["gate"]["passed"])

    particle_name = f"particles_{settings.level}.webp"
    render_particle_sequence(
        model,
        snapshots,
        output / particle_name,
        viewport,
        settings.fps,
        settings.particles,
        settings.flow_speed,
        settings.trail,
        progress=lambda value, message: report(
            0.62 + 0.13 * value,
            "particles",
            message,
        ),
    )

    manifest: dict[str, object] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Atmos20 terrain-coupled Held-Suarez dry forcing with reduced "
            "three-cell overturning closure and ETOPO 2022 lower boundary"
        ),
        "schemaVersion": 5,
        "assetBase": asset_base.rstrip("/") + "/",
        "modelGridDegrees": settings.resolution,
        "width": viewport.width,
        "height": viewport.height,
        "frames": len(snapshots),
        "fps": settings.fps,
        "durationSeconds": len(snapshots) / settings.fps,
        "simulationHours": settings.spinup_hours + settings.analysis_hours,
        "analysisHours": settings.analysis_hours,
        "levels": [settings.level],
        "defaultLevel": settings.level,
        "defaultLayer": "zonalWind",
        "viewport": asdict(viewport),
        "settings": _settings_for_manifest(settings),
        "qualityGatePassed": gate_passed,
        "experiment": {
            "kind": "orographic_held_suarez_with_reduced_three_cell_closure",
            "initialCondition": (
                "terrain-balanced resting 264 K dry atmosphere with zero-mean, "
                "spatially smoothed 0.02 K thermal noise"
            ),
            "forcing": (
                "axisymmetric Held-Suarez Newtonian thermal relaxation and "
                "local-sigma near-surface Rayleigh drag; ETOPO terrain masking, "
                "hydrostatic surface geopotential, slope lift and form drag; "
                "land/ocean lower-boundary temperature; plus a reduced-order "
                "zonal-mean meridional overturning closure"
            ),
            "prescribedZonalWind": False,
            "prescribedPressureSystems": False,
            "parameterizedMeridionalOverturning": True,
            "meridionalOverturningClosure": (
                "mass-neutral zonal-mean v relaxation representing the lower and "
                "upper branches of the Hadley, Ferrel, and polar cells; zonal wind "
                "is generated by the prognostic Coriolis and drag terms"
            ),
            "threeCellClosureParameters": asdict(closure_spec),
            "prescribedCyclones": False,
            "prescribedFronts": False,
            "moistPhysics": False,
            "interactiveFrontendDynamics": False,
            "orographicLift": True,
            "terrainBlocking": True,
            "undergroundPressureLevelsMasked": True,
            "physicalLowerBoundary": (
                "ETOPO 2022 elevation with hydrostatically balanced base surface pressure"
            ),
            "displayGeography": (
                "1-degree ETOPO shaded relief with Natural Earth boundaries"
            ),
            "orographicCouplingParameters": asdict(OrographicCirculationSpec()),
            "seasonalHeatEquatorDeg": settings.effective_seasonal_heat_equator_deg,
            "equatorToPoleContrastK": settings.equator_to_pole_contrast_k,
            "surfaceDragDays": settings.surface_drag_days,
            "tibetHeightScale": settings.tibet_scale,
            "landHeatingScale": settings.land_heating_scale,
            "oceanCurrentScale": settings.ocean_current_scale,
            "recommendedDiagnosticLevelHpa": 900,
            "polarSpongeStartLatitudeDeg": (
                model.meridional_sponge_start_latitude_deg
            ),
            "spinupHours": settings.spinup_hours,
            "analysisWindowHours": [
                round(analysis_start_hours, 3),
                round(snapshots[-1].time_hours, 3),
            ],
            "timeSampling": "equal intervals over one continuous post-spinup integration",
            "scope": (
                "short dry adjustment experiment; not a statistically equilibrated "
                "Held-Suarez climate"
            ),
            "verticalGrid": (
                f"{model.nz} fixed pressure levels, "
                f"{model.config.pressure_bottom_hpa}-{model.config.pressure_top_hpa} hPa, "
                f"{model.config.pressure_step_hpa} hPa spacing"
            ),
            "numerics": (
                "SSPRK3; MC-limited second-order TVD horizontal advection; "
                "resolution-scaled horizontal diffusion and selective divergence damping"
            ),
            "wind_belt_statistics": wind_statistics,
            "passed": gate_passed,
            "reference": "https://doi.org/10.1175/1520-0477(1994)075<1825:AAPFTI>2.0.CO;2",
        },
        "timeline": [],
        "layers": {},
        "particles": {str(settings.level): particle_name},
    }

    for snapshot in snapshots:
        instant = circulation_wind_belt_statistics(
            model,
            [snapshot],
            settings.level,
        )
        wind_values = np.hypot(snapshot.u_m_s, snapshot.v_m_s)[snapshot.active]
        manifest["timeline"].append(
            {
                "forecastHour": round(
                    snapshot.time_hours - analysis_start_hours,
                    3,
                ),
                "modelHour": round(snapshot.time_hours, 3),
                "maxWindMps": round(float(np.max(wind_values)), 2),
                "tropicalZonalWindMps": instant["tropical_mean_m_s"],
                "northernMidlatitudeZonalWindMps": (
                    instant["northern_midlatitude_mean_m_s"]
                ),
                "southernMidlatitudeZonalWindMps": (
                    instant["southern_midlatitude_mean_m_s"]
                ),
                "northernPolarZonalWindMps": (
                    instant["northern_polar_mean_m_s"]
                ),
                "southernPolarZonalWindMps": (
                    instant["southern_polar_mean_m_s"]
                ),
            }
        )

    layer_count = len(CIRCULATION_LAYER_INFO)
    for layer_index, (layer, (label, english, unit)) in enumerate(
        CIRCULATION_LAYER_INFO.items()
    ):
        if layer == "terrain":
            terrain_active = np.ones(model.grid.shape, dtype=bool)
            limits = layer_limits(
                display_surface.elevation_m,
                np.ones_like(display_surface.elevation_m, dtype=bool),
                "terrain",
            )
        else:
            limits = sequence_layer_limits(snapshots, layer)
        rendered: list[Image.Image] = []
        layer_snapshots = snapshots[:1] if layer == "terrain" else snapshots
        for frame_index, snapshot in enumerate(layer_snapshots):
            if layer == "terrain":
                field = model.boundary.surface_elevation_m
                active = terrain_active
            else:
                field, active = snapshot.field(layer)
            background, _, _ = render_field_background(
                model,
                field,
                active,
                layer,
                viewport,
                limits=limits,
                display_surface=display_surface,
                smooth_polar_caps=True,
            )
            rendered.append(background)
            completed = (
                layer_index + (frame_index + 1) / len(layer_snapshots)
            ) / layer_count
            report(
                0.75 + 0.23 * completed,
                "layers",
                f"{label}连续时次 {frame_index + 1}/{len(layer_snapshots)}",
            )
        name = f"{layer}_{settings.level}.webp"
        save_webp_animation(rendered, output / name, settings.fps, quality=84)
        manifest["layers"][layer] = {
            "label": label,
            "english": english,
            "unit": unit,
            "animated": layer != "terrain",
            "assets": {str(settings.level): name},
            "ranges": {str(settings.level): [limits[0], limits[1]]},
        }

    report(0.99, "manifest", "正在写入连续环流播放清单")
    manifest_path = output / "manifest.json"
    temporary_path = output / "manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    completion = (
        "风带、地形与高波数门禁通过，可以播放"
        if gate_passed
        else "计算完成，但风带或数值质量门禁未通过"
    )
    report(1.0, "complete", completion)
    return manifest


def render_assets(
    settings: RenderSettings,
    output: Path,
    *,
    asset_base: str,
    progress: ProgressCallback | None = None,
    clean_output: bool = False,
) -> dict[str, object]:
    """Run the model and write a complete immutable media bundle."""

    report = progress or _noop_progress
    # All output is a complete 2:1 equirectangular texture for the interactive
    # globe (and the flat compatibility fallback when WebGL is unavailable).
    viewport = WORLD_VIEWPORT
    output.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for old_asset in output.iterdir():
            if old_asset.is_file() and old_asset.suffix.lower() in {".webp", ".json"}:
                old_asset.unlink()

    if settings.scenario == "baroclinic":
        report(0.01, "initializing", "正在初始化干斜压波实验")
        return render_baroclinic_assets(
            settings,
            output,
            asset_base=asset_base,
            report=report,
            viewport=viewport,
        )
    if settings.scenario != "circulation":
        raise ValueError(f"unknown render scenario: {settings.scenario}")
    report(0.01, "initializing", "正在初始化 Held-Suarez 连续环流实验")
    return render_circulation_assets(
        settings,
        output,
        asset_base=asset_base,
        report=report,
        viewport=viewport,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and pre-render Atmos20 map animation assets.")
    parser.add_argument("--scenario", choices=("circulation", "baroclinic"), default="circulation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--particles", type=int, default=2400)
    parser.add_argument("--spinup-hours", type=float, default=120.0)
    parser.add_argument("--analysis-hours", type=float, default=120.0)
    parser.add_argument("--grid-degrees", type=float, choices=(1.0, 2.5, 5.0), default=5.0)
    parser.add_argument("--level", type=int, default=900)
    parser.add_argument("--region", choices=("world",), default="world")
    parser.add_argument("--season", choices=("summer", "equinox", "winter"), default="equinox")
    parser.add_argument("--flow-speed", type=float, default=1.0)
    parser.add_argument("--trail", type=float, default=0.94)
    parser.add_argument("--tibet-scale", type=float, default=1.0)
    parser.add_argument("--land-heating-scale", type=float, default=1.0)
    parser.add_argument("--ocean-current-scale", type=float, default=1.0)
    parser.add_argument("--equator-to-pole-contrast-k", type=float, default=60.0)
    parser.add_argument("--surface-drag-days", type=float, default=1.0)
    parser.add_argument("--seasonal-heat-equator-deg", type=float, default=None)
    parser.add_argument("--jet-strength", type=float, default=35.0)
    parser.add_argument("--perturbation-amplitude", type=float, default=1.0)
    parser.add_argument("--hemisphere", choices=("north", "south"), default="north")
    args = parser.parse_args()

    settings = RenderSettings(
        scenario=args.scenario,
        resolution=args.grid_degrees,
        level=args.level,
        region=args.region,
        season=args.season,
        spinup_hours=args.spinup_hours,
        analysis_hours=args.analysis_hours,
        frames=args.frames,
        fps=args.fps,
        particles=args.particles,
        flow_speed=args.flow_speed,
        trail=args.trail,
        tibet_scale=args.tibet_scale,
        land_heating_scale=args.land_heating_scale,
        ocean_current_scale=args.ocean_current_scale,
        equator_to_pole_contrast_k=args.equator_to_pole_contrast_k,
        surface_drag_days=args.surface_drag_days,
        seasonal_heat_equator_deg=args.seasonal_heat_equator_deg,
        jet_strength=args.jet_strength,
        perturbation_amplitude=args.perturbation_amplitude,
        hemisphere=args.hemisphere,
    )
    render_assets(
        settings,
        args.output,
        asset_base="/assets/prerender/",
        progress=lambda value, stage, message: print(f"{value:6.1%} [{stage}] {message}"),
        clean_output=True,
    )
    total_mb = sum(path.stat().st_size for path in args.output.iterdir() if path.is_file()) / (1024 * 1024)
    print(f"Wrote {args.output} ({total_mb:.1f} MiB total)")


if __name__ == "__main__":
    main()
