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
from scipy.ndimage import gaussian_filter

from atmos20 import AtmosphereModel, ModelConfig


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


VIEWPORTS: dict[str, Viewport] = {
    "east_asia_pacific": Viewport("east_asia_pacific", 55.0, 235.0, -20.0, 70.0),
    "east_asia": Viewport("east_asia", 55.0, 235.0, -20.0, 70.0),
    "asia": Viewport("asia", 20.0, 190.0, -15.0, 75.0),
    "world": Viewport("world", 0.0, 360.0, -77.5, 77.5),
    "global": Viewport("global", 0.0, 360.0, -77.5, 77.5),
}


@dataclass(frozen=True, slots=True)
class RenderSettings:
    resolution: float = 2.5
    level: int = 850
    region: str = "east_asia"
    season: str = "summer"
    spinup_hours: float = 3.0
    frames: int = 72
    fps: int = 24
    particles: int = 3600
    flow_speed: float = 1.0
    trail: float = 0.94
    tibet_scale: float = 1.0
    land_heating_scale: float = 1.0
    ocean_current_scale: float = 1.0

    @property
    def timestep_seconds(self) -> float:
        if self.resolution <= 1.0:
            return 120.0
        if self.resolution <= 2.5:
            return 300.0
        return 600.0

    @property
    def seasonal_phase(self) -> float:
        return {"winter": -1.0, "equinox": 0.0, "summer": 1.0}[self.season]


PALETTES = {
    "wind": ["#3d50a3", "#3374b5", "#269ba9", "#36b875", "#8dcc55", "#e2d653", "#e5974e", "#b64173"],
    "temperature": ["#4545a5", "#347cc2", "#42b7b1", "#c3d779", "#f4c85f", "#e87545", "#b93455"],
    "pressure": ["#513392", "#3d70ba", "#43a9a4", "#d4d18e", "#e39b4e", "#b84961"],
    "omega": ["#67339b", "#397dcc", "#47c4c2", "#d6d6b1", "#e99952", "#bc3f62"],
    "geopotential": ["#343d82", "#3769a6", "#399a9d", "#75b567", "#c7c65b", "#d48b54"],
    "terrain": ["#284f85", "#367d75", "#66a762", "#aabb69", "#b9915e", "#8a684f", "#e1ddd0"],
}

LAYER_INFO = {
    "wind": ("风速", "WIND", "m/s"),
    "temperature": ("温度", "TEMPERATURE", "°C"),
    "pressure": ("气压", "MSLP ANOMALY", "hPa"),
    "omega": ("垂直运动", "OMEGA", "Pa/s"),
    "geopotential": ("位势高度", "GEOPOTENTIAL", "m"),
    "terrain": ("真实地形", "RELIEF", "m"),
}


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
    y = (lat - model.grid.lat_deg[0]) / model.config.dlat_deg
    x0 = np.floor(x).astype(int) % nx
    x1 = (x0 + 1) % nx
    y0 = np.clip(np.floor(y).astype(int), 0, ny - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    tx = x - np.floor(x)
    ty = np.clip(y - np.floor(y), 0.0, 1.0)
    south = field[y0, x0] * (1.0 - tx) + field[y0, x1] * tx
    north = field[y1, x0] * (1.0 - tx) + field[y1, x1] * tx
    return south * (1.0 - ty) + north * ty


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
    if layer == "pressure":
        return -12.0, 12.0
    if layer == "omega":
        return -0.5, 0.5
    if layer == "terrain":
        return 0.0, max(4_800.0, float(np.percentile(finite, 99)))
    lower, upper = np.percentile(finite, [1, 99])
    return min(0.0, float(lower)), max(float(upper), 1.0)


def render_background(
    model: AtmosphereModel,
    layer: str,
    level: int,
    viewport: Viewport,
) -> tuple[Image.Image, float, float]:
    lon, lat = pixel_coordinates(viewport)
    field, active = layer_field(model, layer, level)
    lower, upper = layer_limits(field, active, layer)
    sampled = grid_sample(field, lon, lat, model)
    sampled_active = grid_sample(active.astype(float), lon, lat, model) > 0.5
    wind_stops = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
    weather_rgb = colourize(
        sampled,
        lower,
        upper,
        PALETTES[layer],
        stops=wind_stops if layer == "wind" else None,
    ).astype(float)

    elevation = grid_sample(model.boundary.surface_elevation_m, lon, lat, model)
    land = grid_sample(model.boundary.land_fraction, lon, lat, model) > 0.44
    gradient_y, gradient_x = np.gradient(elevation)
    hillshade = np.clip(0.53 - 0.018 * gradient_x + 0.026 * gradient_y, 0.36, 0.72)
    relief_strength = np.where(land, 0.84 + hillshade * 0.30, 0.96)
    base = np.empty_like(weather_rgb)
    base[~land] = np.asarray([57.0, 93.0, 116.0])
    base[land] = np.asarray([91.0, 126.0, 104.0])
    base *= relief_strength[..., None]
    blend = 0.78 if layer != "terrain" else 0.70
    rgb = blend * weather_rgb + (1.0 - blend) * base
    if layer not in {"terrain", "pressure"}:
        ground = ~sampled_active
        rgb[ground] = 0.88 * base[ground] + 0.12 * weather_rgb[ground]

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    draw_boundaries(image, viewport)
    return image, lower, upper


def velocity(model: AtmosphereModel, level: int, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    level_index = model.level_index(level)
    u, v = terrain_filled_wind(model, level_index)
    return grid_sample(u, lon, lat, model), grid_sample(v, lon, lat, model)


def new_particles(
    rng: np.random.Generator,
    count: int,
    viewport: Viewport,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon = rng.uniform(viewport.lon_min, viewport.lon_max, count)
    sin_min = np.sin(np.deg2rad(viewport.lat_min))
    sin_max = np.sin(np.deg2rad(viewport.lat_max))
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
    lon, lat, age, life = new_particles(rng, particle_count, viewport)
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
            | (lat < viewport.lat_min)
            | (lat > viewport.lat_max)
            | (lon < viewport.lon_min)
            | (lon > viewport.lon_max)
        )
        replacement = int(expired.sum())
        if replacement:
            new_lon, new_lat, _new_age, new_life = new_particles(rng, replacement, viewport)
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


def _settings_for_manifest(settings: RenderSettings) -> dict[str, object]:
    values = asdict(settings)
    return {
        "resolution": values["resolution"],
        "level": values["level"],
        "region": values["region"],
        "season": values["season"],
        "spinupHours": values["spinup_hours"],
        "frames": values["frames"],
        "fps": values["fps"],
        "particles": values["particles"],
        "flowSpeed": values["flow_speed"],
        "trail": values["trail"],
        "tibetScale": values["tibet_scale"],
        "landHeatingScale": values["land_heating_scale"],
        "oceanCurrentScale": values["ocean_current_scale"],
    }


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
    viewport = VIEWPORTS[settings.region]
    output.mkdir(parents=True, exist_ok=True)
    if clean_output:
        for old_asset in output.iterdir():
            if old_asset.is_file() and old_asset.suffix.lower() in {".webp", ".json"}:
                old_asset.unlink()

    report(0.01, "initializing", "正在初始化大气模型")
    model = AtmosphereModel(
        ModelConfig(
            dlon_deg=settings.resolution,
            dlat_deg=settings.resolution,
            dt_seconds=settings.timestep_seconds,
            tibet_height_scale=settings.tibet_scale,
            land_heating_scale=settings.land_heating_scale,
            ocean_current_scale=settings.ocean_current_scale,
            seasonal_phase=settings.seasonal_phase,
        )
    )

    spinup_steps = int(round(settings.spinup_hours * 3600.0 / settings.timestep_seconds))
    if spinup_steps:
        for step_index in range(spinup_steps):
            model.step(1)
            fraction = (step_index + 1) / spinup_steps
            report(0.04 + 0.48 * fraction, "spinup", f"模式积分 {step_index + 1}/{spinup_steps}")
    else:
        report(0.52, "spinup", "跳过模式预积分")

    particle_name = f"particles_{settings.level}.webp"
    render_particles(
        model,
        settings.level,
        output / particle_name,
        viewport,
        settings.frames,
        settings.fps,
        settings.particles,
        settings.flow_speed,
        settings.trail,
        progress=lambda value, message: report(0.52 + 0.34 * value, "particles", message),
    )

    manifest: dict[str, object] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source": f"Atmos20 {settings.resolution:g}-degree {settings.season} backend render; Natural Earth 1:110m boundaries",
        "schemaVersion": 2,
        "assetBase": asset_base.rstrip("/") + "/",
        "modelGridDegrees": settings.resolution,
        "width": viewport.width,
        "height": viewport.height,
        "frames": settings.frames,
        "fps": settings.fps,
        "durationSeconds": settings.frames / settings.fps,
        "levels": [settings.level],
        "defaultLevel": settings.level,
        "defaultLayer": "wind",
        "viewport": asdict(viewport),
        "settings": _settings_for_manifest(settings),
        "layers": {},
        "particles": {str(settings.level): particle_name},
    }

    layer_count = len(LAYER_INFO)
    for layer_index, (layer, (label, english, unit)) in enumerate(LAYER_INFO.items()):
        report(0.86 + 0.12 * layer_index / layer_count, "layers", f"正在渲染图层：{label}")
        background, lower, upper = render_background(model, layer, settings.level, viewport)
        name = f"{layer}_{settings.level}.webp"
        background.convert("RGB").save(output / name, "WEBP", quality=88, method=6)
        manifest["layers"][layer] = {
            "label": label,
            "english": english,
            "unit": unit,
            "assets": {str(settings.level): name},
            "ranges": {str(settings.level): [lower, upper]},
        }

    report(0.99, "manifest", "正在写入播放清单")
    manifest_path = output / "manifest.json"
    temporary_path = output / "manifest.json.tmp"
    temporary_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(manifest_path)
    report(1.0, "complete", "渲染完成，可以播放")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and pre-render Atmos20 map animation assets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--particles", type=int, default=3600)
    parser.add_argument("--spinup-hours", type=float, default=3.0)
    parser.add_argument("--grid-degrees", type=float, choices=(1.0, 2.5, 5.0), default=1.0)
    parser.add_argument("--level", type=int, default=850)
    parser.add_argument("--region", choices=tuple(VIEWPORTS), default="east_asia")
    parser.add_argument("--season", choices=("summer", "equinox", "winter"), default="summer")
    parser.add_argument("--flow-speed", type=float, default=1.0)
    parser.add_argument("--trail", type=float, default=0.94)
    parser.add_argument("--tibet-scale", type=float, default=1.0)
    parser.add_argument("--land-heating-scale", type=float, default=1.0)
    parser.add_argument("--ocean-current-scale", type=float, default=1.0)
    args = parser.parse_args()

    settings = RenderSettings(
        resolution=args.grid_degrees,
        level=args.level,
        region=args.region,
        season=args.season,
        spinup_hours=args.spinup_hours,
        frames=args.frames,
        fps=args.fps,
        particles=args.particles,
        flow_speed=args.flow_speed,
        trail=args.trail,
        tibet_scale=args.tibet_scale,
        land_heating_scale=args.land_heating_scale,
        ocean_current_scale=args.ocean_current_scale,
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
