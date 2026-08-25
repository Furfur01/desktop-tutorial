from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .model import AtmosphereModel


VARIABLES = [
    "Wind speed + sea-level pressure",
    "Temperature",
    "Vertical motion",
    "Geopotential height",
    "Surface temperature",
    "Terrain and active layers",
]

def draw_coastlines(ax: plt.Axes, model: AtmosphereModel) -> None:
    lon = np.r_[model.grid.lon_deg, 360.0]
    land_fraction = np.concatenate(
        [model.boundary.land_fraction, model.boundary.land_fraction[:, :1]],
        axis=1,
    )
    ax.contour(
        lon,
        model.grid.lat_deg,
        land_fraction,
        levels=[0.5],
        colors="0.15",
        linewidths=0.65,
        zorder=8,
    )


def _masked_level(model: AtmosphereModel, array: np.ndarray, k: int) -> np.ma.MaskedArray:
    return np.ma.array(array[k], mask=~model.active[k])


def plot_state(
    model: AtmosphereModel,
    pressure_hpa: int | float = 850,
    variable: str = VARIABLES[0],
    vector_stride: int = 3,
) -> Figure:
    k = model.level_index(pressure_hpa)
    p = int(model.pressure_hpa[k])
    lon = model.grid.lon2d_deg
    lat = model.grid.lat2d_deg
    active = model.active[k]

    fig, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    ax.set_xlim(0, 360)
    ax.set_ylim(model.grid.lat_deg.min(), model.grid.lat_deg.max())
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.grid(linewidth=0.35, alpha=0.35)

    if variable == "Wind speed + sea-level pressure":
        field = _masked_level(model, model.wind_speed_m_s(), k)
        finite = field.compressed()
        vmax = max(5.0, float(np.percentile(finite, 98)) if finite.size else 5.0)
        artist = ax.contourf(lon, lat, field, levels=np.linspace(0, vmax, 15), cmap="viridis", extend="max")
        cbar_label = f"wind speed at {p} hPa (m s⁻¹)"
        slp = model.sea_level_pressure_anomaly_hpa()
        amp = max(1.0, float(np.percentile(np.abs(slp), 97)))
        levels = np.linspace(-amp, amp, 11)
        cs = ax.contour(lon, lat, slp, levels=levels, colors="black", linewidths=0.65)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    elif variable == "Temperature":
        field = _masked_level(model, model.temperature_k - 273.15, k)
        finite = field.compressed()
        lo, hi = (float(np.percentile(finite, 2)), float(np.percentile(finite, 98))) if finite.size else (-20, 20)
        if hi - lo < 2.0:
            lo -= 1.0
            hi += 1.0
        artist = ax.contourf(lon, lat, field, levels=np.linspace(lo, hi, 17), cmap="coolwarm", extend="both")
        cbar_label = f"temperature at {p} hPa (°C)"

    elif variable == "Vertical motion":
        # Meteorological omega: negative values correspond to ascent.
        field = _masked_level(model, model.last_omega_pa_s, k)
        finite = field.compressed()
        amp = max(0.01, float(np.percentile(np.abs(finite), 98)) if finite.size else 0.01)
        artist = ax.contourf(lon, lat, field, levels=np.linspace(-amp, amp, 17), cmap="RdBu_r", extend="both")
        cbar_label = f"omega at {p} hPa (Pa s⁻¹; negative = ascent)"

    elif variable == "Geopotential height":
        field = _masked_level(model, model.geopotential_height_m(), k)
        finite = field.compressed()
        lo, hi = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) if finite.size else (0, 1)
        artist = ax.contourf(lon, lat, field, levels=np.linspace(lo, hi, 17), cmap="cividis", extend="both")
        cbar_label = f"geopotential height at {p} hPa (m)"

    elif variable == "Surface temperature":
        field = model.boundary.surface_temperature_k - 273.15
        artist = ax.contourf(lon, lat, field, levels=np.arange(-30, 49, 3), cmap="coolwarm", extend="both")
        cbar_label = "prescribed surface temperature (°C)"

    else:
        active_count = model.active.sum(axis=0)
        field = model.boundary.surface_elevation_m
        artist = ax.contourf(lon, lat, field, levels=np.arange(0, 6001, 400), cmap="terrain", extend="max")
        cbar_label = "surface elevation (m)"
        cs = ax.contour(lon, lat, active_count, levels=[8, 10, 12, 14, 16, 18, 19], colors="black", linewidths=0.6)
        ax.clabel(cs, fontsize=7, fmt=lambda x: f"{int(x)} layers")

    cbar = fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.08, shrink=0.82)
    cbar.set_label(cbar_label)

    # Show underground cells at the selected pressure as solid terrain.
    if variable not in {"Surface temperature", "Terrain and active layers"}:
        underground = np.ma.array(np.ones_like(active, dtype=float), mask=active)
        ax.contourf(lon, lat, underground, levels=[0.5, 1.5], colors=["0.40"], alpha=0.62, zorder=6)

    # Terrain contours stay visible on every map. The 3000 and 4500 m lines
    # make the Tibetan Plateau impossible to miss.
    terrain_levels = [1000, 2000, 3000, 4500]
    tc = ax.contour(
        lon,
        lat,
        model.boundary.surface_elevation_m,
        levels=terrain_levels,
        colors=["#6b4d2e"] * len(terrain_levels),
        linewidths=[0.45, 0.65, 0.9, 1.15],
        zorder=7,
    )
    ax.clabel(tc, fontsize=6, fmt="%d m")

    if variable not in {"Surface temperature", "Terrain and active layers"}:
        stride = max(1, int(vector_stride))
        selector = (slice(None, None, stride), slice(None, None, stride))
        uu = np.where(active, model.u[k], np.nan)[selector]
        vv = np.where(active, model.v[k], np.nan)[selector]
        finite_speed = np.sqrt(uu * uu + vv * vv)
        if np.isfinite(finite_speed).any() and float(np.nanmax(finite_speed)) > 1.0e-5:
            ax.quiver(
                lon[selector],
                lat[selector],
                uu,
                vv,
                color="black",
                width=0.0016,
                headwidth=3.3,
                headlength=4.2,
                scale=None,
                zorder=9,
            )

    draw_coastlines(ax, model)
    day = model.time_seconds / 86_400.0
    ax.set_title(f"Atmos20 · {variable} · {p} hPa · day {day:.2f}")
    return fig


def status_markdown(model: AtmosphereModel) -> str:
    s = model.status()
    blocked = ", ".join(str(x) for x in s["tibet_blocked_levels_hpa"])
    return (
        f"**模拟时间：** {s['simulation_days']:.3f} d  \n"
        f"**风速：** mean {s['mean_wind_m_s']:.2f} m/s · max {s['max_wind_m_s']:.2f} m/s  \n"
        f"**海平面气压：** {s['slp_min_hpa']:.1f}–{s['slp_max_hpa']:.1f} hPa  \n"
        f"**青藏高原格点：** {s['tibet_elevation_m']:.0f} m · 地面气压 {s['tibet_surface_pressure_hpa']:.0f} hPa  \n"
        f"**该格点地下等压层：** {blocked} hPa"
    )


def plot_vertical_section(
    model: AtmosphereModel,
    longitude_deg: float = 90.0,
) -> Figure:
    """Latitude-pressure section at one longitude.

    Zonal wind is shaded, temperature is contoured, and pressure levels that
    lie underground are filled gray. A section near 90°E exposes the deep
    Tibetan terrain mask directly.
    """
    ix = int(np.argmin(np.abs(((model.grid.lon_deg - longitude_deg + 180.0) % 360.0) - 180.0)))
    actual_lon = float(model.grid.lon_deg[ix])
    lat = model.grid.lat_deg
    pressure = model.pressure_hpa
    lat2, p2 = np.meshgrid(lat, pressure)
    active = model.active[:, :, ix]
    zonal_wind = np.ma.array(model.u[:, :, ix], mask=~active)
    temperature = np.ma.array(model.temperature_k[:, :, ix] - 273.15, mask=~active)

    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    finite = zonal_wind.compressed()
    amp = max(5.0, float(np.percentile(np.abs(finite), 98)) if finite.size else 5.0)
    artist = ax.contourf(
        lat2,
        p2,
        zonal_wind,
        levels=np.linspace(-amp, amp, 19),
        cmap="RdBu_r",
        extend="both",
    )
    tfinite = temperature.compressed()
    if tfinite.size:
        tmin = 5.0 * np.floor(float(np.percentile(tfinite, 3)) / 5.0)
        tmax = 5.0 * np.ceil(float(np.percentile(tfinite, 97)) / 5.0)
        if tmax > tmin:
            tc = ax.contour(lat2, p2, temperature, levels=np.arange(tmin, tmax + 1, 5), colors="black", linewidths=0.55)
            ax.clabel(tc, fontsize=7, fmt="%d°C")

    underground = np.ma.array(np.ones_like(active, dtype=float), mask=active)
    ax.contourf(lat2, p2, underground, levels=[0.5, 1.5], colors=["0.42"], alpha=0.78)
    ax.set_ylim(1000, 50)
    ax.set_yticks(np.arange(1000, 49, -100))
    ax.set_xlim(lat.min(), lat.max())
    ax.set_xlabel("latitude")
    ax.set_ylabel("pressure (hPa)")
    ax.grid(linewidth=0.35, alpha=0.3)
    cbar = fig.colorbar(artist, ax=ax, orientation="horizontal", pad=0.09, shrink=0.82)
    cbar.set_label("zonal wind u (m s⁻¹; red=eastward, blue=westward)")
    ax.set_title(
        f"Atmos20 vertical section at {actual_lon:.1f}°E · day {model.time_seconds / 86400.0:.2f}\n"
        "temperature contours; gray = pressure levels below terrain"
    )
    return fig
