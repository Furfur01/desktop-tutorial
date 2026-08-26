from __future__ import annotations

"""Deterministic dry baroclinic-wave initial conditions.

The balanced state follows the pressure-coordinate test of Jablonowski and
Williamson (2006), doi:10.1256/qj.06.12.  It is an analytic, hydrostatic and
gradient-wind-balanced solution of the dry, inviscid primitive equations.
The optional perturbation is their localized 1 m/s Gaussian perturbation to
the zonal wind.  It triggers baroclinic instability; it does not prescribe a
low-pressure centre, a temperature front, or a cyclone-shaped vortex.

``configure_baroclinic_wave_model`` installs the matching idealized lower
boundary and disables the compact model's diabatic and dissipative forcing.
This is important: copying the wind and temperature onto the project's real
terrain would no longer be the analytic balanced test case.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .boundary import BoundaryFields
from .config import ModelConfig
from .grid import LatLonGrid
from .model import AtmosphereModel


Hemisphere = Literal["north", "south"]


@dataclass(frozen=True, slots=True)
class DryBaroclinicWaveSpec:
    """Parameters for the dry Jablonowski--Williamson wave experiment."""

    reference_surface_pressure_pa: float = 100_000.0
    jet_max_m_s: float = 35.0
    jet_sigma: float = 0.252
    tropopause_sigma: float = 0.2
    reference_surface_temperature_k: float = 288.0
    lapse_rate_k_m: float = 0.005
    stratospheric_temperature_coefficient_k: float = 4.8e5

    perturbation_hemisphere: Hemisphere = "north"
    perturbation_amplitude_m_s: float = 1.0
    perturbation_longitude_deg: float = 20.0
    perturbation_latitude_deg: float = 40.0
    perturbation_radius_earth_fraction: float = 0.1


@dataclass(frozen=True, slots=True)
class DryBaroclinicInitialState:
    """Complete initial state plus diagnostics of its two components."""

    u: np.ndarray
    v: np.ndarray
    temperature_k: np.ndarray
    surface_pressure_pa: np.ndarray
    geopotential_m2_s2: np.ndarray
    surface_geopotential_m2_s2: np.ndarray
    surface_temperature_k: np.ndarray
    balanced_u: np.ndarray
    zonal_wind_perturbation: np.ndarray


def _validate_spec(spec: DryBaroclinicWaveSpec) -> None:
    if spec.perturbation_hemisphere not in ("north", "south"):
        raise ValueError("perturbation_hemisphere must be 'north' or 'south'")
    if spec.reference_surface_pressure_pa <= 0.0:
        raise ValueError("reference_surface_pressure_pa must be positive")
    if spec.jet_max_m_s <= 0.0:
        raise ValueError("jet_max_m_s must be positive")
    if not 0.0 < spec.jet_sigma < 1.0:
        raise ValueError("jet_sigma must lie between zero and one")
    if not 0.0 < spec.tropopause_sigma < 1.0:
        raise ValueError("tropopause_sigma must lie between zero and one")
    if spec.reference_surface_temperature_k <= 0.0:
        raise ValueError("reference_surface_temperature_k must be positive")
    if spec.lapse_rate_k_m <= 0.0:
        raise ValueError("lapse_rate_k_m must be positive")
    if spec.perturbation_amplitude_m_s < 0.0:
        raise ValueError("perturbation_amplitude_m_s cannot be negative")
    if not 0.0 < abs(spec.perturbation_latitude_deg) < 90.0:
        raise ValueError("perturbation_latitude_deg must lie between 0 and 90")
    if spec.perturbation_radius_earth_fraction <= 0.0:
        raise ValueError("perturbation_radius_earth_fraction must be positive")


def _horizontal_structure(latitude_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude factors in the analytic gradient-wind solution."""

    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    first = -2.0 * sin_lat**6 * (cos_lat**2 + 1.0 / 3.0) + 10.0 / 63.0
    second = (
        8.0 / 5.0 * cos_lat**3 * (sin_lat**2 + 2.0 / 3.0)
        - np.pi / 4.0
    )
    return first, second


def _mean_temperature(
    sigma: np.ndarray,
    config: ModelConfig,
    spec: DryBaroclinicWaveSpec,
) -> np.ndarray:
    exponent = (
        config.gas_constant_dry_air
        * spec.lapse_rate_k_m
        / config.gravity_m_s2
    )
    tropospheric = spec.reference_surface_temperature_k * sigma**exponent
    stratospheric = spec.stratospheric_temperature_coefficient_k * np.maximum(
        spec.tropopause_sigma - sigma,
        0.0,
    ) ** 5
    return tropospheric + stratospheric


def _stratospheric_integral(
    sigma: np.ndarray,
    tropopause_sigma: float,
) -> np.ndarray:
    """Integral of ``(sigma_t - sigma)^5 d(log(sigma))``.

    It supplies the hydrostatic geopotential associated with the empirical
    lower-stratospheric temperature term.
    """

    sigma_t = tropopause_sigma

    def primitive(value: np.ndarray) -> np.ndarray:
        return (
            sigma_t**5 * np.log(value)
            - 5.0 * sigma_t**4 * value
            + 5.0 * sigma_t**3 * value**2
            - 10.0 / 3.0 * sigma_t**2 * value**3
            + 5.0 / 4.0 * sigma_t * value**4
            - value**5 / 5.0
        )

    return np.where(
        sigma < sigma_t,
        primitive(sigma) - primitive(np.asarray(sigma_t)),
        0.0,
    )


def _mean_geopotential(
    sigma: np.ndarray,
    config: ModelConfig,
    spec: DryBaroclinicWaveSpec,
) -> np.ndarray:
    exponent = (
        config.gas_constant_dry_air
        * spec.lapse_rate_k_m
        / config.gravity_m_s2
    )
    tropospheric = (
        spec.reference_surface_temperature_k
        * config.gravity_m_s2
        / spec.lapse_rate_k_m
        * (1.0 - sigma**exponent)
    )
    stratospheric = (
        -config.gas_constant_dry_air
        * spec.stratospheric_temperature_coefficient_k
        * _stratospheric_integral(sigma, spec.tropopause_sigma)
    )
    return tropospheric + stratospheric


def _balanced_fields(
    sigma: np.ndarray,
    latitude_rad: np.ndarray,
    config: ModelConfig,
    spec: DryBaroclinicWaveSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return balanced zonal wind, temperature, and geopotential."""

    eta_v = (sigma - spec.jet_sigma) * np.pi / 2.0
    cos_eta_v = np.cos(eta_v)
    if np.any(cos_eta_v <= 0.0):
        raise ValueError("pressure range lies outside the analytic test-case domain")

    vertical_jet = spec.jet_max_m_s * cos_eta_v**1.5
    first, second = _horizontal_structure(latitude_rad)
    sin_two_lat_sq = np.sin(2.0 * latitude_rad) ** 2
    balanced_u = vertical_jet[:, None, None] * sin_two_lat_sq[None, :, :]

    mean_t = _mean_temperature(sigma, config, spec)
    temperature_departure = (
        3.0
        / 4.0
        * sigma[:, None, None]
        * np.pi
        * spec.jet_max_m_s
        / config.gas_constant_dry_air
        * np.sin(eta_v)[:, None, None]
        * np.sqrt(cos_eta_v)[:, None, None]
        * (
            first[None, :, :] * (2.0 * vertical_jet)[:, None, None]
            + second[None, :, :]
            * config.earth_radius_m
            * config.rotation_rate_s
        )
    )
    temperature = mean_t[:, None, None] + temperature_departure

    mean_phi = _mean_geopotential(sigma, config, spec)
    geopotential_departure = vertical_jet[:, None, None] * (
        first[None, :, :] * vertical_jet[:, None, None]
        + second[None, :, :]
        * config.earth_radius_m
        * config.rotation_rate_s
    )
    geopotential = mean_phi[:, None, None] + geopotential_departure
    return balanced_u, temperature, geopotential


def build_baroclinic_wave_initial_state(
    config: ModelConfig | None = None,
    *,
    grid: LatLonGrid | None = None,
    pressure_pa: np.ndarray | None = None,
    spec: DryBaroclinicWaveSpec | None = None,
    add_perturbation: bool = True,
) -> DryBaroclinicInitialState:
    """Build a balanced dry atmosphere and an optional wave trigger.

    The basic state contains symmetric midlatitude jets in both hemispheres.
    ``spec.perturbation_hemisphere`` selects which jet receives the localized
    perturbation.  With ``add_perturbation=False`` the returned state is the
    analytic steady solution.
    """

    config = config or ModelConfig()
    spec = spec or DryBaroclinicWaveSpec()
    _validate_spec(spec)
    grid = grid or LatLonGrid.build(config)
    pressure = np.asarray(
        config.pressure_levels_hpa * 100.0 if pressure_pa is None else pressure_pa,
        dtype=float,
    )
    if pressure.ndim != 1 or pressure.size == 0:
        raise ValueError("pressure_pa must be a non-empty one-dimensional array")
    if np.any(pressure <= 0.0) or np.any(
        pressure > spec.reference_surface_pressure_pa * (1.0 + 1.0e-12)
    ):
        raise ValueError("all model pressures must lie between zero and surface pressure")

    sigma = pressure / spec.reference_surface_pressure_pa
    balanced_u, temperature, geopotential = _balanced_fields(
        sigma,
        grid.lat2d_rad,
        config,
        spec,
    )

    perturbation_2d = np.zeros(grid.shape, dtype=float)
    if add_perturbation and spec.perturbation_amplitude_m_s > 0.0:
        centre_latitude = abs(spec.perturbation_latitude_deg)
        if spec.perturbation_hemisphere == "south":
            centre_latitude = -centre_latitude
        centre_lat_rad = np.deg2rad(centre_latitude)
        centre_lon_rad = np.deg2rad(spec.perturbation_longitude_deg)
        lon_rad = np.deg2rad(grid.lon2d_deg)
        great_circle_cosine = (
            np.sin(centre_lat_rad) * np.sin(grid.lat2d_rad)
            + np.cos(centre_lat_rad)
            * np.cos(grid.lat2d_rad)
            * np.cos(lon_rad - centre_lon_rad)
        )
        distance_m = config.earth_radius_m * np.arccos(
            np.clip(great_circle_cosine, -1.0, 1.0)
        )
        radius_m = (
            config.earth_radius_m * spec.perturbation_radius_earth_fraction
        )
        perturbation_2d = spec.perturbation_amplitude_m_s * np.exp(
            -(distance_m / radius_m) ** 2
        )

    perturbation = np.broadcast_to(
        perturbation_2d[None, :, :],
        balanced_u.shape,
    ).copy()
    u = balanced_u + perturbation
    v = np.zeros_like(u)

    surface_sigma = np.asarray([1.0])
    _, surface_temperature_3d, surface_geopotential_3d = _balanced_fields(
        surface_sigma,
        grid.lat2d_rad,
        config,
        spec,
    )
    surface_pressure = np.full(
        grid.shape,
        spec.reference_surface_pressure_pa,
        dtype=float,
    )
    return DryBaroclinicInitialState(
        u=u,
        v=v,
        temperature_k=temperature,
        surface_pressure_pa=surface_pressure,
        geopotential_m2_s2=geopotential,
        surface_geopotential_m2_s2=surface_geopotential_3d[0],
        surface_temperature_k=surface_temperature_3d[0],
        balanced_u=balanced_u,
        zonal_wind_perturbation=perturbation,
    )


def configure_baroclinic_wave_model(
    model: AtmosphereModel,
    *,
    spec: DryBaroclinicWaveSpec | None = None,
    add_perturbation: bool = True,
    boundary_sponge_start_deg: float | None = 62.5,
    boundary_sponge_timescale_hours: float = 3.0,
) -> DryBaroclinicInitialState:
    """Convert ``model`` into the matching dry, adiabatic wave experiment.

    The function deliberately replaces the real-terrain lower boundary with
    the analytic test-case surface geopotential and constant 1000 hPa surface
    pressure.  It also removes radiation, drag, mixing, and mass relaxation;
    retaining those processes would invalidate the published steady state.
    Surface-pressure evolution remains active through column divergence.
    """

    spec = spec or DryBaroclinicWaveSpec()
    state = build_baroclinic_wave_initial_state(
        model.config,
        grid=model.grid,
        pressure_pa=model.pressure_pa,
        spec=spec,
        add_perturbation=add_perturbation,
    )

    model.config = model.config.with_updates(
        horizontal_diffusion_rate_s=0.0,
        vertical_mixing_rate_s=0.0,
        surface_drag_land_s=0.0,
        surface_drag_ocean_s=0.0,
        terrain_blocking_rate_s=0.0,
        orographic_lift_scale=0.0,
        mass_damping_rate_s=0.0,
        surface_pressure_coupling=1.0,
    )
    model.grid.config = model.config

    elevation = state.surface_geopotential_m2_s2 / model.config.gravity_m_s2
    slope_x = model.grid.grad_x(elevation)
    slope_y = model.grid.grad_y(elevation)
    slope = np.hypot(slope_x, slope_y)
    zeros = np.zeros(model.grid.shape, dtype=float)
    model.boundary = BoundaryFields(
        land_mask=np.zeros(model.grid.shape, dtype=bool),
        land_fraction=zeros.copy(),
        surface_temperature_k=state.surface_temperature_k.copy(),
        surface_elevation_m=elevation.copy(),
        base_surface_pressure_pa=state.surface_pressure_pa.copy(),
        seasonal_surface_pressure_anomaly_pa=zeros.copy(),
        terrain_slope_x=slope_x,
        terrain_slope_y=slope_y,
        terrain_slope=slope,
    )

    pressure_3d = model.pressure_pa[:, None, None]
    model.active = pressure_3d <= state.surface_pressure_pa[None, :, :]
    model.lowest_index = np.argmax(model.active, axis=0)
    level_ids = np.arange(model.nz)[:, None, None]
    model.lowest_layer = model.active & (
        level_ids == model.lowest_index[None, :, :]
    )
    model.surface_weight = model._build_surface_weight()

    model.u = np.where(model.active, state.u, 0.0)
    model.v = np.where(model.active, state.v, 0.0)
    model.temperature_k = state.temperature_k.copy()
    model.surface_pressure_anomaly_pa = np.zeros(model.grid.shape, dtype=float)
    model.radiative_equilibrium_k = state.temperature_k.copy()
    model.radiative_rate_s = np.zeros_like(state.temperature_k)
    model.time_seconds = 0.0
    model.last_omega_pa_s = np.zeros_like(model.u)
    model.last_geopotential_m2_s2 = model.compute_geopotential()
    if boundary_sponge_start_deg is not None:
        model.configure_meridional_sponge(
            start_latitude_deg=boundary_sponge_start_deg,
            e_folding_seconds=boundary_sponge_timescale_hours * 3600.0,
            reference_u=state.balanced_u,
            reference_v=np.zeros_like(state.v),
            reference_temperature_k=state.temperature_k,
            reference_pressure_anomaly_pa=np.zeros(model.grid.shape, dtype=float),
        )
    return state


__all__ = [
    "DryBaroclinicInitialState",
    "DryBaroclinicWaveSpec",
    "build_baroclinic_wave_initial_state",
    "configure_baroclinic_wave_model",
]
