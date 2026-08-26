"""Held--Suarez-style dry global-circulation experiment.

This module provides an axisymmetric, physically documented forcing for the
compact pressure-level dynamical core.  It follows the benchmark described by
Held and Suarez (1994): Newtonian relaxation toward an analytic radiative-
equilibrium temperature and Rayleigh friction confined to the lowest part of
the atmosphere.  The canonical configuration remains a flat aquaplanet.  A
separate orographic configuration retains the model's ETOPO lower boundary,
uses local sigma for near-surface physics, and adds no prescribed wind or
pressure centre.

The optional seasonal displacement moves the thermal equator in latitude but
does not introduce any longitude dependence.  Tiny grid-scale random thermal
noise merely breaks the exact zonal symmetry so that resolved eddies can grow;
it does not prescribe a circulation feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import BoundaryFields
from .grid import LatLonGrid
from .model import AtmosphereModel


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class HeldSuarezSpec:
    """Parameters of the dry Held--Suarez circulation benchmark.

    Defaults are the values in Held & Suarez (1994), equations (1)--(4),
    except for the explicitly labelled numerical controls at the end.
    """

    reference_surface_pressure_pa: float = 100_000.0
    equatorial_temperature_k: float = 315.0
    equator_to_pole_contrast_k: float = 60.0
    vertical_stability_k: float = 10.0
    stratospheric_temperature_k: float = 200.0
    free_atmosphere_relaxation_days: float = 40.0
    surface_relaxation_days: float = 4.0
    boundary_layer_sigma: float = 0.7
    surface_drag_days: float = 1.0

    # A zero displacement is the canonical perpetual-equinox experiment.
    # Positive values represent a northward-displaced boreal-summer thermal
    # equator; the forcing remains axisymmetric at every displacement.
    seasonal_heat_equator_deg: float = 0.0

    # Numerical symmetry-breaking and truncated-pole controls.  The thermal
    # perturbation is independent at every grid point and has zero global
    # area-weighted mean on every pressure level.  The sponge is confined to
    # the final two latitude intervals of this pole-avoiding grid.
    # canonical benchmark commonly starts from a horizontally uniform,
    # resting 264 K atmosphere, so the circulation is spun up gradually by
    # the analytic relaxation rather than placed in the initial condition.
    initial_temperature_k: float = 264.0
    initial_temperature_noise_k: float = 0.10
    random_seed: int = 7
    polar_sponge_width_rows: float = 1.5
    polar_sponge_e_folding_hours: float = 12.0


@dataclass(frozen=True, slots=True)
class HeldSuarezForcing:
    """Analytic equilibrium and height-dependent forcing coefficients."""

    equilibrium_temperature_k: np.ndarray
    newtonian_relaxation_rate_s: np.ndarray
    rayleigh_drag_rate_s: np.ndarray


@dataclass(frozen=True, slots=True)
class OrographicCirculationSpec:
    """Earth-like lower-boundary physics layered onto Held--Suarez forcing.

    The analytic Held--Suarez equilibrium still supplies the planetary-scale
    equator-to-pole thermal contrast.  ETOPO elevation changes the local
    pressure surface, removes underground pressure levels, and supplies slope
    lift/form drag.  Near the local surface, the equilibrium temperature is
    blended toward the prescribed land/ocean boundary temperature; no wind or
    pressure centre is prescribed by this configuration.
    """

    surface_temperature_coupling: float = 0.55
    land_drag_multiplier: float = 1.35
    orographic_lift_scale: float = 0.35
    terrain_blocking_e_folding_hours: float = 8.0
    vertical_mixing_rate_s: float = 2.0e-6
    mass_damping_days: float = 8.0
    surface_pressure_coupling: float = 0.15
    initial_temperature_noise_k: float = 0.02
    initial_noise_smoothing_passes: int = 2


@dataclass(frozen=True, slots=True)
class WindBeltStatistics:
    """Area-weighted zonal-wind diagnostics on one pressure level."""

    pressure_hpa: float
    tropical_mean_m_s: float
    northern_midlatitude_mean_m_s: float
    southern_midlatitude_mean_m_s: float
    northern_polar_mean_m_s: float
    southern_polar_mean_m_s: float
    tropical_easterly_fraction: float
    northern_midlatitude_westerly_fraction: float
    southern_midlatitude_westerly_fraction: float
    northern_polar_easterly_fraction: float
    southern_polar_easterly_fraction: float


def _validate_spec(spec: HeldSuarezSpec) -> None:
    positive = {
        "reference_surface_pressure_pa": spec.reference_surface_pressure_pa,
        "equatorial_temperature_k": spec.equatorial_temperature_k,
        "equator_to_pole_contrast_k": spec.equator_to_pole_contrast_k,
        "vertical_stability_k": spec.vertical_stability_k,
        "stratospheric_temperature_k": spec.stratospheric_temperature_k,
        "free_atmosphere_relaxation_days": spec.free_atmosphere_relaxation_days,
        "surface_relaxation_days": spec.surface_relaxation_days,
        "surface_drag_days": spec.surface_drag_days,
        "initial_temperature_k": spec.initial_temperature_k,
        "polar_sponge_e_folding_hours": spec.polar_sponge_e_folding_hours,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < spec.boundary_layer_sigma < 1.0:
        raise ValueError("boundary_layer_sigma must lie between zero and one")
    if abs(spec.seasonal_heat_equator_deg) > 30.0:
        raise ValueError("seasonal_heat_equator_deg must lie within +/-30 degrees")
    if spec.initial_temperature_noise_k < 0.0:
        raise ValueError("initial_temperature_noise_k cannot be negative")
    if spec.polar_sponge_width_rows < 0.0:
        raise ValueError("polar_sponge_width_rows cannot be negative")


def _validate_orographic_spec(spec: OrographicCirculationSpec) -> None:
    positive = {
        "land_drag_multiplier": spec.land_drag_multiplier,
        "terrain_blocking_e_folding_hours": spec.terrain_blocking_e_folding_hours,
        "mass_damping_days": spec.mass_damping_days,
        "surface_pressure_coupling": spec.surface_pressure_coupling,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    non_negative = {
        "surface_temperature_coupling": spec.surface_temperature_coupling,
        "orographic_lift_scale": spec.orographic_lift_scale,
        "vertical_mixing_rate_s": spec.vertical_mixing_rate_s,
        "initial_temperature_noise_k": spec.initial_temperature_noise_k,
    }
    for name, value in non_negative.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} cannot be negative")
    if spec.surface_temperature_coupling > 1.0:
        raise ValueError("surface_temperature_coupling cannot exceed one")
    if not 0.0 < spec.surface_pressure_coupling <= 1.0:
        raise ValueError("surface_pressure_coupling must lie in (0, 1]")
    if (
        not isinstance(spec.initial_noise_smoothing_passes, int)
        or spec.initial_noise_smoothing_passes < 0
    ):
        raise ValueError("initial_noise_smoothing_passes must be a non-negative integer")


def build_held_suarez_boundary(
    grid: LatLonGrid,
    spec: HeldSuarezSpec | None = None,
) -> BoundaryFields:
    """Return the flat, constant-pressure aquaplanet lower boundary."""

    spec = spec or HeldSuarezSpec()
    _validate_spec(spec)
    zeros = np.zeros(grid.shape, dtype=float)
    land = np.zeros(grid.shape, dtype=bool)
    base_pressure = np.full(
        grid.shape,
        spec.reference_surface_pressure_pa,
        dtype=float,
    )
    shifted_latitude = np.deg2rad(
        grid.lat2d_deg - spec.seasonal_heat_equator_deg
    )
    surface_temperature = (
        spec.equatorial_temperature_k
        - spec.equator_to_pole_contrast_k * np.sin(shifted_latitude) ** 2
    )
    return BoundaryFields(
        land_mask=land,
        land_fraction=zeros.copy(),
        surface_temperature_k=surface_temperature,
        surface_elevation_m=zeros.copy(),
        base_surface_pressure_pa=base_pressure,
        seasonal_surface_pressure_anomaly_pa=zeros.copy(),
        terrain_slope_x=zeros.copy(),
        terrain_slope_y=zeros.copy(),
        terrain_slope=zeros.copy(),
    )


def build_held_suarez_forcing(
    model: AtmosphereModel,
    spec: HeldSuarezSpec | None = None,
    *,
    local_surface_pressure_pa: np.ndarray | None = None,
    surface_temperature_k: np.ndarray | None = None,
    surface_temperature_coupling: float = 0.0,
) -> HeldSuarezForcing:
    """Evaluate the standard Held--Suarez thermal and frictional forcing."""

    spec = spec or HeldSuarezSpec()
    _validate_spec(spec)
    if local_surface_pressure_pa is None:
        surface_pressure = np.full(
            model.grid.shape,
            spec.reference_surface_pressure_pa,
            dtype=float,
        )
    else:
        surface_pressure = np.asarray(local_surface_pressure_pa, dtype=float)
        if surface_pressure.shape != model.grid.shape:
            raise ValueError("local_surface_pressure_pa must match the horizontal grid")
        if not np.isfinite(surface_pressure).all() or np.any(surface_pressure <= 0.0):
            raise ValueError("local_surface_pressure_pa must be finite and positive")
    if not 0.0 <= surface_temperature_coupling <= 1.0:
        raise ValueError("surface_temperature_coupling must lie between zero and one")

    sigma = model.pressure_pa[:, None, None] / surface_pressure[None, :, :]
    # Values greater than one lie below terrain and are masked by the model.
    # Clipping them avoids extending the analytic pressure profile underground.
    sigma = np.clip(sigma, 1.0e-6, 1.0)
    shifted_latitude = np.deg2rad(
        model.grid.lat2d_deg - spec.seasonal_heat_equator_deg
    )[None, :, :]
    sin2 = np.sin(shifted_latitude) ** 2
    cos2 = np.cos(shifted_latitude) ** 2

    # Held & Suarez (1994), equation (1).  The expression in brackets is
    # equilibrium potential temperature; multiplication by sigma**kappa
    # converts it to temperature.  The 200 K cap represents the stratosphere.
    equilibrium = (
        spec.equatorial_temperature_k
        - spec.equator_to_pole_contrast_k * sin2
        - spec.vertical_stability_k * np.log(sigma) * cos2
    ) * sigma**model.config.kappa
    equilibrium = np.maximum(equilibrium, spec.stratospheric_temperature_k)
    equilibrium = np.broadcast_to(equilibrium, model.u.shape).copy()

    boundary_fraction = np.maximum(
        0.0,
        (sigma - spec.boundary_layer_sigma)
        / (1.0 - spec.boundary_layer_sigma),
    )
    if surface_temperature_k is not None and surface_temperature_coupling > 0.0:
        surface_temperature = np.asarray(surface_temperature_k, dtype=float)
        if surface_temperature.shape != model.grid.shape:
            raise ValueError("surface_temperature_k must match the horizontal grid")
        if not np.isfinite(surface_temperature).all():
            raise ValueError("surface_temperature_k must be finite")
        analytic_surface = (
            spec.equatorial_temperature_k
            - spec.equator_to_pole_contrast_k * sin2[0]
        )
        surface_correction = surface_temperature - analytic_surface
        equilibrium += (
            surface_temperature_coupling
            * boundary_fraction
            * surface_correction[None, :, :]
        )
        equilibrium = np.maximum(equilibrium, spec.stratospheric_temperature_k)
    free_rate = 1.0 / (
        spec.free_atmosphere_relaxation_days * SECONDS_PER_DAY
    )
    surface_rate = 1.0 / (
        spec.surface_relaxation_days * SECONDS_PER_DAY
    )
    # Held & Suarez (1994), equation (2): relaxation is fastest in the
    # tropical lower troposphere and approaches 40 days aloft.
    newtonian = free_rate + (
        surface_rate - free_rate
    ) * boundary_fraction * np.cos(shifted_latitude) ** 4
    newtonian = np.broadcast_to(newtonian, model.u.shape).copy()

    # Equations (3)--(4): one-day Rayleigh friction at sigma=1, linearly
    # decreasing to exactly zero above sigma_b.
    drag = boundary_fraction / (spec.surface_drag_days * SECONDS_PER_DAY)
    drag = np.broadcast_to(drag, model.u.shape).copy()
    return HeldSuarezForcing(equilibrium, newtonian, drag)


def _smooth_horizontal_noise(noise: np.ndarray, passes: int) -> np.ndarray:
    """Apply a compact periodic 1-2-1-like filter without a SciPy dependency."""

    filtered = np.asarray(noise, dtype=float).copy()
    for _ in range(passes):
        north = np.empty_like(filtered)
        south = np.empty_like(filtered)
        north[:, :-1, :] = filtered[:, 1:, :]
        north[:, -1, :] = filtered[:, -1, :]
        south[:, 1:, :] = filtered[:, :-1, :]
        south[:, 0, :] = filtered[:, 0, :]
        filtered = (
            4.0 * filtered
            + np.roll(filtered, -1, axis=2)
            + np.roll(filtered, 1, axis=2)
            + north
            + south
        ) / 8.0
    return filtered


def configure_orographic_held_suarez_circulation(
    model: AtmosphereModel,
    spec: HeldSuarezSpec | None = None,
    orographic_spec: OrographicCirculationSpec | None = None,
) -> HeldSuarezForcing:
    """Configure an idealized global circulation with a physical ETOPO boundary.

    Unlike :func:`configure_held_suarez_circulation`, this experiment does not
    replace the model boundary with a flat aquaplanet.  It retains the ETOPO
    surface already constructed by :class:`AtmosphereModel`, evaluates the
    Held--Suarez sigma profiles against local surface pressure, and enables the
    core's terrain mask, slope lift, form drag, and land/ocean thermal contrast.
    """

    spec = spec or HeldSuarezSpec(random_seed=model.config.random_seed)
    orographic_spec = orographic_spec or OrographicCirculationSpec()
    _validate_spec(spec)
    _validate_orographic_spec(orographic_spec)
    if not np.any(model.boundary.surface_elevation_m > 0.0):
        raise ValueError("orographic circulation requires a non-flat terrain boundary")

    # The generic seasonal boundary includes an optional balanced warm-start
    # pressure target for the legacy monsoon demo.  This experiment starts from
    # zero pressure anomaly so every pressure feature develops prognostically.
    model.boundary.seasonal_surface_pressure_anomaly_pa.fill(0.0)

    # Balance the lower-boundary geopotential against the isothermal resting
    # reference state.  For T=T0, g z + R T0 ln(ps0/p) is then independent of
    # terrain height on every pressure surface, so adding a mountain does not
    # launch a spurious gravity-wave shock at the first timestep.
    reference_temperature = spec.initial_temperature_k
    balanced_surface_pressure = spec.reference_surface_pressure_pa * np.exp(
        -model.config.gravity_m_s2 * model.boundary.surface_elevation_m
        / (model.config.gas_constant_dry_air * reference_temperature)
    )
    model.boundary.base_surface_pressure_pa[...] = balanced_surface_pressure
    pressure_3d = model.pressure_pa[:, None, None]
    model.active = pressure_3d <= balanced_surface_pressure[None, :, :]
    model.lowest_index = np.argmax(model.active, axis=0)
    level_ids = np.arange(model.nz)[:, None, None]
    model.lowest_layer = model.active & (
        level_ids == model.lowest_index[None, :, :]
    )

    drag_rate = 1.0 / (spec.surface_drag_days * SECONDS_PER_DAY)
    model.config = model.config.with_updates(
        vertical_mixing_rate_s=orographic_spec.vertical_mixing_rate_s,
        surface_drag_land_s=drag_rate * orographic_spec.land_drag_multiplier,
        surface_drag_ocean_s=drag_rate,
        terrain_blocking_rate_s=(
            1.0
            / (orographic_spec.terrain_blocking_e_folding_hours * 3600.0)
        ),
        orographic_lift_scale=orographic_spec.orographic_lift_scale,
        mass_damping_rate_s=(
            1.0 / (orographic_spec.mass_damping_days * SECONDS_PER_DAY)
        ),
        surface_pressure_coupling=orographic_spec.surface_pressure_coupling,
        newtonian_relaxation_rate_scale=1.0,
    )
    forcing = build_held_suarez_forcing(
        model,
        spec,
        local_surface_pressure_pa=model.boundary.base_surface_pressure_pa,
        surface_temperature_k=model.boundary.surface_temperature_k,
        surface_temperature_coupling=(
            orographic_spec.surface_temperature_coupling
        ),
    )
    model.radiative_equilibrium_k = forcing.equilibrium_temperature_k.copy()
    model.radiative_rate_s = forcing.newtonian_relaxation_rate_s.copy()
    model.surface_weight = np.where(
        model.active,
        forcing.rayleigh_drag_rate_s
        / max(drag_rate, np.finfo(float).tiny),
        0.0,
    )

    weights = model.active * model.grid.area_weight[None, :, :]
    denominator = np.sum(weights, axis=2, keepdims=True)
    zonal_equilibrium = np.divide(
        np.sum(model.radiative_equilibrium_k * weights, axis=2, keepdims=True),
        denominator,
        out=np.zeros((*model.u.shape[:2], 1), dtype=float),
        where=denominator > 0.0,
    )
    initial = np.full_like(
        model.radiative_equilibrium_k,
        spec.initial_temperature_k,
    )
    rng = np.random.default_rng(spec.random_seed)
    noise = rng.normal(
        0.0,
        orographic_spec.initial_temperature_noise_k,
        model.u.shape,
    )
    noise = _smooth_horizontal_noise(
        noise,
        orographic_spec.initial_noise_smoothing_passes,
    )
    noise_mean = np.divide(
        np.sum(noise * weights, axis=(1, 2), keepdims=True),
        np.sum(weights, axis=(1, 2), keepdims=True),
        out=np.zeros((model.nz, 1, 1), dtype=float),
        where=np.sum(weights, axis=(1, 2), keepdims=True) > 0.0,
    )
    initial += noise - noise_mean
    model.temperature_k = np.where(
        model.active,
        initial,
        model.radiative_equilibrium_k,
    )
    model.u.fill(0.0)
    model.v.fill(0.0)
    model.surface_pressure_anomaly_pa.fill(0.0)

    model._meridional_sponge_rate_s = None
    model._meridional_sponge_reference_u = None
    model._meridional_sponge_reference_v = None
    model._meridional_sponge_reference_temperature_k = None
    model._meridional_sponge_reference_pressure_pa = None
    model.meridional_sponge_start_latitude_deg = None
    model.meridional_sponge_e_folding_seconds = None
    sponge_width_deg = spec.polar_sponge_width_rows * model.config.dlat_deg
    sponge_start = model.config.lat_limit_deg - sponge_width_deg
    if sponge_width_deg > 0.0 and sponge_start > 0.0:
        model.configure_meridional_sponge(
            start_latitude_deg=sponge_start,
            e_folding_seconds=spec.polar_sponge_e_folding_hours * 3600.0,
            reference_u=np.zeros_like(model.u),
            reference_v=np.zeros_like(model.v),
            reference_temperature_k=np.broadcast_to(
                zonal_equilibrium,
                model.u.shape,
            ),
            reference_pressure_anomaly_pa=np.zeros(model.grid.shape),
        )

    model.time_seconds = 0.0
    model.last_omega_pa_s = np.zeros_like(model.u)
    model.last_geopotential_m2_s2 = model.compute_geopotential()
    return forcing


def configure_held_suarez_circulation(
    model: AtmosphereModel,
    spec: HeldSuarezSpec | None = None,
) -> HeldSuarezForcing:
    """Turn ``model`` into a dry, flat-aquaplanet circulation experiment.

    The model starts from rest at the analytic radiative equilibrium plus
    tiny, globally distributed temperature noise.  Subsequent wind belts and
    eddies therefore arise from the resolved momentum and thermodynamic
    equations under the axisymmetric forcing.
    """

    spec = spec or HeldSuarezSpec(random_seed=model.config.random_seed)
    _validate_spec(spec)
    if np.any(model.pressure_pa > spec.reference_surface_pressure_pa + 1.0e-9):
        raise ValueError("model pressure levels extend below the reference surface")

    # Disable non-benchmark physics.  Horizontal diffusion remains the
    # model's explicit grid-scale closure; all other subgrid forcing below is
    # exactly the Held--Suarez formulation.
    drag_rate = 1.0 / (spec.surface_drag_days * SECONDS_PER_DAY)
    model.config = model.config.with_updates(
        vertical_mixing_rate_s=0.0,
        surface_drag_land_s=drag_rate,
        surface_drag_ocean_s=drag_rate,
        terrain_blocking_rate_s=0.0,
        orographic_lift_scale=0.0,
        mass_damping_rate_s=0.0,
        surface_pressure_coupling=1.0,
        newtonian_relaxation_rate_scale=1.0,
    )
    model.boundary = build_held_suarez_boundary(model.grid, spec)

    model.active = np.ones_like(model.u, dtype=bool)
    model.lowest_index = np.zeros(model.grid.shape, dtype=int)
    model.lowest_layer = np.zeros_like(model.active)
    model.lowest_layer[0] = True

    forcing = build_held_suarez_forcing(model, spec)
    model.radiative_equilibrium_k = forcing.equilibrium_temperature_k.copy()
    model.radiative_rate_s = forcing.newtonian_relaxation_rate_s.copy()

    # The core multiplies its 2-D land/ocean drag coefficient by this vertical
    # weight.  Setting it to the canonical sigma ramp produces the exact
    # Held--Suarez Rayleigh profile in ``AtmosphereModel._rhs``.
    model.surface_weight = (
        forcing.rayleigh_drag_rate_s
        / max(drag_rate, np.finfo(float).tiny)
    )

    model.u.fill(0.0)
    model.v.fill(0.0)
    rng = np.random.default_rng(spec.random_seed)
    noise = rng.normal(0.0, spec.initial_temperature_noise_k, model.u.shape)
    weights = model.grid.area_weight[None, :, :]
    layer_mean = np.sum(noise * weights, axis=(1, 2), keepdims=True) / np.sum(
        weights,
        axis=(1, 2),
        keepdims=True,
    )
    model.temperature_k = (
        np.full_like(model.radiative_equilibrium_k, spec.initial_temperature_k)
        + noise
        - layer_mean
    )
    model.surface_pressure_anomaly_pa.fill(0.0)

    # Clear any experiment installed previously, then optionally absorb only
    # waves that reach the last one or two rows beside the truncated pole.
    model._meridional_sponge_rate_s = None
    model._meridional_sponge_reference_u = None
    model._meridional_sponge_reference_v = None
    model._meridional_sponge_reference_temperature_k = None
    model._meridional_sponge_reference_pressure_pa = None
    model.meridional_sponge_start_latitude_deg = None
    model.meridional_sponge_e_folding_seconds = None
    sponge_width_deg = spec.polar_sponge_width_rows * model.config.dlat_deg
    sponge_start = model.config.lat_limit_deg - sponge_width_deg
    if sponge_width_deg > 0.0 and sponge_start > 0.0:
        model.configure_meridional_sponge(
            start_latitude_deg=sponge_start,
            e_folding_seconds=(
                spec.polar_sponge_e_folding_hours * 3600.0
            ),
            reference_u=np.zeros_like(model.u),
            reference_v=np.zeros_like(model.v),
            reference_temperature_k=model.radiative_equilibrium_k,
            reference_pressure_anomaly_pa=np.zeros(model.grid.shape),
        )

    model.time_seconds = 0.0
    model.last_omega_pa_s = np.zeros_like(model.u)
    model.last_geopotential_m2_s2 = model.compute_geopotential()
    return forcing


def wind_belt_statistics(
    model: AtmosphereModel,
    pressure_hpa: float = 850.0,
) -> WindBeltStatistics:
    """Summarize the resolved 850-hPa wind belts without longitude cherry-picking."""

    k = model.level_index(pressure_hpa)
    wind = model.u[k]
    latitude = model.grid.lat2d_deg
    weights = model.grid.area_weight

    def mean_and_fraction(
        low: float,
        high: float,
        *,
        easterly: bool,
    ) -> tuple[float, float]:
        mask = (latitude >= low) & (latitude <= high)
        selected_weights = np.where(mask, weights, 0.0)
        denominator = float(np.sum(selected_weights))
        mean = float(np.sum(wind * selected_weights) / denominator)
        desired = wind < 0.0 if easterly else wind > 0.0
        fraction = float(
            np.sum(selected_weights * desired) / denominator
        )
        return mean, fraction

    tropical, tropical_easterly = mean_and_fraction(-20.0, 20.0, easterly=True)
    north_mid, north_mid_west = mean_and_fraction(30.0, 60.0, easterly=False)
    south_mid, south_mid_west = mean_and_fraction(-60.0, -30.0, easterly=False)
    north_polar, north_polar_east = mean_and_fraction(65.0, 80.0, easterly=True)
    south_polar, south_polar_east = mean_and_fraction(-80.0, -65.0, easterly=True)
    return WindBeltStatistics(
        pressure_hpa=float(model.pressure_hpa[k]),
        tropical_mean_m_s=tropical,
        northern_midlatitude_mean_m_s=north_mid,
        southern_midlatitude_mean_m_s=south_mid,
        northern_polar_mean_m_s=north_polar,
        southern_polar_mean_m_s=south_polar,
        tropical_easterly_fraction=tropical_easterly,
        northern_midlatitude_westerly_fraction=north_mid_west,
        southern_midlatitude_westerly_fraction=south_mid_west,
        northern_polar_easterly_fraction=north_polar_east,
        southern_polar_easterly_fraction=south_polar_east,
    )
