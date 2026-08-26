"""Objective diagnostics for synoptic-scale thermal fronts.

This module deliberately does not know where a front is expected to be.  It
diagnoses fronts from a two-dimensional temperature-like scalar and the wind
on one model level.  Potential temperature is preferred, although temperature
is equivalent up to a constant factor on a fixed pressure surface.

The broad frontal zone is where the horizontal thermal gradient exceeds an
absolute, physically interpretable threshold.  The thinner ``front_line`` is
the zero crossing of the thermal-front parameter (TFP) inside that zone; this
locates a local maximum of the cross-front thermal gradient.  Cold/warm type is
derived from motion of isotherms between two model times when available.  A
single-time diagnosis falls back to normal thermal advection and is explicitly
reported with lower confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Protocol

import numpy as np


EARTH_RADIUS_M = 6.371e6
HUNDRED_KM_M = 100_000.0
THREE_HOURS_S = 10_800.0


class FrontType(IntEnum):
    """Integer labels stored in :attr:`FrontDiagnostics.front_type`."""

    NONE = 0
    COLD = 1
    WARM = 2
    STATIONARY = 3


@dataclass(frozen=True, slots=True)
class FrontDiagnostics:
    """Objective frontal diagnostics on one horizontal model level.

    Array fields have the same two-dimensional shape as the input thermal
    field.  ``front_type`` uses :class:`FrontType` integer values.  A positive
    ``normal_motion_m_s`` is motion toward warmer air and therefore denotes a
    cold front; a negative value denotes a warm front.
    """

    gradient_x_k_m: np.ndarray
    gradient_y_k_m: np.ndarray
    gradient_magnitude_k_per_100km: np.ndarray
    thermal_front_parameter_k_per_100km2: np.ndarray
    kinematic_frontogenesis_k_per_100km_per_3h: np.ndarray
    front_zone: np.ndarray
    front_line: np.ndarray
    front_type: np.ndarray
    classification_confidence: np.ndarray
    normal_motion_m_s: np.ndarray
    classification_tendency_k_s: np.ndarray
    classification_method: str


class _GridProtocol(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def grad_x(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray: ...

    def grad_y(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class _GradientOperators:
    grad_x: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad_y: Callable[[np.ndarray, np.ndarray], np.ndarray]
    tan_lat_over_radius_m: np.ndarray


def _as_rectilinear_coordinates(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)

    if lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != shape or lon.shape != shape:
            raise ValueError("two-dimensional latitude/longitude must match the field shape")
        lat_1d = lat[:, 0]
        lon_1d = lon[0, :]
        if not np.allclose(lat, lat_1d[:, None], atol=1.0e-10, rtol=0.0):
            raise ValueError("latitude must describe a rectilinear grid")
        lon_error = (lon - lon_1d[None, :] + 180.0) % 360.0 - 180.0
        if not np.allclose(lon_error, 0.0, atol=1.0e-10, rtol=0.0):
            raise ValueError("longitude must describe a rectilinear grid")
        return lat_1d, lon_1d

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("latitude/longitude must both be 1-D or both be 2-D")
    if (lat.size, lon.size) != shape:
        raise ValueError("latitude/longitude sizes must match the field shape")
    return lat, lon


def _coordinate_operators(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    shape: tuple[int, int],
    earth_radius_m: float,
) -> _GradientOperators:
    """Build spherical derivatives with periodic longitude."""

    lat_1d, lon_1d = _as_rectilinear_coordinates(lat_deg, lon_deg, shape)
    if shape[0] < 3 or shape[1] < 3:
        raise ValueError("front diagnosis needs at least a 3 x 3 horizontal grid")
    if not np.all(np.isfinite(lat_1d)) or not np.all(np.isfinite(lon_1d)):
        raise ValueError("latitude/longitude coordinates must be finite")

    lat_rad = np.deg2rad(lat_1d)
    lon_rad = np.deg2rad(lon_1d)
    if np.any(np.abs(np.diff(lat_rad)) < 1.0e-12):
        raise ValueError("latitude coordinates must be distinct")

    # The wrapped east-to-west angular separation is correct at both 0/360
    # and -180/180 seams and retains the sign of coordinate orientation.
    lon_delta = (
        np.roll(lon_rad, -1) - np.roll(lon_rad, 1) + np.pi
    ) % (2.0 * np.pi) - np.pi
    if np.any(np.abs(lon_delta) < 1.0e-12):
        raise ValueError("longitude coordinates must be distinct and span a periodic grid")

    cos_lat = np.cos(lat_rad)
    if np.any(np.abs(cos_lat) < 1.0e-6):
        raise ValueError("longitude derivatives are undefined at an exact geographic pole")
    dx = earth_radius_m * cos_lat[:, None] * lon_delta[None, :]

    def grad_x(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        east = np.roll(field, -1, axis=-1)
        west = np.roll(field, 1, axis=-1)
        east = np.where(np.roll(mask, -1, axis=-1), east, field)
        west = np.where(np.roll(mask, 1, axis=-1), west, field)
        return (east - west) / dx

    def grad_y(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        north = np.empty_like(field)
        south = np.empty_like(field)
        north_mask = np.empty_like(mask)
        south_mask = np.empty_like(mask)
        north[:-1] = field[1:]
        north[-1] = field[-1]
        south[1:] = field[:-1]
        south[0] = field[0]
        north_mask[:-1] = mask[1:]
        north_mask[-1] = mask[-1]
        south_mask[1:] = mask[:-1]
        south_mask[0] = mask[0]
        north = np.where(north_mask, north, field)
        south = np.where(south_mask, south, field)

        delta = np.empty(lat_rad.size, dtype=float)
        delta[1:-1] = lat_rad[2:] - lat_rad[:-2]
        delta[0] = lat_rad[1] - lat_rad[0]
        delta[-1] = lat_rad[-1] - lat_rad[-2]
        return (north - south) / (earth_radius_m * delta[:, None])

    metric = np.broadcast_to(
        (np.tan(lat_rad) / earth_radius_m)[:, None],
        shape,
    )
    return _GradientOperators(
        grad_x=grad_x,
        grad_y=grad_y,
        tan_lat_over_radius_m=metric,
    )


def _grid_operators(grid: _GridProtocol, shape: tuple[int, int]) -> _GradientOperators:
    if tuple(grid.shape) != shape:
        raise ValueError("grid shape must match the thermal field")

    def grad_x(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray(grid.grad_x(field, mask), dtype=float)

    def grad_y(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.asarray(grid.grad_y(field, mask), dtype=float)

    # LatLonGrid exposes these attributes.  A grid implementation that does
    # not expose latitude is assumed to return derivatives in a locally
    # Cartesian basis, for which the metric coefficient is zero.
    lat2d_rad = getattr(grid, "lat2d_rad", None)
    config = getattr(grid, "config", None)
    radius = float(getattr(config, "earth_radius_m", EARTH_RADIUS_M))
    if lat2d_rad is None:
        metric = np.zeros(shape, dtype=float)
    else:
        lat2d_rad = np.asarray(lat2d_rad, dtype=float)
        if lat2d_rad.shape != shape:
            raise ValueError("grid latitude shape must match the thermal field")
        metric = np.tan(lat2d_rad) / radius
    return _GradientOperators(
        grad_x=grad_x,
        grad_y=grad_y,
        tan_lat_over_radius_m=metric,
    )


def _masked_smooth(field: np.ndarray, valid: np.ndarray, passes: int) -> np.ndarray:
    """Apply a weak five-point filter without crossing invalid cells."""

    out = field.copy()
    for _ in range(passes):
        east = np.roll(out, -1, axis=-1)
        west = np.roll(out, 1, axis=-1)
        north = np.empty_like(out)
        south = np.empty_like(out)
        north[:-1] = out[1:]
        north[-1] = out[-1]
        south[1:] = out[:-1]
        south[0] = out[0]

        east_ok = np.roll(valid, -1, axis=-1)
        west_ok = np.roll(valid, 1, axis=-1)
        north_ok = np.empty_like(valid)
        south_ok = np.empty_like(valid)
        north_ok[:-1] = valid[1:]
        north_ok[-1] = valid[-1]
        south_ok[1:] = valid[:-1]
        south_ok[0] = valid[0]

        weighted = 4.0 * out
        weight = np.full(out.shape, 4.0)
        for neighbour, neighbour_ok in (
            (east, east_ok),
            (west, west_ok),
            (north, north_ok),
            (south, south_ok),
        ):
            weighted += np.where(neighbour_ok, neighbour, 0.0)
            weight += neighbour_ok
        out = np.where(valid, weighted / weight, out)
    return out


def _tfp_zero_crossing(
    tfp: np.ndarray,
    front_zone: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return grid cells straddling a meaningful TFP sign reversal."""

    east = np.roll(tfp, -1, axis=-1)
    west = np.roll(tfp, 1, axis=-1)
    north = np.empty_like(tfp)
    south = np.empty_like(tfp)
    north[:-1] = tfp[1:]
    north[-1] = tfp[-1]
    south[1:] = tfp[:-1]
    south[0] = tfp[0]

    east_ok = np.roll(valid, -1, axis=-1)
    west_ok = np.roll(valid, 1, axis=-1)
    north_ok = np.empty_like(valid)
    south_ok = np.empty_like(valid)
    north_ok[:-1] = valid[1:]
    north_ok[-1] = False
    south_ok[1:] = valid[:-1]
    south_ok[0] = False

    neighbours = ((east, east_ok), (west, west_ok), (north, north_ok), (south, south_ok))
    local_abs_max = np.abs(tfp)
    has_positive = tfp > 0.0
    has_negative = tfp < 0.0
    for neighbour, neighbour_ok in neighbours:
        local_abs_max = np.maximum(local_abs_max, np.where(neighbour_ok, np.abs(neighbour), 0.0))
        has_positive |= neighbour_ok & (neighbour > 0.0)
        has_negative |= neighbour_ok & (neighbour < 0.0)

    # TFP has units K/(100 km)^2.  The relative criterion selects the grid
    # point closest to zero while the absolute floor rejects round-off in
    # exactly uniform fields.
    meaningful = local_abs_max > 1.0e-8
    near_zero = np.abs(tfp) <= 0.35 * local_abs_max
    return front_zone & valid & meaningful & near_zero & has_positive & has_negative


def diagnose_fronts(
    thermal_field_k: np.ndarray,
    u_m_s: np.ndarray,
    v_m_s: np.ndarray,
    *,
    grid: _GridProtocol | None = None,
    lat_deg: np.ndarray | None = None,
    lon_deg: np.ndarray | None = None,
    previous_thermal_field_k: np.ndarray | None = None,
    time_delta_seconds: float | None = None,
    valid_mask: np.ndarray | None = None,
    min_gradient_k_per_100km: float = 1.5,
    min_normal_motion_m_s: float = 1.0,
    smoothing_passes: int = 1,
    earth_radius_m: float = EARTH_RADIUS_M,
) -> FrontDiagnostics:
    """Diagnose frontal zones, frontal lines, frontogenesis and front type.

    Parameters
    ----------
    thermal_field_k:
        Temperature or, preferably, potential temperature on one pressure
        surface.  It must be a two-dimensional ``(latitude, longitude)``
        array.
    u_m_s, v_m_s:
        Eastward and northward wind components on the same surface.
    grid or lat_deg/lon_deg:
        Supply either a model grid exposing ``grad_x``/``grad_y`` or
        rectilinear latitude and longitude coordinates.  Longitude is always
        treated as periodic.
    previous_thermal_field_k, time_delta_seconds:
        When both are supplied, isotherm displacement provides an objective
        cold/warm classification.  Without them, normal thermal advection is
        used only as a low-confidence instantaneous proxy.
    min_gradient_k_per_100km:
        Absolute physical threshold for a frontal zone.  The default
        1.5 K/(100 km) deliberately does not promote weak gradients merely
        because they are large relative to a quiet domain.

    Notes
    -----
    The thermal-front parameter is

    ``TFP = -grad(|grad(theta)|) dot grad(theta)/|grad(theta)|``.

    Its zero crossing in a sufficiently strong baroclinic zone marks the
    local maximum thermal gradient.  Kinematic frontogenesis is the local
    horizontal deformation/convergence contribution to
    ``D|grad(theta)|/Dt``; diabatic and vertical terms are not included.
    """

    thermal = np.asarray(thermal_field_k, dtype=float)
    u = np.asarray(u_m_s, dtype=float)
    v = np.asarray(v_m_s, dtype=float)
    if thermal.ndim != 2:
        raise ValueError("thermal_field_k must be a two-dimensional array")
    if u.shape != thermal.shape or v.shape != thermal.shape:
        raise ValueError("wind arrays must match thermal_field_k")
    if min_gradient_k_per_100km <= 0.0:
        raise ValueError("min_gradient_k_per_100km must be positive")
    if min_normal_motion_m_s < 0.0:
        raise ValueError("min_normal_motion_m_s cannot be negative")
    if not isinstance(smoothing_passes, int) or smoothing_passes < 0:
        raise ValueError("smoothing_passes must be a non-negative integer")
    if not np.isfinite(earth_radius_m) or earth_radius_m <= 0.0:
        raise ValueError("earth_radius_m must be positive")

    valid = np.isfinite(thermal) & np.isfinite(u) & np.isfinite(v)
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != thermal.shape:
            raise ValueError("valid_mask must match thermal_field_k")
        valid &= supplied_mask

    if grid is not None:
        if lat_deg is not None or lon_deg is not None:
            raise ValueError("supply grid or latitude/longitude, not both")
        operators = _grid_operators(grid, thermal.shape)
    else:
        if lat_deg is None or lon_deg is None:
            raise ValueError("latitude and longitude are required when grid is omitted")
        operators = _coordinate_operators(
            lat_deg,
            lon_deg,
            thermal.shape,
            float(earth_radius_m),
        )

    safe_thermal = np.where(valid, thermal, 0.0)
    analysed = _masked_smooth(safe_thermal, valid, smoothing_passes)
    grad_x = operators.grad_x(analysed, valid)
    grad_y = operators.grad_y(analysed, valid)
    grad_x = np.where(valid, grad_x, 0.0)
    grad_y = np.where(valid, grad_y, 0.0)
    gradient_si = np.hypot(grad_x, grad_y)
    gradient_scaled = gradient_si * HUNDRED_KM_M

    grad_gradient_x = operators.grad_x(gradient_si, valid)
    grad_gradient_y = operators.grad_y(gradient_si, valid)
    normal_x = np.divide(grad_x, gradient_si, out=np.zeros_like(grad_x), where=gradient_si > 0.0)
    normal_y = np.divide(grad_y, gradient_si, out=np.zeros_like(grad_y), where=gradient_si > 0.0)
    tfp_si = -(grad_gradient_x * normal_x + grad_gradient_y * normal_y)
    tfp_scaled = tfp_si * HUNDRED_KM_M**2

    # Horizontal kinematic frontogenesis: D|grad(theta)|/Dt from the
    # deformation and convergence of the resolved horizontal wind.
    ux = operators.grad_x(u, valid)
    uy = operators.grad_y(u, valid)
    vx = operators.grad_x(v, valid)
    vy = operators.grad_y(v, valid)
    # Covariant derivatives in the local east/north basis.  These metric
    # terms matter at high latitude: without them, even solid-body rotation
    # would spuriously appear to deform and strengthen a front.
    ux_covariant = ux - v * operators.tan_lat_over_radius_m
    vx_covariant = vx + u * operators.tan_lat_over_radius_m
    numerator = (
        grad_x * grad_x * ux_covariant
        + grad_x * grad_y * (uy + vx_covariant)
        + grad_y * grad_y * vy
    )
    frontogenesis_si = -np.divide(
        numerator,
        gradient_si,
        out=np.zeros_like(numerator),
        where=gradient_si > 0.0,
    )
    frontogenesis_scaled = frontogenesis_si * HUNDRED_KM_M * THREE_HOURS_S

    front_zone = valid & (gradient_scaled >= min_gradient_k_per_100km)
    front_line = _tfp_zero_crossing(tfp_scaled, front_zone, valid)

    advective_tendency = -(u * grad_x + v * grad_y)
    wind_normal = u * normal_x + v * normal_y
    if previous_thermal_field_k is not None:
        if time_delta_seconds is None or not np.isfinite(time_delta_seconds) or time_delta_seconds <= 0.0:
            raise ValueError("a positive time_delta_seconds is required with a previous field")
        previous = np.asarray(previous_thermal_field_k, dtype=float)
        if previous.shape != thermal.shape:
            raise ValueError("previous_thermal_field_k must match the current field")
        previous_valid = valid & np.isfinite(previous)
        previous_safe = np.where(previous_valid, previous, 0.0)
        previous_analysed = _masked_smooth(previous_safe, previous_valid, smoothing_passes)
        tendency = (analysed - previous_analysed) / float(time_delta_seconds)
        tendency = np.where(previous_valid, tendency, 0.0)
        classification_valid = front_zone & previous_valid
        method = "two_time_isotherm_motion"
    else:
        if time_delta_seconds is not None:
            raise ValueError("time_delta_seconds is only meaningful with a previous field")
        tendency = advective_tendency
        classification_valid = front_zone
        method = "single_time_normal_advection_proxy"

    # For theta(x,t)=f(x-c*t), -theta_t/|grad(theta)| is frontal motion
    # along the unit normal pointing from cold toward warm air.
    normal_motion = -np.divide(
        tendency,
        gradient_si,
        out=np.zeros_like(tendency),
        where=gradient_si > 0.0,
    )
    normal_motion = np.where(classification_valid, normal_motion, 0.0)

    front_type = np.full(thermal.shape, int(FrontType.NONE), dtype=np.int8)
    moving_cold = classification_valid & (normal_motion >= min_normal_motion_m_s)
    moving_warm = classification_valid & (normal_motion <= -min_normal_motion_m_s)
    stationary = classification_valid & ~moving_cold & ~moving_warm
    front_type[moving_cold] = int(FrontType.COLD)
    front_type[moving_warm] = int(FrontType.WARM)
    front_type[stationary] = int(FrontType.STATIONARY)

    strength = np.clip(
        (np.abs(normal_motion) - min_normal_motion_m_s) / max(10.0 - min_normal_motion_m_s, 1.0),
        0.0,
        1.0,
    )
    confidence = np.zeros_like(thermal)
    if previous_thermal_field_k is not None:
        # Agreement with resolved normal wind is a useful check against a
        # purely diabatic temperature change masquerading as front motion.
        agreement = np.exp(-np.abs(normal_motion - wind_normal) / 10.0)
        confidence[classification_valid] = (
            0.50 + 0.30 * strength[classification_valid]
        ) * (0.75 + 0.25 * agreement[classification_valid])
        confidence[stationary] = 0.45 * agreement[stationary]
    else:
        # A one-time thermal-advection sign is suggestive, not a strict front
        # type.  Keep its reported confidence below 0.45 by construction.
        confidence[classification_valid] = 0.20 + 0.24 * strength[classification_valid]
        confidence[stationary] = 0.10

    for array in (
        grad_x,
        grad_y,
        gradient_scaled,
        tfp_scaled,
        frontogenesis_scaled,
        confidence,
        normal_motion,
        tendency,
    ):
        array[~valid] = 0.0

    return FrontDiagnostics(
        gradient_x_k_m=grad_x,
        gradient_y_k_m=grad_y,
        gradient_magnitude_k_per_100km=gradient_scaled,
        thermal_front_parameter_k_per_100km2=tfp_scaled,
        kinematic_frontogenesis_k_per_100km_per_3h=frontogenesis_scaled,
        front_zone=front_zone,
        front_line=front_line,
        front_type=front_type,
        classification_confidence=confidence,
        normal_motion_m_s=normal_motion,
        classification_tendency_k_s=tendency,
        classification_method=method,
    )


__all__ = ["FrontDiagnostics", "FrontType", "diagnose_fronts"]
