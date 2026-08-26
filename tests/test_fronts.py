from __future__ import annotations

import numpy as np

from atmos20.fronts import EARTH_RADIUS_M, FrontType, diagnose_fronts


def _coordinates(step_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arange(-10.0, 10.0 + step_deg, step_deg)
    lon = np.arange(0.0, 360.0, step_deg)
    return lat, lon


def _translating_pair(
    lat: np.ndarray,
    lon: np.ndarray,
    front_longitude_deg: float,
) -> np.ndarray:
    # A periodic scalar necessarily has an even number of transitions.  The
    # boundary at front_longitude has warm air to its east, while the boundary
    # 180 degrees away has warm air to its west.  Translating both eastward
    # therefore supplies a synthetic cold front and warm front in one field.
    phase = np.deg2rad(lon - front_longitude_deg)
    profile = 280.0 + 8.0 * np.tanh(np.sin(phase) / 0.04)
    return np.broadcast_to(profile, (lat.size, lon.size)).copy()


def test_two_times_distinguish_synthetic_cold_and_warm_fronts() -> None:
    lat, lon = _coordinates()
    dt_seconds = 3.0 * 3600.0
    current = _translating_pair(lat, lon, 180.0)
    previous = _translating_pair(lat, lon, 179.5)
    u = np.full_like(current, 5.0)
    v = np.zeros_like(current)

    result = diagnose_fronts(
        current,
        u,
        v,
        lat_deg=lat,
        lon_deg=lon,
        previous_thermal_field_k=previous,
        time_delta_seconds=dt_seconds,
        min_normal_motion_m_s=0.5,
        smoothing_passes=1,
    )

    equator = int(np.argmin(np.abs(lat)))
    cold_window = np.abs((lon - 180.0 + 180.0) % 360.0 - 180.0) <= 4.0
    warm_window = np.minimum(lon, 360.0 - lon) <= 4.0
    cold_line = result.front_line[equator] & cold_window
    warm_line = result.front_line[equator] & warm_window

    assert cold_line.any()
    assert warm_line.any()
    assert np.all(result.front_type[equator, cold_line] == int(FrontType.COLD))
    assert np.all(result.front_type[equator, warm_line] == int(FrontType.WARM))
    assert np.all(result.normal_motion_m_s[equator, cold_line] > 0.0)
    assert np.all(result.normal_motion_m_s[equator, warm_line] < 0.0)
    assert np.all(result.classification_confidence[equator, cold_line | warm_line] >= 0.5)
    assert result.classification_method == "two_time_isotherm_motion"


def test_uniform_thermal_field_never_reports_a_front() -> None:
    lat, lon = _coordinates(5.0)
    thermal = np.full((lat.size, lon.size), 280.0)
    u = np.full_like(thermal, 30.0)
    v = np.full_like(thermal, -15.0)

    result = diagnose_fronts(
        thermal,
        u,
        v,
        lat_deg=lat,
        lon_deg=lon,
    )

    assert not result.front_zone.any()
    assert not result.front_line.any()
    assert np.all(result.front_type == int(FrontType.NONE))
    assert np.all(result.gradient_magnitude_k_per_100km == 0.0)
    assert np.isfinite(result.thermal_front_parameter_k_per_100km2).all()
    assert np.isfinite(result.kinematic_frontogenesis_k_per_100km_per_3h).all()


def test_longitude_derivative_and_front_line_are_periodic_at_seam() -> None:
    lat, lon = _coordinates(5.0)
    amplitude_k = 20.0
    thermal = 280.0 + amplitude_k * np.sin(np.deg2rad(lon))[None, :]
    thermal = np.broadcast_to(thermal, (lat.size, lon.size)).copy()
    zeros = np.zeros_like(thermal)

    result = diagnose_fronts(
        thermal,
        zeros,
        zeros,
        lat_deg=lat,
        lon_deg=lon,
        min_gradient_k_per_100km=0.20,
        smoothing_passes=0,
    )

    equator = int(np.argmin(np.abs(lat)))
    expected_seam_gradient = amplitude_k / EARTH_RADIUS_M
    assert np.isclose(
        result.gradient_x_k_m[equator, 0],
        expected_seam_gradient,
        rtol=0.01,
    )
    # A non-periodic derivative would create a seam spike instead of the
    # analytic sinusoidal maximum and would fail both assertions.
    assert result.front_line[equator, 0]
    assert result.gradient_magnitude_k_per_100km.max() < 0.35


def test_single_time_classification_is_explicitly_low_confidence() -> None:
    lat, lon = _coordinates()
    thermal = _translating_pair(lat, lon, 180.0)
    u = np.full_like(thermal, 8.0)
    v = np.zeros_like(thermal)

    result = diagnose_fronts(
        thermal,
        u,
        v,
        lat_deg=lat,
        lon_deg=lon,
        min_normal_motion_m_s=0.5,
    )

    assert result.classification_method == "single_time_normal_advection_proxy"
    assert np.max(result.classification_confidence) < 0.45
    assert np.any(result.front_type == int(FrontType.COLD))
    assert np.any(result.front_type == int(FrontType.WARM))


def test_spherical_metric_does_not_frontogenetically_deform_solid_body_rotation() -> None:
    lat = np.arange(-60.0, 60.1, 2.0)
    lon = np.arange(0.0, 360.0, 2.0)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    # Give the scalar gradients both eastward and northward components, so a
    # missing spherical metric term cannot accidentally cancel from the test.
    thermal = (
        280.0
        + 8.0 * np.sin(np.deg2rad(lon2d)) * np.cos(np.deg2rad(lat2d))
        + 0.08 * lat2d
    )
    angular_velocity_s = 1.0e-5
    u = angular_velocity_s * EARTH_RADIUS_M * np.cos(np.deg2rad(lat2d))
    v = np.zeros_like(u)

    result = diagnose_fronts(
        thermal,
        u,
        v,
        lat_deg=lat,
        lon_deg=lon,
        min_gradient_k_per_100km=0.05,
        smoothing_passes=0,
    )

    # Centered finite differences leave a tiny truncation residual, but a
    # rigid rotation has no strain and hence no physical frontogenesis.
    interior = np.abs(lat2d) <= 55.0
    assert np.max(
        np.abs(result.kinematic_frontogenesis_k_per_100km_per_3h[interior])
    ) < 2.0e-4
