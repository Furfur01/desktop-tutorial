from __future__ import annotations

import numpy as np
import pytest

from atmos20.config import ModelConfig
from atmos20.grid import LatLonGrid


def test_meridional_gradient_uses_second_order_one_sided_edges() -> None:
    grid = LatLonGrid.build(
        ModelConfig(dlon_deg=30.0, dlat_deg=5.0, lat_limit_deg=10.0)
    )
    y_m = grid.lat2d_deg * np.pi / 180.0 * grid.config.earth_radius_m
    field = 7.0 + 2.5e-6 * y_m + 3.0e-13 * y_m**2
    expected = 2.5e-6 + 6.0e-13 * y_m

    np.testing.assert_allclose(
        grid.grad_y(field),
        expected,
        rtol=0.0,
        atol=1.0e-18,
    )


def test_tvd_advection_preserves_constant_field() -> None:
    grid = LatLonGrid.build(ModelConfig(dlon_deg=10.0, dlat_deg=10.0))
    shape = (2, *grid.shape)
    field = np.full(shape, 273.0)
    mask = np.ones(shape, dtype=bool)
    u = np.full(shape, 18.0)
    v = np.full(shape, -4.0)
    assert np.all(grid.tvd_advection(field, u, v, mask) == 0.0)


def test_tvd_reconstruction_is_more_accurate_for_smooth_zonal_wave() -> None:
    grid = LatLonGrid.build(ModelConfig(dlon_deg=5.0, dlat_deg=5.0))
    field = np.sin(np.deg2rad(grid.lon2d_deg))[None, ...]
    mask = np.ones_like(field, dtype=bool)
    u = np.full_like(field, 20.0)
    v = np.zeros_like(field)
    exact = u * np.cos(np.deg2rad(grid.lon2d_deg))[None, ...] / (
        grid.config.earth_radius_m * np.maximum(grid.cos_lat, 0.15)
    )
    upwind_error = np.sqrt(np.mean((grid.upwind_advection(field, u, v, mask) - exact) ** 2))
    tvd_error = np.sqrt(np.mean((grid.tvd_advection(field, u, v, mask) - exact) ** 2))
    assert tvd_error < 0.65 * upwind_error


def test_unknown_advection_scheme_is_rejected() -> None:
    grid = LatLonGrid.build(ModelConfig(dlon_deg=10.0, dlat_deg=10.0))
    field = np.zeros((1, *grid.shape))
    mask = np.ones_like(field, dtype=bool)
    with pytest.raises(ValueError, match="Unknown advection scheme"):
        grid.advection(field, field, field, mask, "invented")
