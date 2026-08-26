from __future__ import annotations

import numpy as np

from atmos20 import AtmosphereModel, ModelConfig


def _small_config(**updates: float) -> ModelConfig:
    return ModelConfig(
        dlon_deg=30.0,
        dlat_deg=30.0,
        dt_seconds=240.0,
        **updates,
    )


def test_momentum_tendency_includes_pressure_vertical_advection() -> None:
    model = AtmosphereModel(
        _small_config(
            horizontal_diffusion_rate_s=0.0,
            vertical_mixing_rate_s=0.0,
        )
    )
    pressure = model.pressure_pa[:, None, None]
    u = np.broadcast_to(1.0e-3 * pressure, model.u.shape).copy()
    v = np.broadcast_to(-5.0e-4 * pressure, model.v.shape).copy()
    temperature = model.temperature_k.copy()
    ps_anom = np.zeros_like(model.surface_pressure_anomaly_pa)

    def tendency_with_omega(omega_pa_s: float):
        omega = np.full_like(model.u, omega_pa_s)
        omega[~model.active] = 0.0

        def diagnose(
            _u: np.ndarray,
            _v: np.ndarray,
            _temperature: np.ndarray,
            _ps_anom: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            return omega, np.zeros_like(model.surface_pressure_anomaly_pa)

        model._diagnose_omega_and_mass_tendency = diagnose  # type: ignore[method-assign]
        return model._rhs(u, v, temperature, ps_anom)

    still = tendency_with_omega(0.0)
    sinking = tendency_with_omega(0.2)

    # Use a full-column ocean cell so the analytic pressure derivatives are
    # exactly du/dp=1e-3 and dv/dp=-5e-4 in SI units.
    full_columns = np.all(model.active, axis=0) & (~model.boundary.land_mask)
    iy, ix = np.argwhere(full_columns)[0]
    k = model.level_index(500)
    assert np.isclose(sinking.u[k, iy, ix] - still.u[k, iy, ix], -2.0e-4)
    assert np.isclose(sinking.v[k, iy, ix] - still.v[k, iy, ix], 1.0e-4)


def test_surface_pressure_coupling_can_use_full_column_mass_tendency() -> None:
    weak = AtmosphereModel(
        _small_config(
            surface_pressure_coupling=0.02,
            mass_damping_rate_s=0.0,
            horizontal_diffusion_rate_s=0.0,
        )
    )
    full = AtmosphereModel(
        _small_config(
            surface_pressure_coupling=1.0,
            mass_damping_rate_s=0.0,
            horizontal_diffusion_rate_s=0.0,
        )
    )
    zonal_wave = np.sin(np.deg2rad(weak.grid.lon2d_deg))[None, :, :]
    u = np.broadcast_to(zonal_wave, weak.u.shape).copy()
    v = np.zeros_like(u)
    u[~weak.active] = 0.0

    _, weak_dps = weak._diagnose_omega_and_mass_tendency(
        u,
        v,
        weak.temperature_k,
        np.zeros_like(weak.surface_pressure_anomaly_pa),
    )
    _, full_dps = full._diagnose_omega_and_mass_tendency(
        u,
        v,
        full.temperature_k,
        np.zeros_like(full.surface_pressure_anomaly_pa),
    )

    assert np.max(np.abs(weak_dps)) > 0.0
    assert np.allclose(full_dps, 50.0 * weak_dps)


def test_surface_pressure_anomaly_limit_is_configurable() -> None:
    default = AtmosphereModel(_small_config())
    wider = AtmosphereModel(
        _small_config(surface_pressure_anomaly_limit_pa=6_000.0)
    )
    for model in (default, wider):
        model.surface_pressure_anomaly_pa.fill(0.0)
        model.surface_pressure_anomaly_pa[0, 0] = 4_000.0
        model.surface_pressure_anomaly_pa[0, 1] = -4_000.0
        model._apply_constraints()

    assert default.surface_pressure_anomaly_pa[0, 0] == 2_500.0
    assert default.surface_pressure_anomaly_pa[0, 1] == -2_500.0
    assert wider.surface_pressure_anomaly_pa[0, 0] == 4_000.0
    assert wider.surface_pressure_anomaly_pa[0, 1] == -4_000.0


def test_newtonian_relaxation_can_be_lengthened_or_disabled() -> None:
    default = AtmosphereModel(_small_config())
    slower = AtmosphereModel(
        _small_config(newtonian_relaxation_rate_scale=0.25)
    )
    disabled = AtmosphereModel(
        _small_config(newtonian_relaxation_rate_scale=0.0)
    )

    assert np.allclose(slower.radiative_rate_s, 0.25 * default.radiative_rate_s)
    assert np.count_nonzero(disabled.radiative_rate_s) == 0


def test_subgrid_diffusion_and_vertical_mixing_can_be_disabled() -> None:
    model = AtmosphereModel(
        _small_config(
            horizontal_diffusion_rate_s=0.0,
            vertical_mixing_rate_s=0.0,
        )
    )
    probe = np.arange(model.u.size, dtype=float).reshape(model.u.shape)

    assert model.config.horizontal_diffusion_rate_s == 0.0
    assert np.count_nonzero(model._vertical_mixing(probe)) == 0
