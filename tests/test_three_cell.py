from __future__ import annotations

import numpy as np

from atmos20 import (
    HeldSuarezSpec,
    ModelConfig,
    ThreeCellAtmosphereModel,
    ThreeCellClosure,
    ThreeCellSpec,
    configure_held_suarez_circulation,
    configure_orographic_held_suarez_circulation,
    OrographicCirculationSpec,
)


def _coarse_model(spec: ThreeCellSpec | None = None) -> ThreeCellAtmosphereModel:
    model = ThreeCellAtmosphereModel(
        ModelConfig(
            dlon_deg=60.0,
            dlat_deg=10.0,
            lat_limit_deg=80.0,
            pressure_step_hpa=100,
            dt_seconds=600.0,
        )
    )
    configure_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
    )
    model.set_three_cell_closure(spec)
    return model


def test_target_overturning_has_zero_column_mass_transport() -> None:
    model = _coarse_model(ThreeCellSpec(spinup_e_folding_days=0.25))
    closure = model.three_cell_closure
    assert closure is not None
    target = closure.target_v(model, time_seconds=30.0 * 86_400.0)

    # The core uses equally thick pressure layers in its continuity integral.
    np.testing.assert_allclose(np.sum(target, axis=0), 0.0, atol=2.0e-15)
    np.testing.assert_allclose(np.ptp(target, axis=2), 0.0, atol=0.0)

    rng = np.random.default_rng(4)
    tendency = closure.v_tendency(
        model,
        rng.normal(size=model.v.shape),
        time_seconds=30.0 * 86_400.0,
    )
    np.testing.assert_allclose(np.sum(tendency, axis=0), 0.0, atol=2.0e-18)


def test_overturning_amplitude_grows_smoothly_from_rest() -> None:
    model = _coarse_model(ThreeCellSpec(spinup_e_folding_days=2.0))
    closure = model.three_cell_closure
    assert closure is not None
    at_start = closure.target_v(model, 0.0)
    after_one_day = closure.target_v(model, 86_400.0)
    after_ten_days = closure.target_v(model, 10.0 * 86_400.0)

    assert np.count_nonzero(at_start) == 0
    assert np.max(np.abs(after_one_day)) > 0.0
    assert np.max(np.abs(after_ten_days)) > np.max(np.abs(after_one_day))


def test_cell_boundaries_are_c1_continuous_and_move_with_season() -> None:
    closure = ThreeCellClosure(ThreeCellSpec(thermal_equator_deg=15.0))
    # 0/30/60/90 in the transformed coordinate map to these physical
    # boundaries on the north and south sides of the displaced heat equator.
    boundaries = np.array([-90.0, -55.0, -20.0, 15.0, 40.0, 65.0, 90.0])
    np.testing.assert_allclose(closure.latitude_mode(boundaries), 0.0, atol=2.0e-30)

    epsilon = 1.0e-4
    left = closure.latitude_mode(boundaries[1:-1] - epsilon)
    right = closure.latitude_mode(boundaries[1:-1] + epsilon)
    np.testing.assert_allclose(left, right, atol=2.0e-9)
    # A sin^2 envelope approaches every boundary quadratically (zero slope).
    assert np.max(np.abs(left)) < 2.0e-9
    assert np.max(np.abs(right)) < 2.0e-9


def test_zero_temperature_contrast_switches_closure_off() -> None:
    spec = ThreeCellSpec(equator_to_pole_contrast_k=0.0)
    model = _coarse_model(spec)
    closure = model.three_cell_closure
    assert closure is not None
    assert closure.ramp_fraction(100.0 * 86_400.0) == 0.0
    assert np.count_nonzero(closure.target_v(model, 100.0 * 86_400.0)) == 0
    assert np.count_nonzero(closure.v_tendency(model, model.v, 100.0 * 86_400.0)) == 0


def test_closure_adds_only_meridional_momentum_tendency() -> None:
    model = _coarse_model(
        ThreeCellSpec(spinup_e_folding_days=0.25, closure_e_folding_hours=1.0)
    )
    model.time_seconds = 2.0 * 86_400.0
    closure = model.three_cell_closure
    assert closure is not None

    model.three_cell_closure = None
    without = model._rhs(
        model.u,
        model.v,
        model.temperature_k,
        model.surface_pressure_anomaly_pa,
    )
    model.three_cell_closure = closure
    with_closure = model._rhs(
        model.u,
        model.v,
        model.temperature_k,
        model.surface_pressure_anomaly_pa,
    )

    np.testing.assert_allclose(with_closure.u, without.u)
    np.testing.assert_allclose(with_closure.temperature, without.temperature)
    np.testing.assert_allclose(
        with_closure.surface_pressure_anomaly,
        without.surface_pressure_anomaly,
    )
    assert np.max(np.abs(with_closure.v - without.v)) > 0.0


def test_coriolis_and_drag_generate_three_900_hpa_wind_belts() -> None:
    model = _coarse_model(
        ThreeCellSpec(
            spinup_e_folding_days=0.25,
            closure_e_folding_hours=1.0,
            reference_lower_branch_speed_m_s=2.0,
            cell_strengths=(1.0, 0.8, 0.7),
        )
    )
    model.advance_hours(120.0)

    k = model.level_index(900.0)
    zonal_mean = np.mean(model.u[k], axis=1)
    latitude = model.grid.lat_deg

    def belt_mean(low: float, high: float) -> float:
        selected = (latitude >= low) & (latitude <= high)
        return float(np.mean(zonal_mean[selected]))

    # Neither u nor ps is prescribed.  These signs arise because the lower
    # branches cross planetary vorticity and the core applies Rayleigh drag.
    assert belt_mean(10.0, 20.0) < -0.5
    assert belt_mean(-20.0, -10.0) < -0.5
    assert belt_mean(40.0, 50.0) > 0.5
    assert belt_mean(-50.0, -40.0) > 0.5
    assert belt_mean(70.0, 80.0) < -0.2
    assert belt_mean(-80.0, -70.0) < -0.2
    assert np.max(np.abs(model.u)) < 20.0


def test_terrain_closure_is_zonal_and_mass_neutral_across_active_air() -> None:
    model = ThreeCellAtmosphereModel(
        ModelConfig(
            dlon_deg=10.0,
            dlat_deg=10.0,
            pressure_step_hpa=100,
            dt_seconds=300.0,
            seasonal_phase=0.0,
        )
    )
    configure_orographic_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
        OrographicCirculationSpec(initial_temperature_noise_k=0.0),
    )
    model.set_three_cell_closure(
        ThreeCellSpec(spinup_e_folding_days=0.25)
    )
    closure = model.three_cell_closure
    assert closure is not None
    target = closure.target_v(model, time_seconds=30.0 * 86_400.0)

    np.testing.assert_allclose(
        np.sum(np.where(model.active, target, 0.0), axis=(0, 2)),
        0.0,
        atol=5.0e-14,
    )
    for level in range(model.nz):
        for latitude in range(model.ny):
            values = target[level, latitude, model.active[level, latitude]]
            if values.size:
                np.testing.assert_allclose(np.ptp(values), 0.0, atol=2.0e-15)
