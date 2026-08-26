from __future__ import annotations

import numpy as np
from PIL import Image

from atmos20 import AtmosphereModel, HeldSuarezSpec, ModelConfig, configure_held_suarez_circulation
from scripts.prerender_windy import (
    RenderSnapshot,
    RenderSettings,
    WORLD_VIEWPORT,
    _empty_front_diagnostics,
    circulation_wind_belt_statistics,
    load_display_surface,
    save_webp_animation,
)
from windy_app import CONFIG_RESPONSE, ValidationError, validate_render_payload


def test_renderer_uses_one_complete_world_texture() -> None:
    assert RenderSettings().region == "world"
    assert WORLD_VIEWPORT.name == "world"
    assert WORLD_VIEWPORT.lon_min == 0.0
    assert WORLD_VIEWPORT.lon_max == 360.0
    assert WORLD_VIEWPORT.lat_min == -90.0
    assert WORLD_VIEWPORT.lat_max == 90.0
    assert WORLD_VIEWPORT.width == 2 * WORLD_VIEWPORT.height


def test_api_does_not_advertise_a_region_picker() -> None:
    parameters = CONFIG_RESPONSE["parameters"]
    assert isinstance(parameters, dict)
    assert "region" not in parameters
    assert validate_render_payload({}).region == "world"


def test_legacy_regions_normalize_to_world() -> None:
    for legacy in ("east_asia", "east_asia_pacific", "asia", "global", "world"):
        assert validate_render_payload({"region": legacy}).region == "world"

    try:
        validate_render_payload({"region": "mars"})
    except ValidationError:
        pass
    else:
        raise AssertionError("unknown regions must be rejected")


def test_default_payload_selects_global_circulation_sequence() -> None:
    settings = validate_render_payload({})
    assert settings.scenario == "circulation"
    assert settings.level == 900
    assert settings.spinup_hours == 120.0
    assert settings.analysis_hours == 120.0
    assert settings.frames == 48
    assert settings.timestep_seconds == 300.0
    assert settings.equator_to_pole_contrast_k == 60.0
    assert settings.surface_drag_days == 1.0
    assert CONFIG_RESPONSE["parameters"]["frames"]["default"] == settings.frames
    assert CONFIG_RESPONSE["parameters"]["fps"]["default"] == settings.fps
    assert CONFIG_RESPONSE["parameters"]["particles"]["default"] == settings.particles


def test_baroclinic_rejects_levels_below_its_900_hpa_model_bottom() -> None:
    for level in (1000, 950):
        try:
            validate_render_payload(
                {
                    "scenario": "baroclinic",
                    "level": level,
                    "spinupHours": 240.0,
                }
            )
        except ValidationError:
            pass
        else:
            raise AssertionError(f"baroclinic mode accepted unavailable {level} hPa")


def test_baroclinic_duration_and_circulation_duration_are_separately_validated() -> None:
    for payload in (
        {"scenario": "baroclinic", "level": 850, "spinupHours": 48.0},
        {"scenario": "baroclinic", "level": 850, "spinupHours": 246.0},
        {"scenario": "circulation", "spinupHours": 24.0},
    ):
        try:
            validate_render_payload(payload)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"invalid duration accepted: {payload}")

    circulation = validate_render_payload(
        {"scenario": "circulation", "spinupHours": 120.0}
    )
    assert circulation.scenario == "circulation"
    assert circulation.spinup_hours == 120.0


def test_webp_encoder_keeps_model_timestamps_for_identical_frames(tmp_path) -> None:
    output = tmp_path / "timeline.webp"
    frames = [Image.new("RGB", (16, 8), (20, 40, 60)) for _ in range(4)]

    save_webp_animation(frames, output, fps=4, quality=80)

    with Image.open(output) as encoded:
        assert encoded.n_frames == len(frames)


def test_display_relief_uses_packaged_one_degree_etopo() -> None:
    surface = load_display_surface()
    assert surface.lon_deg is not None and surface.lon_deg.size == 360
    assert surface.lat_deg is not None and surface.lat_deg.size == 156
    assert surface.elevation_m.shape == (156, 360)
    assert float(np.max(surface.elevation_m)) > 5_000.0


def test_api_passes_terrain_and_surface_heating_controls() -> None:
    settings = validate_render_payload(
        {
            "tibetScale": 1.4,
            "landHeatingScale": 0.75,
            "oceanCurrentScale": 1.2,
        }
    )
    assert settings.tibet_scale == 1.4
    assert settings.land_heating_scale == 0.75
    assert settings.ocean_current_scale == 1.2


def test_quality_gate_rejects_equatorial_high_wavenumber_string() -> None:
    model = AtmosphereModel(
        ModelConfig(
            dlon_deg=5.0,
            dlat_deg=5.0,
            pressure_step_hpa=100,
            dt_seconds=300.0,
        )
    )
    configure_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            initial_temperature_noise_k=0.0,
            polar_sponge_width_rows=0.0,
        ),
    )
    latitude = model.grid.lat2d_deg
    base = np.where(
        np.abs(latitude) <= 20.0,
        -2.0,
        np.where(np.abs(latitude) <= 60.0, 5.0, -2.0),
    )
    wave = 3.0 * np.sin(np.deg2rad(13.0 * model.grid.lon2d_deg))
    u = base + np.where(np.abs(latitude) <= 10.0, wave, 0.0)
    zeros = np.zeros(model.grid.shape)
    snapshot = RenderSnapshot(
        time_hours=120.0,
        u_m_s=u,
        v_m_s=zeros,
        temperature_k=np.full(model.grid.shape, 264.0),
        potential_temperature_k=np.full(model.grid.shape, 264.0),
        pressure_anomaly_hpa=zeros,
        omega_pa_s=zeros,
        geopotential_height_m=zeros,
        active=np.ones(model.grid.shape, dtype=bool),
        fronts=_empty_front_diagnostics(model.grid.shape),
    )
    k = model.level_index(900)
    model.u[k] = u
    stats = circulation_wind_belt_statistics(model, [snapshot, snapshot], 900)

    assert stats["gate"]["window_mean_passed"]
    assert not stats["gate"]["tropical_high_wavenumber_passed"]
    assert not stats["gate"]["passed"]
