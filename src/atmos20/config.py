from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for the idealized 20-level atmospheric model.

    The default vertical grid is 1000, 950, ..., 50 hPa: twenty fixed
    pressure levels separated by 50 hPa. The default 2.5-degree horizontal
    grid uses area-aggregated ETOPO 2022 relief and remains interactive.
    """

    dlon_deg: float = 2.5
    dlat_deg: float = 2.5
    lat_limit_deg: float = 77.5
    dt_seconds: float = 300.0

    pressure_bottom_hpa: int = 1000
    pressure_top_hpa: int = 50
    pressure_step_hpa: int = 50

    earth_radius_m: float = 6.371e6
    rotation_rate_s: float = 7.2921159e-5
    gravity_m_s2: float = 9.80665
    gas_constant_dry_air: float = 287.05
    heat_capacity_cp: float = 1004.0

    horizontal_diffusion_rate_s: float = 1.5e-5
    vertical_mixing_rate_s: float = 7.0e-6
    surface_drag_land_s: float = 1.0 / 86_400.0
    surface_drag_ocean_s: float = 1.0 / (2.0 * 86_400.0)
    terrain_blocking_rate_s: float = 1.0 / (4.0 * 3600.0)
    mass_damping_rate_s: float = 1.0 / (12.0 * 86_400.0)
    surface_pressure_coupling: float = 0.02

    # Lower-boundary controls exposed by the UI.
    tibet_height_scale: float = 1.0
    land_heating_scale: float = 1.0
    ocean_current_scale: float = 1.0
    # +1: boreal summer, 0: equinox, -1: boreal winter.
    seasonal_phase: float = 1.0
    surface_lapse_rate_k_m: float = 0.0055
    thermal_low_pressure_pa_per_k: float = 60.0

    random_seed: int = 7

    @property
    def pressure_levels_hpa(self) -> np.ndarray:
        return np.arange(
            self.pressure_bottom_hpa,
            self.pressure_top_hpa - 1,
            -self.pressure_step_hpa,
            dtype=float,
        )

    @property
    def n_levels(self) -> int:
        return int(self.pressure_levels_hpa.size)

    @property
    def kappa(self) -> float:
        return self.gas_constant_dry_air / self.heat_capacity_cp

    def with_updates(self, **kwargs: float) -> "ModelConfig":
        return replace(self, **kwargs)
