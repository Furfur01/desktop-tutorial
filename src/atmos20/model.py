from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .boundary import BoundaryFields, build_boundary
from .config import ModelConfig
from .grid import LatLonGrid


@dataclass(slots=True)
class Tendencies:
    u: np.ndarray
    v: np.ndarray
    temperature: np.ndarray
    surface_pressure_anomaly: np.ndarray
    omega: np.ndarray
    geopotential: np.ndarray


class AtmosphereModel:
    """Idealized hydrostatic, pressure-level atmosphere with terrain masks.

    The model is intentionally compact enough for interactive use. It evolves
    horizontal wind, temperature, and a column-mass/surface-pressure anomaly on
    twenty pressure levels (1000 to 50 hPa at 50 hPa spacing). Hydrostatic
    geopotential, pressure-coordinate vertical velocity, Newtonian thermal
    forcing, vertical mixing, and terrain blocking are diagnosed each step.

    It is a mechanism model, not a numerical-weather-prediction system.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        boundary: BoundaryFields | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.grid = LatLonGrid.build(self.config)
        self.boundary = boundary or build_boundary(self.config, self.grid)

        self.pressure_hpa = self.config.pressure_levels_hpa
        self.pressure_pa = self.pressure_hpa * 100.0
        self.dp_pa = float(self.config.pressure_step_hpa * 100.0)
        self.nz = self.pressure_hpa.size
        self.ny, self.nx = self.grid.shape

        p3 = self.pressure_pa[:, None, None]
        # A level exists only where its pressure is lower than local surface
        # pressure. Over the Tibetan Plateau, most lower-tropospheric levels
        # are therefore solid terrain rather than air.
        self.active = p3 <= self.boundary.base_surface_pressure_pa[None, :, :]
        self.lowest_index = np.argmax(self.active, axis=0)
        level_ids = np.arange(self.nz)[:, None, None]
        self.lowest_layer = self.active & (level_ids == self.lowest_index[None, :, :])

        self.surface_weight = self._build_surface_weight()
        self.radiative_equilibrium_k = self._build_radiative_equilibrium()
        self.radiative_rate_s = self._build_radiative_rate()

        self.u = np.zeros((self.nz, self.ny, self.nx), dtype=np.float64)
        self.v = np.zeros_like(self.u)
        self.temperature_k = self._initial_temperature()
        self.surface_pressure_anomaly_pa = (
            self.boundary.seasonal_surface_pressure_anomaly_pa.copy()
        )

        # Optional absorbing boundary used by idealized global experiments on
        # this truncated latitude-longitude grid.  Ordinary circulation runs
        # leave it disabled.  References are installed explicitly by the
        # experiment so the sponge removes reflected grid noise rather than
        # prescribing a weather system.
        self._meridional_sponge_rate_s: np.ndarray | None = None
        self._meridional_sponge_reference_u: np.ndarray | None = None
        self._meridional_sponge_reference_v: np.ndarray | None = None
        self._meridional_sponge_reference_temperature_k: np.ndarray | None = None
        self._meridional_sponge_reference_pressure_pa: np.ndarray | None = None
        self.meridional_sponge_start_latitude_deg: float | None = None
        self.meridional_sponge_e_folding_seconds: float | None = None

        self.time_seconds = 0.0
        self.last_omega_pa_s = np.zeros_like(self.u)
        self.last_geopotential_m2_s2 = self.compute_geopotential()

    def configure_meridional_sponge(
        self,
        *,
        start_latitude_deg: float,
        e_folding_seconds: float,
        reference_u: np.ndarray,
        reference_v: np.ndarray,
        reference_temperature_k: np.ndarray,
        reference_pressure_anomaly_pa: np.ndarray,
    ) -> None:
        """Install a smooth absorbing boundary near the truncated polar rows.

        The compact model does not include the singular latitude-longitude
        poles: its meridional walls sit at ``+-lat_limit_deg``.  A sine-squared
        relaxation layer prevents gravity-wave and two-grid reflections there.
        The rate is exactly zero equatorward of ``start_latitude_deg`` and
        reaches one e-fold per ``e_folding_seconds`` at the outermost row.
        """

        limit = float(self.config.lat_limit_deg)
        start = float(start_latitude_deg)
        timescale = float(e_folding_seconds)
        if not 0.0 <= start < limit:
            raise ValueError("sponge start latitude must lie inside the model boundary")
        if not np.isfinite(timescale) or timescale <= 0.0:
            raise ValueError("sponge e-folding time must be positive")

        expected_3d = self.u.shape
        expected_2d = self.grid.shape
        for name, value, shape in (
            ("reference_u", reference_u, expected_3d),
            ("reference_v", reference_v, expected_3d),
            ("reference_temperature_k", reference_temperature_k, expected_3d),
            ("reference_pressure_anomaly_pa", reference_pressure_anomaly_pa, expected_2d),
        ):
            if np.shape(value) != shape:
                raise ValueError(f"{name} must have shape {shape}")

        fraction = np.clip(
            (np.abs(self.grid.lat2d_deg) - start) / (limit - start),
            0.0,
            1.0,
        )
        self._meridional_sponge_rate_s = (
            np.sin(0.5 * np.pi * fraction) ** 2 / timescale
        )
        self._meridional_sponge_reference_u = np.asarray(reference_u, dtype=float).copy()
        self._meridional_sponge_reference_v = np.asarray(reference_v, dtype=float).copy()
        self._meridional_sponge_reference_temperature_k = np.asarray(
            reference_temperature_k, dtype=float
        ).copy()
        self._meridional_sponge_reference_pressure_pa = np.asarray(
            reference_pressure_anomaly_pa, dtype=float
        ).copy()
        self.meridional_sponge_start_latitude_deg = start
        self.meridional_sponge_e_folding_seconds = timescale

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_surface_weight(self) -> np.ndarray:
        sigma = self.pressure_pa[:, None, None] / self.boundary.base_surface_pressure_pa[None, :, :]
        # Only the lowest roughly 250 hPa strongly feel the surface.
        weight = np.exp(-((1.0 - sigma) / 0.19) ** 2)
        return np.where(self.active, np.clip(weight, 0.0, 1.0), 0.0)

    def _build_radiative_equilibrium(self) -> np.ndarray:
        p = self.pressure_pa[:, None, None]
        z_std = 8000.0 * np.log(101_325.0 / p)
        dz_above_surface = np.maximum(
            0.0,
            z_std - self.boundary.surface_elevation_m[None, :, :],
        )
        lapse_profile = (
            self.boundary.surface_temperature_k[None, :, :]
            - 0.0063 * dz_above_surface
        )
        # Simple lower stratosphere: cap the cooling and permit a weak warming
        # with height above about 11 km.
        stratosphere = 216.0 + 0.0010 * np.maximum(z_std - 11_000.0, 0.0)
        teq = np.maximum(lapse_profile, stratosphere)
        teq = np.clip(teq, 190.0, 322.0)
        return np.where(self.active, teq, teq)

    def _build_radiative_rate(self) -> np.ndarray:
        pfrac = self.pressure_pa[:, None, None] / 100_000.0
        tau_days = 0.8 + 13.0 * (1.0 - pfrac**2)
        return self.config.newtonian_relaxation_rate_scale / (
            tau_days * 86_400.0
        )

    def _initial_temperature(self) -> np.ndarray:
        # Start from the zonal mean of the prescribed equilibrium. Named land
        # and current anomalies then spin up the circulation instead of being
        # baked directly into the initial wind field.
        weights = self.active.astype(float)
        numerator = np.sum(self.radiative_equilibrium_k * weights, axis=2, keepdims=True)
        denominator = np.maximum(np.sum(weights, axis=2, keepdims=True), 1.0)
        zonal = numerator / denominator
        initial = np.broadcast_to(zonal, self.radiative_equilibrium_k.shape).copy()

        rng = np.random.default_rng(self.config.random_seed)
        perturb = rng.normal(0.0, 0.03, size=initial.shape)
        initial += perturb * self.active
        initial[~self.active] = self.radiative_equilibrium_k[~self.active]
        return initial

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _gather_lowest(self, field: np.ndarray) -> np.ndarray:
        return np.take_along_axis(field, self.lowest_index[None, :, :], axis=0)[0]

    def sea_level_pressure_hpa(self) -> np.ndarray:
        # The prognostic anomaly represents the column-mass departure from a
        # 1013.25 hPa reference. Reporting it directly avoids the very noisy
        # exponential reduction of pressure measured on 5-km terrain.
        return 1013.25 + self.surface_pressure_anomaly_pa / 100.0

    def sea_level_pressure_anomaly_hpa(self) -> np.ndarray:
        slp = self.sea_level_pressure_hpa()
        weighted_mean = np.sum(slp * self.grid.area_weight) / np.sum(self.grid.area_weight)
        return slp - weighted_mean

    def geopotential_height_m(self) -> np.ndarray:
        return self.last_geopotential_m2_s2 / self.config.gravity_m_s2

    def wind_speed_m_s(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v)

    def level_index(self, pressure_hpa: float | int) -> int:
        return int(np.argmin(np.abs(self.pressure_hpa - float(pressure_hpa))))

    def status(self) -> dict[str, Any]:
        tibet_y = int(np.argmin(np.abs(self.grid.lat_deg - 32.0)))
        tibet_x = int(np.argmin(np.abs(((self.grid.lon_deg - 87.0 + 180.0) % 360.0) - 180.0)))
        blocked = self.pressure_hpa[~self.active[:, tibet_y, tibet_x]].astype(int).tolist()
        wind = self.wind_speed_m_s()
        return {
            "simulation_days": self.time_seconds / 86_400.0,
            "max_wind_m_s": float(np.nanmax(np.where(self.active, wind, np.nan))),
            "mean_wind_m_s": float(np.nanmean(np.where(self.active, wind, np.nan))),
            "slp_min_hpa": float(np.nanmin(self.sea_level_pressure_hpa())),
            "slp_max_hpa": float(np.nanmax(self.sea_level_pressure_hpa())),
            "tibet_elevation_m": float(self.boundary.surface_elevation_m[tibet_y, tibet_x]),
            "tibet_surface_pressure_hpa": float(self.boundary.base_surface_pressure_pa[tibet_y, tibet_x] / 100.0),
            "tibet_blocked_levels_hpa": blocked,
        }

    # ------------------------------------------------------------------
    # Dynamical core
    # ------------------------------------------------------------------
    def compute_geopotential(
        self,
        temperature_k: np.ndarray | None = None,
        surface_pressure_anomaly_pa: np.ndarray | None = None,
    ) -> np.ndarray:
        t = self.temperature_k if temperature_k is None else temperature_k
        ps_anom = (
            self.surface_pressure_anomaly_pa
            if surface_pressure_anomaly_pa is None
            else surface_pressure_anomaly_pa
        )
        ps = np.clip(
            self.boundary.base_surface_pressure_pa + ps_anom,
            30_000.0,
            110_000.0,
        )
        phi = np.zeros_like(t)
        surface_phi = self.config.gravity_m_s2 * self.boundary.surface_elevation_m
        r = self.config.gas_constant_dry_air

        for k in range(self.nz):
            first = self.lowest_index == k
            if np.any(first):
                phi[k, first] = (
                    surface_phi[first]
                    + r * t[k, first] * np.log(ps[first] / self.pressure_pa[k])
                )
            if k > 0:
                continuation = self.active[k] & self.active[k - 1]
                if np.any(continuation):
                    dlnp = np.log(self.pressure_pa[k - 1] / self.pressure_pa[k])
                    phi[k, continuation] = (
                        phi[k - 1, continuation]
                        + r
                        * 0.5
                        * (t[k - 1, continuation] + t[k, continuation])
                        * dlnp
                    )
            phi[k, ~self.active[k]] = surface_phi[~self.active[k]]
        return phi

    def _vertical_derivative(self, field: np.ndarray) -> np.ndarray:
        out = np.zeros_like(field)
        for k in range(self.nz):
            below = max(k - 1, 0)
            above = min(k + 1, self.nz - 1)
            have_below = self.active[below]
            have_above = self.active[above]

            central = self.active[k] & have_below & have_above & (above != below)
            if np.any(central):
                out[k, central] = (
                    field[above, central] - field[below, central]
                ) / (self.pressure_pa[above] - self.pressure_pa[below])

            only_above = self.active[k] & (~have_below) & have_above & (above != k)
            if np.any(only_above):
                out[k, only_above] = (
                    field[above, only_above] - field[k, only_above]
                ) / (self.pressure_pa[above] - self.pressure_pa[k])

            only_below = self.active[k] & have_below & (~have_above) & (below != k)
            if np.any(only_below):
                out[k, only_below] = (
                    field[k, only_below] - field[below, only_below]
                ) / (self.pressure_pa[k] - self.pressure_pa[below])
        return out

    def _vertical_mixing(self, field: np.ndarray) -> np.ndarray:
        mix = np.zeros_like(field)
        for k in range(self.nz):
            if k > 0:
                pair = self.active[k] & self.active[k - 1]
                mix[k, pair] += field[k - 1, pair] - field[k, pair]
            if k + 1 < self.nz:
                pair = self.active[k] & self.active[k + 1]
                mix[k, pair] += field[k + 1, pair] - field[k, pair]
        # Boundary-layer mixing is stronger than free-tropospheric mixing.
        rate = self.config.vertical_mixing_rate_s * (0.45 + 1.8 * self.surface_weight)
        return rate * mix

    def _diagnose_omega_and_mass_tendency(
        self,
        u: np.ndarray,
        v: np.ndarray,
        temperature_k: np.ndarray,
        ps_anom: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        div = self.grid.divergence(u, v, self.active)
        omega = np.zeros_like(div)
        running = np.zeros((self.ny, self.nx), dtype=float)

        # Integrate continuity downward from omega=0 at the model top.
        for k in range(self.nz - 1, -1, -1):
            layer_div = np.where(self.active[k], div[k], 0.0)
            omega[k] = running - 0.5 * layer_div * self.dp_pa
            running = running - layer_div * self.dp_pa

        # Mechanical lift over terrain. The lowest active wind is projected
        # onto the terrain slope, then converted from w (m/s) to omega (Pa/s).
        u_low = self._gather_lowest(u)
        v_low = self._gather_lowest(v)
        t_low = self._gather_lowest(temperature_k)
        w_orog = (
            u_low * self.boundary.terrain_slope_x
            + v_low * self.boundary.terrain_slope_y
        ) * self.config.orographic_lift_scale
        rho_low = self.boundary.base_surface_pressure_pa / (
            self.config.gas_constant_dry_air * np.clip(t_low, 200.0, 325.0)
        )
        omega_orog = -rho_low * self.config.gravity_m_s2 * w_orog
        omega_orog = np.clip(omega_orog, -0.8, 0.8)
        omega += self.surface_weight * omega_orog[None, :, :]
        omega[~self.active] = 0.0

        dps = self.config.surface_pressure_coupling * running
        # Total atmospheric mass is conserved to numerical precision.
        mean_dps = np.sum(dps * self.grid.area_weight) / np.sum(self.grid.area_weight)
        dps -= mean_dps
        dps -= self.config.mass_damping_rate_s * (
            ps_anom - self.boundary.seasonal_surface_pressure_anomaly_pa
        )
        dps += self.config.horizontal_diffusion_rate_s * self.grid.laplacian_index(ps_anom)
        return omega, dps

    def _rhs(
        self,
        u: np.ndarray,
        v: np.ndarray,
        temperature_k: np.ndarray,
        ps_anom: np.ndarray,
    ) -> Tendencies:
        phi = self.compute_geopotential(temperature_k, ps_anom)
        dphidx = self.grid.grad_x(phi, self.active)
        dphidy = self.grid.grad_y(phi, self.active)

        omega, dps = self._diagnose_omega_and_mass_tendency(u, v, temperature_k, ps_anom)

        scheme = self.config.advection_scheme
        adv_u = self.grid.advection(u, u, v, self.active, scheme)
        adv_v = self.grid.advection(v, u, v, self.active, scheme)
        adv_t = self.grid.advection(temperature_k, u, v, self.active, scheme)
        dudp = self._vertical_derivative(u)
        dvdp = self._vertical_derivative(v)

        metric_coriolis = (
            2.0 * self.config.rotation_rate_s * self.grid.sin_lat
            + u * np.tan(self.grid.lat2d_rad) / self.config.earth_radius_m
        )

        du = -adv_u - omega * dudp + metric_coriolis * v - dphidx
        dv = -adv_v - omega * dvdp - metric_coriolis * u - dphidy

        # Damp only the divergent component of fast pressure/gravity waves.
        # For a Fourier mode, +K grad(div V) gives -K k^2 times its
        # longitudinal velocity while leaving non-divergent rotation intact.
        divergence_damping = self.config.divergence_damping_m2_s
        if divergence_damping > 0.0:
            horizontal_divergence = self.grid.divergence(u, v, self.active)
            du += divergence_damping * self.grid.grad_x(
                horizontal_divergence,
                self.active,
            )
            dv += divergence_damping * self.grid.grad_y(
                horizontal_divergence,
                self.active,
            )

        dtdp = self._vertical_derivative(temperature_k)
        dtemp = (
            -adv_t
            - omega * dtdp
            + self.config.kappa
            * temperature_k
            * omega
            / self.pressure_pa[:, None, None]
            + self.radiative_rate_s
            * (self.radiative_equilibrium_k - temperature_k)
        )

        # Horizontal and vertical sub-grid mixing.
        nu = self.config.horizontal_diffusion_rate_s
        du += nu * self.grid.laplacian_index(u, self.active) + self._vertical_mixing(u)
        dv += nu * self.grid.laplacian_index(v, self.active) + self._vertical_mixing(v)
        dtemp += nu * self.grid.laplacian_index(temperature_k, self.active)
        dtemp += self._vertical_mixing(temperature_k)

        # Land/ocean surface drag.
        drag2d = np.where(
            self.boundary.land_mask,
            self.config.surface_drag_land_s,
            self.config.surface_drag_ocean_s,
        )
        drag = self.surface_weight * drag2d[None, :, :]
        du -= drag * u
        dv -= drag * v

        # Terrain form drag and cross-slope blocking. This term is distinct
        # from the underground-layer mask: it actively turns/damps low-level
        # flow along mountain flanks, making winds route around Tibet/Rockies.
        slope = self.boundary.terrain_slope
        safe_slope = np.maximum(slope, 1.0e-8)
        nx = self.boundary.terrain_slope_x / safe_slope
        ny = self.boundary.terrain_slope_y / safe_slope
        normal_wind = u * nx[None, :, :] + v * ny[None, :, :]
        slope_strength = np.clip((slope / 0.0016) ** 2, 0.0, 18.0)
        block_rate = (
            self.config.terrain_blocking_rate_s
            * self.surface_weight
            * slope_strength[None, :, :]
        )
        du -= block_rate * normal_wind * nx[None, :, :]
        dv -= block_rate * normal_wind * ny[None, :, :]
        du -= 0.16 * block_rate * u
        dv -= 0.16 * block_rate * v

        # Absorb waves reflected by the artificial meridional walls.  The
        # reference state is the experiment's exact balanced basic state, not
        # a cyclone or a front.  Removing the area-weighted pressure tendency
        # keeps global column mass unchanged by the numerical sponge.
        if self._meridional_sponge_rate_s is not None:
            rate2d = self._meridional_sponge_rate_s
            rate3d = rate2d[None, :, :]
            du -= rate3d * (u - self._meridional_sponge_reference_u)
            dv -= rate3d * (v - self._meridional_sponge_reference_v)
            dtemp -= rate3d * (
                temperature_k - self._meridional_sponge_reference_temperature_k
            )
            sponge_pressure = -rate2d * (
                ps_anom - self._meridional_sponge_reference_pressure_pa
            )
            sponge_pressure -= (
                np.sum(sponge_pressure * self.grid.area_weight)
                / np.sum(self.grid.area_weight)
            )
            dps += sponge_pressure

        du[~self.active] = 0.0
        dv[~self.active] = 0.0
        dtemp[~self.active] = 0.0
        return Tendencies(du, dv, dtemp, dps, omega, phi)

    def _apply_constraints(self) -> None:
        self.u[~self.active] = 0.0
        self.v[~self.active] = 0.0
        self.temperature_k[~self.active] = self.radiative_equilibrium_k[~self.active]
        self.u = np.clip(self.u, -160.0, 160.0)
        self.v = np.clip(self.v, -160.0, 160.0)
        self.temperature_k = np.clip(self.temperature_k, 175.0, 330.0)
        ps_limit = self.config.surface_pressure_anomaly_limit_pa
        self.surface_pressure_anomaly_pa = np.clip(
            self.surface_pressure_anomaly_pa,
            -ps_limit,
            ps_limit,
        )
        mean = np.sum(self.surface_pressure_anomaly_pa * self.grid.area_weight) / np.sum(
            self.grid.area_weight
        )
        self.surface_pressure_anomaly_pa -= mean

    def step(self, n_steps: int = 1) -> None:
        """Advance with the third-order strong-stability-preserving RK scheme.

        The former two-stage explicit trapezoidal method has no stable interval
        on the imaginary axis.  In this pressure-coordinate system the fast,
        nearly non-dissipative gravity-wave modes therefore grew into a
        resolution-locked string of alternating equatorial extrema.  SSPRK3
        retains the same explicit right-hand side and timestep while providing
        a finite imaginary-axis stability interval and monotone damping of the
        grid-scale mode under the existing spatial diffusion.
        """

        dt = self.config.dt_seconds
        for _ in range(int(n_steps)):
            u0 = self.u.copy()
            v0 = self.v.copy()
            t0 = self.temperature_k.copy()
            p0 = self.surface_pressure_anomaly_pa.copy()

            k1 = self._rhs(u0, v0, t0, p0)
            u1 = u0 + dt * k1.u
            v1 = v0 + dt * k1.v
            t1 = t0 + dt * k1.temperature
            p1 = p0 + dt * k1.surface_pressure_anomaly

            k2 = self._rhs(u1, v1, t1, p1)
            u2 = 0.75 * u0 + 0.25 * (u1 + dt * k2.u)
            v2 = 0.75 * v0 + 0.25 * (v1 + dt * k2.v)
            t2 = 0.75 * t0 + 0.25 * (t1 + dt * k2.temperature)
            p2 = 0.75 * p0 + 0.25 * (
                p1 + dt * k2.surface_pressure_anomaly
            )

            k3 = self._rhs(u2, v2, t2, p2)
            self.u = (u0 + 2.0 * (u2 + dt * k3.u)) / 3.0
            self.v = (v0 + 2.0 * (v2 + dt * k3.v)) / 3.0
            self.temperature_k = (
                t0 + 2.0 * (t2 + dt * k3.temperature)
            ) / 3.0
            self.surface_pressure_anomaly_pa = (
                p0
                + 2.0 * (p2 + dt * k3.surface_pressure_anomaly)
            ) / 3.0
            self.last_omega_pa_s = k3.omega
            self.last_geopotential_m2_s2 = k3.geopotential
            self._apply_constraints()
            self.time_seconds += dt

    def advance_hours(self, hours: float) -> None:
        n = max(1, int(round(float(hours) * 3600.0 / self.config.dt_seconds)))
        self.step(n)

    def copy_state(self) -> dict[str, np.ndarray | float]:
        return {
            "u": self.u.copy(),
            "v": self.v.copy(),
            "temperature_k": self.temperature_k.copy(),
            "surface_pressure_anomaly_pa": self.surface_pressure_anomaly_pa.copy(),
            "time_seconds": self.time_seconds,
        }
