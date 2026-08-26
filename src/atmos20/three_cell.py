"""Reduced-order, axisymmetric three-cell overturning closure.

This module is deliberately separate from the prognostic dynamical core.  It
is not a claim that Hadley, Ferrel, and polar cells can be resolved by the
coarse interactive model.  Instead, it supplies the missing *zonal-mean eddy
closure*: a mass-neutral meridional overturning mode is relaxed into ``v``;
the core must still generate ``u`` through Coriolis acceleration and remove it
through its ordinary Rayleigh drag.  Surface pressure and zonal wind are never
prescribed by this closure.

The target is continuous (including a continuous first derivative) at the
thermal equator and at the 30/60/90-degree cell boundaries.  Its pressure
mode has exactly zero discrete vertical sum on a flat equally spaced pressure
grid.  With terrain masks, the projection instead removes the integrated
zonal-mean meridional transport at every latitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .model import AtmosphereModel, Tendencies


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True, slots=True)
class ThreeCellSpec:
    """Controls for the reduced-order overturning closure.

    ``reference_lower_branch_speed_m_s`` is the peak target meridional wind
    at the lowest pressure level when ``equator_to_pole_contrast_k`` equals
    ``reference_contrast_k``.  Cell strengths are relative Hadley, Ferrel,
    and polar-cell amplitudes, respectively.
    """

    enabled: bool = True
    equator_to_pole_contrast_k: float = 60.0
    reference_contrast_k: float = 60.0
    reference_lower_branch_speed_m_s: float = 1.80
    cell_strengths: tuple[float, float, float] = (1.0, 0.80, 0.70)
    thermal_equator_deg: float = 0.0
    spinup_e_folding_days: float = 2.0
    closure_e_folding_hours: float = 2.0


def _validate_spec(spec: ThreeCellSpec) -> None:
    if not np.isfinite(spec.equator_to_pole_contrast_k):
        raise ValueError("equator_to_pole_contrast_k must be finite")
    if spec.equator_to_pole_contrast_k < 0.0:
        raise ValueError("equator_to_pole_contrast_k cannot be negative")
    for name, value in (
        ("reference_contrast_k", spec.reference_contrast_k),
        ("reference_lower_branch_speed_m_s", spec.reference_lower_branch_speed_m_s),
        ("spinup_e_folding_days", spec.spinup_e_folding_days),
        ("closure_e_folding_hours", spec.closure_e_folding_hours),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if len(spec.cell_strengths) != 3:
        raise ValueError("cell_strengths must contain Hadley, Ferrel, and polar values")
    if any((not np.isfinite(value) or value < 0.0) for value in spec.cell_strengths):
        raise ValueError("cell_strengths must be finite and non-negative")
    if not np.isfinite(spec.thermal_equator_deg) or abs(spec.thermal_equator_deg) > 30.0:
        raise ValueError("thermal_equator_deg must lie within +/-30 degrees")


class ThreeCellClosure:
    """Construct the mass-neutral zonal-mean meridional-flow target."""

    def __init__(self, spec: ThreeCellSpec | None = None) -> None:
        self.spec = spec or ThreeCellSpec()
        _validate_spec(self.spec)

    def ramp_fraction(self, time_seconds: float) -> float:
        """Return the smooth spin-up fraction, exactly zero at ``t=0``."""

        if not self.spec.enabled or self.spec.equator_to_pole_contrast_k == 0.0:
            return 0.0
        time = max(0.0, float(time_seconds))
        tau = self.spec.spinup_e_folding_days * SECONDS_PER_DAY
        return float(-np.expm1(-time / tau))

    def latitude_mode(self, latitude_deg: np.ndarray | float) -> np.ndarray:
        """Return the continuous lower-branch direction/strength by latitude.

        Latitude is mapped separately from the seasonal thermal equator to
        each geographic pole.  Thus the thermal equator moves while both
        poles remain fixed; the three equal-width cells in the transformed
        coordinate retain boundaries at 0, 30, 60, and 90 degrees.
        """

        latitude = np.asarray(latitude_deg, dtype=float)
        delta = self.spec.thermal_equator_deg
        north = latitude >= delta
        pole_distance = np.where(north, 90.0 - delta, 90.0 + delta)
        transformed = np.abs(latitude - delta) * 90.0 / pole_distance
        inside = transformed < 90.0
        cell_index = np.minimum((transformed / 30.0).astype(int), 2)
        local_coordinate = (transformed - 30.0 * cell_index) / 30.0

        # sin^2 makes both value and first derivative zero at every boundary.
        envelope = np.sin(np.pi * local_coordinate) ** 2
        strengths = np.asarray(self.spec.cell_strengths, dtype=float)[cell_index]
        hemisphere_sign = np.where(north, 1.0, -1.0)
        # Lower branches: equatorward, poleward, equatorward.
        cell_sign = np.where(cell_index == 1, 1.0, -1.0)
        result = hemisphere_sign * cell_sign * strengths * envelope
        return np.where(inside, result, 0.0)

    @staticmethod
    def vertical_mode(pressure_hpa: np.ndarray) -> np.ndarray:
        """Return a shallow lower/deep upper branch with zero pressure sum.

        A symmetric cosine gives the upper branch the same peak speed as the
        thin boundary-layer return flow.  With no Held--Suarez drag aloft that
        unrealistically accelerates a zonal-mean upper jet.  A surface-peaked
        exponential instead spreads the compensating branch through the deep
        atmosphere, so its local speed is weaker while the discrete column
        transport remains exactly zero.
        """

        pressure = np.asarray(pressure_hpa, dtype=float)
        if pressure.ndim != 1 or pressure.size < 2:
            raise ValueError("pressure_hpa must be a one-dimensional multi-level grid")
        span = float(np.max(pressure) - np.min(pressure))
        if not np.isfinite(span) or span <= 0.0:
            raise ValueError("pressure_hpa must span more than one pressure")
        vertical_coordinate = (np.max(pressure) - pressure) / span
        mode = np.exp(-vertical_coordinate / 0.22)

        # Project the mode onto exactly zero discrete column transport for the
        # equally thick pressure layers used by the compact dynamical core.
        mode -= np.mean(mode)
        lower_index = int(np.argmax(pressure))
        scale = mode[lower_index]
        if abs(scale) < np.finfo(float).eps:
            raise ValueError("pressure grid cannot define a lower branch")
        return mode / scale

    def target_v(self, model: AtmosphereModel, time_seconds: float | None = None) -> np.ndarray:
        """Return the zonally symmetric, column-mass-neutral target ``v``."""

        time = model.time_seconds if time_seconds is None else float(time_seconds)
        amplitude = (
            self.spec.reference_lower_branch_speed_m_s
            * self.spec.equator_to_pole_contrast_k
            / self.spec.reference_contrast_k
            * self.ramp_fraction(time)
        )
        if amplitude == 0.0:
            return np.zeros_like(model.v)
        vertical = self.vertical_mode(model.pressure_hpa)[:, None, None]
        latitude = self.latitude_mode(model.grid.lat2d_deg)[None, :, :]
        target = amplitude * vertical * latitude
        target = np.broadcast_to(target, model.v.shape).copy()
        target[~model.active] = 0.0
        return self._remove_zonal_mass_mean(model, target)

    @staticmethod
    def _remove_zonal_mass_mean(
        model: AtmosphereModel,
        field: np.ndarray,
    ) -> np.ndarray:
        """Remove net zonal-mean meridional mass transport at each latitude.

        Over flat terrain this is identical to removing the vertical mean in
        every column.  With mountains, the number of active pressure layers
        varies by longitude; projecting the whole latitude circle preserves a
        longitude-uniform closure over the air cells instead of imprinting the
        terrain mask as a prescribed local circulation anomaly.
        """

        active = model.active
        active_count = np.sum(active, axis=(0, 2))
        transport = np.sum(np.where(active, field, 0.0), axis=(0, 2))
        barotropic = np.divide(
            transport,
            active_count,
            out=np.zeros(model.ny, dtype=float),
            where=active_count > 0,
        )
        neutral = np.where(active, field - barotropic[None, :, None], 0.0)
        return neutral

    def v_tendency(
        self,
        model: AtmosphereModel,
        v: np.ndarray,
        time_seconds: float | None = None,
    ) -> np.ndarray:
        """Relax only the zonal-mean ``v`` toward the overturning target.

        Eddies are not pointwise erased: the closure computes one active-cell
        longitude mean at each latitude and level and adds a longitude-uniform
        tendency.  Consequently it changes only the unresolved zonal mean.
        """

        if not self.spec.enabled:
            return np.zeros_like(v)
        active_count = np.sum(model.active, axis=2, keepdims=True)
        zonal_mean_v = np.divide(
            np.sum(np.where(model.active, v, 0.0), axis=2, keepdims=True),
            active_count,
            out=np.zeros((*v.shape[:2], 1), dtype=float),
            where=active_count > 0,
        )
        target = self.target_v(model, time_seconds)
        target_zonal_mean = np.divide(
            np.sum(np.where(model.active, target, 0.0), axis=2, keepdims=True),
            active_count,
            out=np.zeros((*v.shape[:2], 1), dtype=float),
            where=active_count > 0,
        )
        rate = 1.0 / (self.spec.closure_e_folding_hours * 3600.0)
        tendency = rate * (target_zonal_mean - zonal_mean_v)
        tendency = np.broadcast_to(tendency, v.shape).copy()
        tendency[~model.active] = 0.0
        # The closure itself cannot inject a barotropic meridional current.
        return self._remove_zonal_mass_mean(model, tendency)


class ThreeCellAtmosphereModel(AtmosphereModel):
    """``AtmosphereModel`` with an optional reduced-order three-cell closure.

    No generic-core modification is required.  Adding the tendency inside
    ``_rhs`` means both stages of the existing Runge--Kutta step see the
    meridional acceleration, so the ordinary Coriolis term generates zonal
    wind rather than the closure assigning it.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        boundary=None,
        three_cell_spec: ThreeCellSpec | None = None,
    ) -> None:
        super().__init__(config=config, boundary=boundary)
        self.three_cell_closure: ThreeCellClosure | None = (
            ThreeCellClosure(three_cell_spec) if three_cell_spec is not None else None
        )

    def set_three_cell_closure(self, spec: ThreeCellSpec | None) -> None:
        """Enable/replace the closure, or disable it by passing ``None``."""

        self.three_cell_closure = ThreeCellClosure(spec) if spec is not None else None

    def _rhs(
        self,
        u: np.ndarray,
        v: np.ndarray,
        temperature_k: np.ndarray,
        ps_anom: np.ndarray,
    ) -> Tendencies:
        tendency = super()._rhs(u, v, temperature_k, ps_anom)
        if self.three_cell_closure is not None:
            tendency.v += self.three_cell_closure.v_tendency(
                self,
                v,
                self.time_seconds,
            )
        return tendency


__all__ = [
    "ThreeCellAtmosphereModel",
    "ThreeCellClosure",
    "ThreeCellSpec",
]
