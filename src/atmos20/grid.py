from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass(slots=True)
class LatLonGrid:
    config: ModelConfig
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    lon2d_deg: np.ndarray
    lat2d_deg: np.ndarray
    lat2d_rad: np.ndarray
    cos_lat: np.ndarray
    sin_lat: np.ndarray
    dx_m: np.ndarray
    dy_m: float
    area_weight: np.ndarray

    @classmethod
    def build(cls, config: ModelConfig) -> "LatLonGrid":
        lon = np.arange(0.0, 360.0, config.dlon_deg, dtype=float)
        lat = np.arange(
            -config.lat_limit_deg,
            config.lat_limit_deg + 0.25 * config.dlat_deg,
            config.dlat_deg,
            dtype=float,
        )
        lon2d, lat2d = np.meshgrid(lon, lat)
        lat_rad = np.deg2rad(lat2d)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        dlon_rad = np.deg2rad(config.dlon_deg)
        dlat_rad = np.deg2rad(config.dlat_deg)
        dx = config.earth_radius_m * np.maximum(cos_lat, 0.15) * dlon_rad
        dy = config.earth_radius_m * dlat_rad
        return cls(
            config=config,
            lon_deg=lon,
            lat_deg=lat,
            lon2d_deg=lon2d,
            lat2d_deg=lat2d,
            lat2d_rad=lat_rad,
            cos_lat=cos_lat,
            sin_lat=sin_lat,
            dx_m=dx,
            dy_m=dy,
            area_weight=cos_lat,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.lon2d_deg.shape

    @staticmethod
    def _north(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[..., :-1, :] = a[..., 1:, :]
        out[..., -1, :] = a[..., -1, :]
        return out

    @staticmethod
    def _south(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        out[..., 1:, :] = a[..., :-1, :]
        out[..., 0, :] = a[..., 0, :]
        return out

    def grad_x(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        east = np.roll(field, -1, axis=-1)
        west = np.roll(field, 1, axis=-1)
        if mask is not None:
            east = np.where(np.roll(mask, -1, axis=-1), east, field)
            west = np.where(np.roll(mask, 1, axis=-1), west, field)
        return (east - west) / (2.0 * self.dx_m)

    def grad_y(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        north = self._north(field)
        south = self._south(field)
        if mask is not None:
            north = np.where(self._north(mask), north, field)
            south = np.where(self._south(mask), south, field)
        return (north - south) / (2.0 * self.dy_m)

    def laplacian_index(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        east = np.roll(field, -1, axis=-1)
        west = np.roll(field, 1, axis=-1)
        north = self._north(field)
        south = self._south(field)
        if mask is not None:
            east = np.where(np.roll(mask, -1, axis=-1), east, field)
            west = np.where(np.roll(mask, 1, axis=-1), west, field)
            north = np.where(self._north(mask), north, field)
            south = np.where(self._south(mask), south, field)
        return east + west + north + south - 4.0 * field

    def upwind_advection(
        self,
        field: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        east = np.roll(field, -1, axis=-1)
        west = np.roll(field, 1, axis=-1)
        north = self._north(field)
        south = self._south(field)

        east = np.where(np.roll(mask, -1, axis=-1), east, field)
        west = np.where(np.roll(mask, 1, axis=-1), west, field)
        north = np.where(self._north(mask), north, field)
        south = np.where(self._south(mask), south, field)

        dfdx = np.where(u >= 0.0, (field - west) / self.dx_m, (east - field) / self.dx_m)
        dfdy = np.where(v >= 0.0, (field - south) / self.dy_m, (north - field) / self.dy_m)
        return u * dfdx + v * dfdy

    def divergence(self, u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Finite-volume spherical divergence with solid-wall terrain masks."""

        east_mask = mask & np.roll(mask, -1, axis=-1)
        west_mask = mask & np.roll(mask, 1, axis=-1)
        u_e = 0.5 * (u + np.roll(u, -1, axis=-1)) * east_mask
        u_w = 0.5 * (u + np.roll(u, 1, axis=-1)) * west_mask
        div_x = (u_e - u_w) / self.dx_m

        cos3 = self.cos_lat
        vc = v * cos3
        north_mask = mask & self._north(mask)
        south_mask = mask & self._south(mask)
        vc_n = 0.5 * (vc + self._north(vc)) * north_mask
        vc_s = 0.5 * (vc + self._south(vc)) * south_mask
        vc_n[..., -1, :] = 0.0
        vc_s[..., 0, :] = 0.0
        div_y = (vc_n - vc_s) / (self.dy_m * np.maximum(cos3, 0.15))
        return np.where(mask, div_x + div_y, 0.0)
