"""Build the compact Atmos20 topography asset from NOAA ETOPO 2022.

The ERDDAP request samples the 1 arc-minute source every 0.25 degree over the
model domain. Four-by-four source samples are then aggregated into each 1
degree Atmos20 cell. The resulting package asset is small enough to ship with
the model, so normal runs never need network access or a GIS dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.io import netcdf_file


SOURCE_URL = (
    "https://oceanwatch.pifsc.noaa.gov/erddap/griddap/"
    "ETOPO_2022_v1_60s.nc?z%5B682:15:10117%5D%5B7:15:21592%5D"
)
DEFAULT_CACHE = Path("data/etopo_2022_surface_0p25deg.nc")
DEFAULT_OUTPUT_DIR = Path("src/atmos20/data")


def _aggregate(samples: np.ndarray, factor: int, lon_roll: int) -> tuple[np.ndarray, np.ndarray]:
    samples = np.roll(samples, lon_roll, axis=1)
    ny = samples.shape[0] // factor
    nx = samples.shape[1] // factor
    blocks = samples.reshape(ny, factor, nx, factor)
    land_samples = blocks > 0.0
    land_count = land_samples.sum(axis=(1, 3))
    land_fraction = land_count.astype(np.float32) / float(factor * factor)
    positive_relief = np.maximum(blocks, 0.0).sum(axis=(1, 3))
    elevation = np.divide(
        positive_relief,
        land_count,
        out=np.zeros_like(positive_relief, dtype=np.float32),
        where=land_count > 0,
    )
    return elevation.astype(np.float32), land_fraction


def _write_asset(
    output: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    elevation: np.ndarray,
    land_fraction: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lon_deg=lon.astype(np.float32),
        lat_deg=lat.astype(np.float32),
        elevation_m=elevation,
        land_fraction=land_fraction,
        source=np.asarray("NOAA NCEI ETOPO 2022 ice-surface elevation"),
        source_url=np.asarray(SOURCE_URL),
    )
    print(
        f"wrote {output} ({output.stat().st_size / 1024:.1f} KiB); "
        f"elevation range {elevation.min():.0f}..{elevation.max():.0f} m"
    )


def build_assets(source: Path, output_dir: Path) -> None:
    with netcdf_file(source, "r", mmap=False) as dataset:
        samples = np.asarray(dataset.variables["z"].data, dtype=np.float32)
        source_lat = np.asarray(dataset.variables["latitude"].data)
        source_lon = np.asarray(dataset.variables["longitude"].data)

    if samples.shape != (630, 1440):
        raise ValueError(f"unexpected ETOPO subset shape: {samples.shape}")
    if not np.allclose(source_lat[[0, -1]], [-78.625, 78.625]):
        raise ValueError("unexpected ETOPO latitude coordinates")
    if not np.allclose(source_lon[[0, -1]], [0.125, 359.875]):
        raise ValueError("unexpected ETOPO longitude coordinates")

    elevation_1deg, land_1deg = _aggregate(samples[3:627], 4, 2)
    _write_asset(
        output_dir / "etopo_2022_1deg.npz",
        np.arange(0.0, 360.0, 1.0),
        np.arange(-77.5, 78.0, 1.0),
        elevation_1deg,
        land_1deg,
    )

    elevation_2p5, land_2p5 = _aggregate(samples, 10, 5)
    _write_asset(
        output_dir / "etopo_2022_2p5deg.npz",
        np.arange(0.0, 360.0, 2.5),
        np.arange(-77.5, 78.0, 2.5),
        elevation_2p5,
        land_2p5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download or not args.cache.exists():
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE_URL}")
        urlretrieve(SOURCE_URL, args.cache)
    build_assets(args.cache, args.output_dir)


if __name__ == "__main__":
    main()
