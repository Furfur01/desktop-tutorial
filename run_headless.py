from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from atmos20 import AtmosphereModel, ModelConfig
from atmos20.render import VARIABLES, plot_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Atmos20 mechanism model without the web UI.")
    parser.add_argument("--days", type=float, default=5.0)
    parser.add_argument("--resolution", type=float, choices=[1.0, 2.5, 5.0], default=2.5)
    parser.add_argument("--level", type=int, default=850)
    parser.add_argument("--variable", choices=VARIABLES, default=VARIABLES[0])
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(
        dlon_deg=args.resolution,
        dlat_deg=args.resolution,
        dt_seconds=120.0 if args.resolution == 1.0 else (300.0 if args.resolution == 2.5 else 600.0),
    )
    model = AtmosphereModel(config)
    model.advance_hours(args.days * 24.0)

    fig = plot_state(model, args.level, args.variable)
    fig.savefig(args.output / "atmos20_state.png", dpi=170)
    plt.close(fig)

    np.savez_compressed(
        args.output / "atmos20_state.npz",
        lon_deg=model.grid.lon_deg,
        lat_deg=model.grid.lat_deg,
        pressure_hpa=model.pressure_hpa,
        u_m_s=model.u,
        v_m_s=model.v,
        temperature_k=model.temperature_k,
        omega_pa_s=model.last_omega_pa_s,
        geopotential_m2_s2=model.last_geopotential_m2_s2,
        sea_level_pressure_hpa=model.sea_level_pressure_hpa(),
        surface_temperature_k=model.boundary.surface_temperature_k,
        terrain_m=model.boundary.surface_elevation_m,
        active_mask=model.active,
        time_seconds=model.time_seconds,
    )
    print(model.status())


if __name__ == "__main__":
    main()
