"""Run a coarse Held--Suarez spin-up and print objective wind-belt metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from atmos20 import (
    AtmosphereModel,
    HeldSuarezSpec,
    ModelConfig,
    configure_held_suarez_circulation,
    wind_belt_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=120.0)
    parser.add_argument("--interval-days", type=float, default=10.0)
    parser.add_argument("--seasonal-latitude", type=float, default=0.0)
    parser.add_argument("--dlon", type=float, default=10.0)
    parser.add_argument("--dlat", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--pressure-step", type=int, default=50)
    args = parser.parse_args()

    config = ModelConfig(
        dlon_deg=args.dlon,
        dlat_deg=args.dlat,
        lat_limit_deg=87.5,
        dt_seconds=args.dt,
        pressure_step_hpa=args.pressure_step,
        horizontal_diffusion_rate_s=1.5e-5,
        surface_pressure_anomaly_limit_pa=8_000.0,
    )
    model = AtmosphereModel(config)
    configure_held_suarez_circulation(
        model,
        HeldSuarezSpec(
            seasonal_heat_equator_deg=args.seasonal_latitude,
        ),
    )

    interval_steps = max(
        1,
        int(round(args.interval_days * 86_400.0 / config.dt_seconds)),
    )
    target_steps = int(round(args.days * 86_400.0 / config.dt_seconds))
    completed = 0
    while completed < target_steps:
        steps = min(interval_steps, target_steps - completed)
        model.step(steps)
        completed += steps
        stats = asdict(wind_belt_statistics(model, 850.0))
        stats["day"] = model.time_seconds / 86_400.0
        stats["max_wind_m_s"] = float(model.wind_speed_m_s().max())
        print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
