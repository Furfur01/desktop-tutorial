# Atmos20 — 气象模拟

[中文说明](README.zh-CN.md)

Atmos20 is an interactive, idealized atmospheric-circulation and weather-system simulator. It integrates a compact 20-level dry atmosphere, renders the evolving model state into browser-safe animated media, and plays the result on a rotatable globe.

> Atmos20 is a mechanism demonstration and numerical playground. It is not a weather forecast model, reanalysis product, or climate projection.

## What it does

The repository currently contains two experiment families:

- **Terrain-coupled global circulation** — Held–Suarez-style thermal relaxation and surface drag, ETOPO lower-boundary terrain, land/ocean temperature contrasts, orographic coupling, and a reduced three-cell overturning closure.
- **Dry baroclinic-wave life cycle** — an idealized mid-latitude instability experiment with objective cyclone and front diagnostics.

Each run produces synchronized animated WebP assets for wind, signed zonal wind, temperature, pressure anomaly, vertical motion, geopotential height, fronts where applicable, and particle trajectories. The browser receives rendered media and a JSON manifest; it does not receive the full numerical state.

## Quick start

Atmos20 requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .[dev]

python scripts/prerender_windy.py \
  --scenario circulation \
  --output web/assets/prerender \
  --grid-degrees 5 \
  --spinup-hours 120 \
  --analysis-hours 120 \
  --frames 48 \
  --fps 12 \
  --particles 2400 \
  --level 900

python windy_app.py
```

Open the address printed by `windy_app.py`.

## Manual computation and pre-rendered results

The left drawer now has two result sources:

- **Pre-rendered result** loads a completed bundle from `web/assets/prerenders/catalog.json`. This is the fastest way to explore the model and does not start a local integration.
- **Manual computation** sends the visible settings to the local backend. The backend integrates the model, renders all media, validates the circulation gate, and loads the new bundle when it is complete.

The repository ships with the original 10-day default bundle. Additional catalog entries become selectable after their cloud or local renders are published.

## Long integrations

Named profiles use days instead of making users enter very large hour counts. Internally, the renderer still converts those durations to seconds and advances one continuous model trajectory.

The included library covers:

| Profile | Grid | Simulated time | Purpose |
|---|---:|---:|---|
| `circulation-default-10d` | 5° | 10 days | Current default |
| `circulation-long-equinox-90d` | 5° | 90 days | 60-day spin-up + 30-day playback |
| `circulation-long-summer-60d` | 5° | 60 days | Boreal-summer thermal equator |
| `circulation-long-winter-60d` | 5° | 60 days | Boreal-winter thermal equator |
| `circulation-quality-20d` | 2.5° | 20 days | Finer terrain and wind belts |
| `baroclinic-north-10d` | 5° | 10 days | Northern dry life cycle |
| `baroclinic-south-10d` | 5° | 10 days | Southern dry life cycle |

Edit `prerender/profiles.json` to add a controlled parameter combination. Avoid treating the profile list as a Cartesian sweep: media size and compute cost grow quickly.

## Cloud pre-rendering with GitHub Actions

The workflow `.github/workflows/prerender.yml` is deliberately manual. In the repository’s **Actions** tab, run **build pre-render library**, then enter comma-separated profile IDs or `all`.

The workflow:

1. validates the requested profile list;
2. fans independent profiles out across a bounded matrix;
3. uses the runner’s available CPU threads for compiled numerical libraries;
4. renders each profile with a 330-minute per-job ceiling;
5. checks frame count, timeline consistency, circulation quality, and visible frame-to-frame motion;
6. uploads each bundle as a short-lived workflow artifact; and
7. optionally commits successful bundles plus the rebuilt catalog back to the selected branch.

Generated media can make the repository large. Select only the profiles that are useful, and review your repository’s current GitHub Actions usage and billing policy before launching expensive runs.

## Local multi-core rendering

A single atmospheric trajectory is causally ordered in time, so Atmos20 does not split one integration across unrelated processes. Independent profiles are safe to parallelize.

```bash
python scripts/prerender_profiles.py list

python scripts/prerender_profiles.py render-many \
  --select circulation-long-equinox-90d,baroclinic-north-10d \
  --jobs 2
```

`--jobs` controls the number of independent worker processes. Keep it below the number of profiles and reduce it when memory is limited. The GitHub Actions workflow performs the same coarse-grained parallelism with separate matrix jobs.

To render one profile:

```bash
python scripts/prerender_profiles.py render \
  --profile circulation-long-equinox-90d
```

To rebuild the browser catalog after copying or deleting bundles:

```bash
python scripts/prerender_profiles.py catalog
```

## Dynamic playback

Atmos20’s visual output is sampled from successive model states rather than from one frozen field:

- scalar layers and particle paths share the same model timeline;
- animated WebP frames are decoded and uploaded to WebGL textures in lockstep;
- the bottom time scrubber can jump to an exact model frame;
- long runs display simulated days rather than an opaque frame number;
- browsers without the required WebGL/WebCodecs path fall back to native two-dimensional animated images.

Every published bundle should pass:

```bash
python scripts/validate_prerender.py \
  web/assets/prerenders/<profile-id>/manifest.json
```

The validator rejects missing assets, mismatched frame counts, non-monotonic timelines, failed circulation gates, and animations whose sampled frames are visually identical.

## Numerical and physical scope

The current model uses:

- 20 fixed pressure levels from 1000 to 50 hPa at 50 hPa spacing;
- a longitude–latitude grid with underground pressure levels masked by terrain;
- SSPRK3 time integration;
- MC-limited second-order TVD horizontal advection;
- resolution-scaled diffusion, divergence damping, vertical mixing, and simple dry forcing;
- ETOPO 2022 relief for the lower boundary and Natural Earth boundaries for display context.

The global-circulation experiment is intended to show circulation mechanisms and terrain coupling. The baroclinic experiment is intended to show instability growth and front diagnosis. Neither configuration includes moist convection, cloud microphysics, radiation, data assimilation, or operational boundary conditions.

## Repository layout

```text
src/atmos20/                  numerical model and diagnostics
scripts/prerender_windy.py    model integration and animated media renderer
scripts/prerender_profiles.py named profiles, parallel profile runner, catalog builder
scripts/validate_prerender.py media and motion validator
prerender/profiles.json       long-run and representative parameter profiles
windy_app.py                  local HTTP server and on-demand render API
web/                          globe interface and bundled media
.github/workflows/            tests and manual cloud pre-render workflow
```

## Tests

```bash
pytest
python scripts/prerender_profiles.py --registry prerender/profiles.json matrix --select all
python scripts/validate_prerender.py web/assets/prerender/manifest.json
```

The last command validates the bundled default animation.

## References and data

- Held, I. M. & Suarez, M. J. (1994), *A Proposal for the Intercomparison of the Dynamical Cores of Atmospheric General Circulation Models*.
- Jablonowski, C. & Williamson, D. L. (2006), *A Baroclinic Instability Test Case for Atmospheric Model Dynamical Cores*.
- NOAA National Centers for Environmental Information, ETOPO 2022.
- Natural Earth 1:110m administrative boundaries.

## License

MIT. See [LICENSE](LICENSE).
