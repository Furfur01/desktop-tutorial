# Atmos20

A small, interactive **20-level idealized Earth atmosphere** for experimenting with boreal-summer circulation.

- **Vertical grid:** 1000–50 hPa, one layer every 50 hPa (20 levels)
- **Fixed lower boundary:** prescribed land temperature, prescribed SST, fixed warm/cold current anomalies
- **Real global terrain:** a compact 1° aggregate derived from NOAA NCEI ETOPO 2022
- **Terrain-aware flow:** underground pressure levels are masked, cross-mountain flux is blocked, low-level normal wind receives form drag, and upslope flow creates vertical motion
- **Two viewers:** a scientific Gradio inspector for model fields and a map-first,
  compute-then-play Windy-style view with backend rendering

The frontend includes a maximum-detail 1° grid. A benchmark on the reference machine found it too slow for comfortable interaction, so the area-aggregated 2.5° grid is the default; a 5° mode is also available.

## What you can inspect

The frontend provides a horizontal pressure-level map and a longitude-pressure vertical section. At 850 hPa, the central Tibetan Plateau is below ground and is rendered as a solid obstacle; at 90°E, the vertical section shows the plateau reaching roughly 550–600 hPa.

## Run the frontend

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python app.py
```

Open the local Gradio address printed in the terminal.

## Run the compute-then-play flow map

The Windy-style page uses an on-demand backend render queue. Adjust model,
boundary-condition, viewport, and animation parameters in the left drawer,
then choose **开始计算**. Python advances Atmos20 and renders the map layers and
particle loop; the page shows real stage progress and automatically plays the
new result when its manifest is complete.

The browser still receives only WebP media and a small JSON manifest. It never
downloads model arrays or advances atmospheric particles itself.

Start the server:

```bash
python windy_app.py
```

Open `http://127.0.0.1:7861`. The panel offers quick, balanced, and quality
presets plus 1°/2.5°/5° grids, all 20 pressure levels, boreal
summer/equinox/winter forcing, East Asia/Asia/global views, simulation duration,
three lower-boundary multipliers, frame rate,
particle density, flow speed, trail length, and display opacity. Only one heavy
calculation runs at a time; up to five completed render bundles are retained.

The default is boreal summer. A seasonally balanced surface-pressure warm start
pairs the hot Asian landmass with a continental thermal low, so a short render
does not begin with the spurious northerly outflow produced by a zero-pressure,
zero-wind cold start. Terrain elevation also cools the prescribed land surface,
preventing the Tibetan Plateau from being initialized near 40 °C.

`web/assets/prerender/` is only the instant initial view. Regenerate that
fallback after changing renderer code with:

```bash
python scripts/prerender_windy.py
```

## Run without the frontend

```bash
python run_headless.py --days 5 --level 850
```

This writes `output/atmos20_state.png` and a compressed `output/atmos20_state.npz` containing every model field.

## What the Tibetan Plateau actually does here

At each horizontal cell the base surface pressure is estimated from prescribed elevation and temperature. A pressure level exists only where it lies above the local ground. Near central Tibet, where the idealized plateau exceeds roughly 4–5 km, the 1000, 950, 900 ... and several additional lower pressure levels become solid terrain. Finite-volume face fluxes into those cells are set to zero, so an 850 hPa wind must route around the plateau. The lowest active layers also receive slope-dependent form drag and an upslope vertical-velocity boundary condition.

The map keeps the 3000 and 4500 m terrain contours visible, and underground regions at the selected pressure level are shaded gray.

## Equations and limitations

The core is a deliberately simplified hydrostatic pressure-level model with upwind horizontal advection, diagnosed pressure velocity, Newtonian thermal relaxation, surface-pressure mass adjustment, vertical mixing, surface drag, and terrain forcing. See [`docs/model.md`](docs/model.md) for the equations and implementation choices.

It is useful for mechanism experiments and visualization. It is not suitable for weather forecasts or quantitative climate attribution.

## Terrain data

The packaged terrain asset is generated from the NOAA NCEI ETOPO 2022
ice-surface elevation grid. The source 1 arc-minute field is sampled at 0.25°
and area-aggregated into 1° and 2.5° Atmos20 assets. Normal model runs use the
compact packaged assets and do not download data. To regenerate them:

```bash
pip install -e .[dev]
python scripts/fetch_etopo.py --download
```

## Tests

```bash
pip install -e .[dev]
pytest
```

The tests verify the 20-level grid, underground masking over Tibet, nonzero terrain influence on low-level momentum, and short-term numerical stability.
