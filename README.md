# Atmos20

A small, interactive **20-level idealized Earth atmosphere** for experimenting with boreal-summer circulation.

- **Vertical grid:** 1000–50 hPa, one layer every 50 hPa (20 levels)
- **Fixed lower boundary:** prescribed land temperature, prescribed SST, fixed warm/cold current anomalies
- **Explicit major terrain:** broad Tibetan Plateau/Himalaya, Rockies, Andes, Iranian Plateau, Ethiopian Highlands and others
- **Terrain-aware flow:** underground pressure levels are masked, cross-mountain flux is blocked, low-level normal wind receives form drag, and upslope flow creates vertical motion
- **Interactive frontend:** select any pressure level, inspect wind/SLP, temperature, omega, geopotential, surface temperature, or active layers; advance by a chosen number of simulated hours or stream frames continuously

The default 5° grid is intended to run interactively. A 2.5° option is available for a more detailed but slower experiment.

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

## Tests

```bash
pip install -e .[dev]
pytest
```

The tests verify the 20-level grid, underground masking over Tibet, nonzero terrain influence on low-level momentum, and short-term numerical stability.
