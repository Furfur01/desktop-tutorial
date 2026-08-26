# Atmos20

A small, interactive **idealized Earth atmosphere** for experimenting with
global circulation and the dry life cycle of a midlatitude baroclinic wave.

- **Vertical grid:** 1000–50 hPa, one layer every 50 hPa (20 levels) in the
  circulation model; the dry baroclinic experiment currently integrates
  950–50 hPa in the 5° preview and 900–50 hPa at finer resolutions
- **Default global-circulation experiment:** Held–Suarez dry thermal forcing
  plus an explicitly documented reduced-order Hadley/Ferrel/polar overturning
  closure; no zonal wind or pressure centres are prescribed
- **Physical lower boundary:** packaged ETOPO 2022 relief sets surface
  geopotential and pressure, masks underground pressure levels, and drives
  slope lift/form drag; the renderer uses the matching 1° shaded relief
- **Two viewers:** a scientific Gradio inspector for model fields and a map-first,
  compute-then-play Windy-style view with backend rendering

The frontend includes a maximum-detail 1° grid. The scientific field inspector
defaults to the area-aggregated 2.5° circulation grid. The longer dry
baroclinic-wave playback defaults to 5°; 2.5° is its practical high-quality
preset, while a 1° global life cycle can take hours.

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
boundary-condition and animation parameters in the left drawer,
then choose **开始计算**. Python advances Atmos20 and renders the map layers and
particle loop; the page shows real stage progress and automatically plays the
new result when its manifest is complete.

The browser still receives only WebP media and a small JSON manifest. It never
downloads model arrays or advances atmospheric particles itself.

Start the server:

```bash
python windy_app.py --host 127.0.0.1 --port 7861
```

Open `http://127.0.0.1:7861`. The default **全球地形环流 · HS + ETOPO + 闭合**
experiment uses a 5° grid at 900 hPa, a five-day establishment period, a
continuous five-day analysis/playback window, 48 frames at 12 fps, and 2400
particles. The 2.5° preset raises spatial and playback resolution. Only one
heavy calculation runs at a time, and up to five completed render bundles are
retained. A result is published only when all five directional checks—tropical
easterlies, two midlatitude westerly belts, and two polar easterly belts—pass at
every sampled time, the numerical wind-speed limits pass, and the tropical
high-wavenumber gravity-wave gate finds no growing grid-locked pattern.

The same standard run can be queued through the small HTTP API (the response
contains a `statusUrl` to poll until its manifest is ready):

```bash
curl -X POST http://127.0.0.1:7861/api/render -H "Content-Type: application/json" -d '{"scenario":"circulation","resolution":5,"level":900,"region":"world","spinupHours":120,"analysisHours":120,"frames":48,"fps":12,"particles":2400,"equatorToPoleContrastK":60,"surfaceDragDays":1,"seasonalHeatEquatorDeg":0}'
```

Controls are scenario-aware. The circulation experiment exposes the
equator-to-pole temperature contrast, near-surface drag time, seasonal
thermal-equator latitude, Tibetan height, seasonal land heating, and ocean-
current temperature anomalies. The dry baroclinic experiment exposes the developing
hemisphere, jet peak speed, and trigger amplitude. Both scenarios expose the
grid, pressure-level view, animation frame rate, particle density, flow speed,
trail length, and display opacity.

### What is physical in the three-cell circulation experiment

Held–Suarez Newtonian relaxation supplies the zonally symmetric radiative
temperature contrast, while sigma-dependent Rayleigh drag represents the
planetary boundary layer. The compact grid cannot resolve the long eddy
statistics needed to produce a stable Ferrel cell, so the default mode adds a
transparent reduced-order closure: it relaxes only the zonal-mean meridional
wind toward vertically mass-neutral lower/upper branches for the Hadley,
Ferrel, and polar cells. It never assigns zonal wind or surface pressure.
The prognostic momentum equation generates the east/west wind signs through
Coriolis acceleration and drag. Setting the thermal contrast to zero switches
the closure off; moving the thermal equator shifts the continuous cell basis
seasonally.

This is a conceptual global-circulation simulator, not a claim that the coarse
primitive-equation grid has independently resolved a climate. The manifest
records the closure and terrain parameters. The circulation starts from a
terrain-balanced resting state: ETOPO changes local surface pressure and
surface geopotential, removes below-ground levels, and supplies mechanical
lift/form drag. No wind field or pressure centre is inserted.

### What is physical in the dry cyclone experiment

The initial atmosphere follows the analytic pressure-coordinate test of
[Jablonowski and Williamson (2006)](https://doi.org/10.1256/qj.06.12): its
zonal jets, temperature, and geopotential are hydrostatically and
gradient-wind balanced. The only imposed trigger is the published localized
Gaussian perturbation to zonal wind—1 m/s by default, centred near 20°E and
40° in the selected hemisphere. No low-pressure centre, cyclone-shaped vortex,
warm-sector triangle, cold front, or warm front is drawn into the initial
state.

Python continuously integrates the dry primitive-equation model through the
requested life cycle and samples real model states along that same run. It then
pre-renders synchronized field and particle WebP animations for playback; this
is not a frozen field animated in the browser, nor an offline cache prepared in
advance. Playback keeps hour 0 through day 10 but samples the late, rapidly
developing part of the life cycle more densely. The front layer is an objective diagnostic of the evolving output:
potential-temperature gradient and the thermal-front parameter locate frontal
zones and lines, while isotherm-normal motion over an approximately fixed
six-hour diagnostic lag distinguishes cold and warm fronts independently of
the nonuniform playback sampling. The minimum-pressure and front-cell readouts
in the timeline are diagnostics, not forcing inputs.

The dry experiment is intentionally idealized. It has no moisture, clouds,
latent heating, or precipitation, and its analytic lower boundary replaces the
real-terrain dynamics used by the circulation scenario; the circulation-only
mechanical terrain-lift parameterization is explicitly disabled. The fixed
pressure grid, weak numerical diffusion/mixing, and truncated latitude domain
make this a mechanism demonstration rather than a quantitative reproduction of
the published reference solution. The dynamical grid
stops at ±77.5°, although the contextual world map continues to the poles. A
smooth absorbing layer from 62.5° poleward relaxes only boundary departures
back toward the unperturbed analytic state, preventing the artificial wall
from reflecting grid noise into the weather region; timeline diagnostics use
only the undamped latitude band. At 5°, fronts are broad and their symbols are
necessarily coarse; use 2.5° when
the frontal geometry matters more than render time. The experiment is suitable
for studying a physically triggered baroclinic life cycle, not for reproducing
a particular observed storm.

Every render is a complete 2:1 equirectangular world texture (360° longitude by
180° latitude) mapped onto the draggable orthographic globe. A flat global view
is used only as a compatibility fallback when WebGL 2 is unavailable.
`web/assets/prerender/` is only the instant initial view. Regenerate that fallback
after changing renderer code with:

```bash
python scripts/prerender_windy.py --scenario circulation --output web/assets/prerender --grid-degrees 5 --spinup-hours 120 --analysis-hours 120 --frames 48 --fps 12 --particles 2400 --level 900
```

The renderer CLI uses these same circulation defaults, so `python
scripts/prerender_windy.py` regenerates the default browser fallback directly.

## Run without the frontend

```bash
python run_headless.py --days 5 --level 850
```

This writes `output/atmos20_state.png` and a compressed `output/atmos20_state.npz` containing every model field.

## What the Tibetan Plateau actually does here

At each horizontal cell the base surface pressure is estimated from prescribed elevation and temperature. A pressure level exists only where it lies above the local ground. Near central Tibet, where the idealized plateau exceeds roughly 4–5 km, the 1000, 950, 900 ... and several additional lower pressure levels become solid terrain. Finite-volume face fluxes into those cells are set to zero, so an 850 hPa wind must route around the plateau. The lowest active layers also receive slope-dependent form drag and an upslope vertical-velocity boundary condition.

The map keeps the 3000 and 4500 m terrain contours visible, and underground regions at the selected pressure level are shaded gray.

## Equations and limitations

The core is a deliberately simplified hydrostatic pressure-level model. The
default three-cell setup uses SSPRK3 time stepping, TVD advection,
Held–Suarez relaxation and local-sigma drag, terrain coupling, selective
divergence damping, plus the declared reduced-order overturning closure. The dry baroclinic setup
uses the same TVD family with an analytically balanced boundary and only weak
numerical mixing. See [`docs/model.md`](docs/model.md) for the equations, front
diagnostics, and implementation choices.

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

The tests verify the pressure-level grid, underground masking over Tibet,
terrain-balanced initialization, nonzero terrain influence on low-level momentum,
rejection of equatorial high-wavenumber strings, advection behavior,
baroclinic initial-state balance and perturbation reproducibility, objective
front classification, and short-term numerical stability.
