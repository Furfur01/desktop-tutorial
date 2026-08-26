# Model equations, baroclinic life cycle, and terrain coupling

Atmos20 is a compact mechanism model intended for interactive experiments. Its
seasonal-circulation configuration uses the fixed pressure levels

\[
p_k = 1000,950,\ldots,50\ \mathrm{hPa}
\]

and evolves horizontal wind \((u,v)\), temperature \(T\), and a two-dimensional
surface-pressure anomaly. The dry baroclinic-wave playback uses the same 50 hPa
spacing. Its coarse 5° preview extends from 950 to 50 hPa to reduce the
surface-to-lowest-level gap; finer runs retain a 900 hPa bottom so a deeper low
cannot put the lowest fixed level below the evolving analytic 1000 hPa surface.

The Windy-style interface has two distinct experiment configurations. Its
default global-circulation mode combines Held–Suarez thermal relaxation, a
declared reduced-order three-cell overturning closure, and an ETOPO 2022 lower
boundary. The dry baroclinic experiment starts from a published analytic
balance and intentionally disables terrestrial terrain. These modes share
numerical operators but are not the same boundary-value problem.

## Horizontal momentum

At each active pressure level the model advances a hydrostatic primitive-equation-like momentum balance:

\[
\frac{D\mathbf{v}}{Dt}+f\mathbf{k}\times\mathbf{v}
=-\nabla_p\Phi+\mathbf{F}_{mix}+\mathbf{F}_{drag}+\mathbf{F}_{orog}.
\]

The material derivative includes both horizontal advection and pressure-coordinate
vertical advection \(-\omega\,\partial/\partial p\). The Windy circulation and
baroclinic experiments use a monotonically limited, second-order TVD
reconstruction and SSPRK3 time stepping. Weak resolution-scaled diffusion plus
selective \(K_\delta\nabla(\nabla\cdot\mathbf v)\) damping suppress divergent
grid-scale gravity waves without directly damping rotational flow.
The legacy terrain inspector retains first-order upwind advection for
robustness.

## Reduced three-cell circulation mode

The thermal equilibrium is the Held–Suarez dry benchmark profile. Newtonian
relaxation approaches it on 40-day free-atmosphere and four-day near-surface
timescales. Rayleigh drag acts below \(\sigma=0.7\) and reaches the selected
surface drag rate at \(\sigma=1\). A movable thermal equator supplies the
seasonal phase without longitude-local heat sources or named pressure centres.

At this interactive resolution the primitive core does not resolve enough
baroclinic-eddy statistics to maintain a Ferrel cell. The default circulation
therefore declares a reduced-order closure rather than hiding a prescribed
zonal wind. A continuous latitude mode contains lower branches that are
equatorward in the Hadley and polar cells and poleward in the Ferrel cell. Its
pressure mode reverses aloft and is projected to exactly zero discrete column
mass transport. Only the zonal-mean meridional tendency is relaxed toward this
mode:

\[
\left.\frac{\partial \overline v}{\partial t}\right|_{3c}
=\frac{v_{3c}(\phi,p,t)-\overline v}{\tau_{3c}}.
\]

The closure does not modify \(u\), surface pressure, or temperature directly.
The model's prognostic Coriolis and Rayleigh terms generate tropical easterlies,
midlatitude westerlies, and polar easterlies from the lower-branch meridional
motion. Its amplitude scales with the selected equator-to-pole temperature
contrast, starts from zero, and vanishes identically when that contrast is
zero. This is a conceptual zonal-mean eddy/overturning parameterization, not a
claim that the coarse grid resolves a statistically equilibrated climate.

After the establishment period, every displayed field and the particle layer
are rendered from the same continuous sequence of model snapshots. Publication
requires five wind-belt signs at every sample, numerical wind-speed limits, and
a tropical high-wavenumber gate that rejects a growing \(k\ge8\) zonal string.
ETOPO is the physical lower boundary: local sigma controls near-surface drag,
high terrain removes underground levels, and resolved slopes supply lift and
form drag. Natural Earth boundaries remain display context only.

## Hydrostatic geopotential

Geopotential is integrated upward from the actual surface:

\[
\Phi(p)=g z_s+R_d\int_p^{p_s}T\,d\ln p.
\]

The lower boundary therefore changes the height and horizontal gradient of pressure surfaces. Surface-pressure anomalies alter the integration limit and feed back onto all levels.

## Thermodynamics and vertical motion

The temperature tendency contains horizontal advection, pressure-coordinate
vertical advection, adiabatic heating/cooling, and, where enabled, Newtonian
relaxation toward an equilibrium profile derived from the prescribed surface
temperature:

\[
\frac{DT}{Dt}-\kappa\frac{T\omega}{p}=Q_{relax}+Q_{mix}.
\]

Vertical pressure velocity is diagnosed from continuity by integrating horizontal divergence downward from \(\omega=0\) at the model top. In the seasonal-circulation experiment, a terrain-following lower-boundary contribution is added from

\[
w_s=u_s\frac{\partial z_s}{\partial x}+v_s\frac{\partial z_s}{\partial y}.
\]

The dry baroclinic experiment disables this project-specific mechanical-lift
term. Its analytic surface geopotential supplies the hydrostatic lower boundary
but is not interpreted as mountain topography.

## Dry baroclinic-wave experiment

The temperate-cyclone mode follows the pressure-coordinate test case of
[Jablonowski and Williamson (2006)](https://doi.org/10.1256/qj.06.12). Its
analytic zonal wind, temperature, and geopotential satisfy hydrostatic and
gradient-wind balance. The basic state contains symmetric midlatitude jets;
the selected hemisphere determines which jet is disturbed, not where a cyclone
is prescribed.

The only trigger is a localized Gaussian perturbation to zonal wind, centred at
20°E and 40° latitude in the selected hemisphere and set to 1 m/s by default.
The implementation does **not** insert a surface low, pressure anomaly,
cyclone-shaped vortex, thermal wedge, cold front, or warm front. With the
perturbation disabled, the analytic basic state is the control solution. With
it enabled, resolved temperature advection, convergence, vertical motion,
surface-pressure tendency, and rotation allow the baroclinic wave to amplify
and produce the cyclone life cycle.

This configuration replaces the project's real-terrain boundary with the
matching analytic surface geopotential and constant 1000 hPa reference surface
pressure. Newtonian relaxation, surface drag, terrain blocking, prescribed
seasonal heating, mechanical terrain lift, and mass relaxation are disabled. Only weak numerical
diffusion and mixing are restored for grid-scale stability; they do not define
the location or shape of a storm.

Because this compact latitude–longitude grid ends at ±77.5° rather than
including singular pole points, the baroclinic setup also uses a sine-squared
absorbing layer from 62.5° to the outer row. It relaxes departures in
\(u,v,T,p_s'\) toward the unperturbed analytic basic state, with a three-hour
e-folding time only at the wall; its pressure tendency has zero area-weighted
mean so it does not change total column mass. Front and cyclone timeline
diagnostics use only the undamped midlatitude domain.

For the Windy-style view, the model is integrated continuously from hour 0 to
the requested final time (240 hours in the standard preset). Output frames are
snapshots along that one integration, including the changing wind used by the
particle animation. Python renders the synchronized field and flow frames to
WebP after the calculation; the browser only decodes and plays them. It does
not advance a simplified atmospheric model or animate particles through a
single frozen wind field. Output times use a quadratic ease-out mapping, so the
full initial-to-day-10 evolution is retained while the rapidly developing late
stage receives more playback frames.

Cold/warm classification uses isotherm-normal motion over an approximately
fixed six-hour lag sampled from a separate three-hour diagnostic history. This
keeps the physical classification interval independent of the deliberately
nonuniform playback-frame spacing.

## Objective front diagnosis

Fronts are diagnosed from model output on the selected pressure surface rather
than supplied as geometric overlays. On a fixed pressure level the potential
temperature is

\[
\theta=T\left(\frac{p_0}{p}\right)^\kappa.
\]

The broad frontal zone requires a physically scaled thermal-gradient threshold,
\(G=|\nabla_p\theta|\). A thin front line is placed at a zero crossing of the
thermal-front parameter within that zone,

\[
\mathrm{TFP}=-\nabla_p G\cdot
\frac{\nabla_p\theta}{{|\nabla_p\theta|}}.
\]

The diagnostic also computes horizontal kinematic frontogenesis using spherical
covariant velocity gradients. Cold-versus-warm classification uses two
successive model times: the diagnosed isotherm-normal motion is

\[
c_n=-\frac{\partial\theta/\partial t}{|\nabla_p\theta|}.
\]

The unit normal points from cold toward warm air, so positive \(c_n\) is labelled
a cold front and negative \(c_n\) a warm front after a minimum-motion test.
Confidence also checks agreement with the resolved normal wind. This makes the
symbols a diagnostic of the simulated thermal structure and motion, not a
manually drawn triangular cyclone template.

## Seasonal lower-boundary temperature

The lower boundary
exposes a seasonal phase: +1 for boreal summer, 0 for an
equinox, and -1 for boreal winter. It shifts the prescribed land and ocean warm
belts between hemispheres and scales the surface-temperature anomalies. Land
temperature includes an elevation lapse-rate correction.

The Windy circulation does not import the legacy monsoon pressure centres. It
starts with zero pressure anomaly and a 264 K terrain-balanced hydrostatic
reference pressure, then relaxes the lowest local-sigma layers partly toward
the land/ocean boundary temperature. Pressure anomalies and wind responses
therefore develop through the prognostic equations. The older Gradio inspector
retains its separate balanced seasonal-pressure warm start for short monsoon
demonstrations; that initialization is not used by the Windy circulation.

## How mountains affect circulation-mode wind

Real terrain is active in both the Windy three-cell circulation and the Gradio
terrain inspector, but not in the analytic dry baroclinic experiment. In the
Windy circulation it starts from a hydrostatically balanced reference surface
pressure, so loading topography does not itself launch a gravity-wave pulse.
Terrain enters four
separate pieces of the model:

1. **Underground pressure levels are removed.** A cell is active only when \(p_k\le p_s(z_s,T_s)\). Surface elevation is a 1° aggregate derived from NOAA NCEI ETOPO 2022; near central Tibet many lower pressure levels are solid terrain.
2. **Masked finite-volume fluxes impose a wall.** Momentum and mass cannot flow through a neighbouring underground cell at that pressure level.
3. **Cross-slope form drag turns and damps low-level flow.** The normal component \(\mathbf{v}\cdot\nabla z_s\) is preferentially reduced within the boundary layer.
4. **Upslope flow produces vertical motion.** The diagnosed \(w_s\) feeds adiabatic cooling/heating and the vertical circulation.

## Scope and resolution limits

This is not a forecast model. Both configurations omit moisture, clouds,
resolved radiation transfer, ocean feedback, sub-grid convection, a fully
conservative terrain-following coordinate, and a resolved stratosphere. In
particular, the dry cyclone has no latent heating, cloud shield, or
precipitation, so its fronts and pressure deepening represent dry baroclinic
dynamics only.

The dynamical latitude grid ends at ±77.5° to avoid the pressure-coordinate
longitude singularity at the poles. The globe and contextual land map extend
to ±90°, but those polar caps are not prognostic atmosphere; the renderer
extends the last resolved rows using zonal-mean polar caps. At the default 5°
resolution, frontal gradients occupy several coarse cells and the diagnosed
front symbols are broad. The 2.5° preset improves frontal geometry at a much
higher compute cost; 1° is available but a full global life cycle can require
hours.

The circulation configuration is designed to show how fixed seasonal heating,
ocean-current temperature anomalies, rotation, hydrostatic structure, and
major mountain systems organize broad flow. The baroclinic configuration is
designed to test whether a small, reproducible disturbance grows into a
midlatitude cyclone and objectively diagnosed cold/warm fronts without those
features being prescribed. Neither configuration should be used for a weather
forecast or quantitative climate attribution.
