# Model equations and terrain coupling

Atmos20 is a compact mechanism model intended for interactive experiments. It uses fixed pressure levels

\[
p_k = 1000,950,\ldots,50\ \mathrm{hPa}
\]

and evolves horizontal wind \((u,v)\), temperature \(T\), and a two-dimensional surface-pressure anomaly.

## Horizontal momentum

At each active pressure level the model advances a hydrostatic primitive-equation-like momentum balance:

\[
\frac{D\mathbf{v}}{Dt}+f\mathbf{k}\times\mathbf{v}
=-\nabla_p\Phi+\mathbf{F}_{mix}+\mathbf{F}_{drag}+\mathbf{F}_{orog}.
\]

Advection is first-order upwind for robustness. Horizontal diffusion and nearest-level vertical mixing remove grid-scale noise.

## Hydrostatic geopotential

Geopotential is integrated upward from the actual surface:

\[
\Phi(p)=g z_s+R_d\int_p^{p_s}T\,d\ln p.
\]

The lower boundary therefore changes the height and horizontal gradient of pressure surfaces. Surface-pressure anomalies alter the integration limit and feed back onto all levels.

## Thermodynamics and vertical motion

The temperature tendency contains horizontal advection, pressure-coordinate vertical advection, adiabatic heating/cooling, and Newtonian relaxation toward an equilibrium profile derived from the prescribed surface temperature:

\[
\frac{DT}{Dt}-\kappa\frac{T\omega}{p}=Q_{relax}+Q_{mix}.
\]

Vertical pressure velocity is diagnosed from continuity by integrating horizontal divergence downward from \(\omega=0\) at the model top. A terrain-following lower-boundary contribution is added from

\[
w_s=u_s\frac{\partial z_s}{\partial x}+v_s\frac{\partial z_s}{\partial y}.
\]

## Season and balanced initialization

The lower boundary exposes a seasonal phase: +1 for boreal summer, 0 for an
equinox, and -1 for boreal winter. It shifts the prescribed land and ocean warm
belts between hemispheres and scales the named continental heating anomalies.
Land surface temperature is reduced with terrain elevation before the
hydrostatic surface pressure is calculated.

A zero-wind, zero-pressure-anomaly start is not balanced with the summer Asian
heating: the raised warm-column thickness initially accelerates air away from
the continent and resembles a winter monsoon for several simulated days.
Atmos20 therefore initializes a zero-global-mean seasonal surface-pressure
anomaly. Hot land receives a thermal low, while broad South Asian and western
Pacific pressure centres supply the corresponding summer monsoon background.
The mass-damping term relaxes toward this seasonal target rather than toward a
spatially uniform pressure anomaly. The prognostic model still evolves the
field after initialization.

## How mountains affect wind

Terrain enters four separate pieces of the model:

1. **Underground pressure levels are removed.** A cell is active only when \(p_k\le p_s(z_s,T_s)\). Surface elevation is a 1° aggregate derived from NOAA NCEI ETOPO 2022; near central Tibet many lower pressure levels are solid terrain.
2. **Masked finite-volume fluxes impose a wall.** Momentum and mass cannot flow through a neighbouring underground cell at that pressure level.
3. **Cross-slope form drag turns and damps low-level flow.** The normal component \(\mathbf{v}\cdot\nabla z_s\) is preferentially reduced within the boundary layer.
4. **Upslope flow produces vertical motion.** The diagnosed \(w_s\) feeds adiabatic cooling/heating and the vertical circulation.

## Scope

This is not a forecast model. It omits moisture, clouds, radiation transfer, ocean feedback, sub-grid convection, a fully conservative terrain-following coordinate, and a resolved stratosphere. It is designed to show how fixed summer heating, ocean-current temperature anomalies, rotation, hydrostatic structure, and major mountain systems organize circulation while still running interactively on a personal computer.
