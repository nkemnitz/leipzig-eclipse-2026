# Credits and data licences

Every input is open data. Nothing here was scraped, purchased, or used outside its
licence, and each source is named below with what it actually contributed.

## Elevation and imagery — Staatsbetrieb Geobasisinformation und Vermessung Sachsen (GeoSN)

Licence: **[dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0)** — free use including
redistribution and commercial use, with attribution.

| Product | What it is | What it does here |
|---|---|---|
| **DOM1** | 1 m digital *surface* model (first return: roofs, canopy) | the skyline that blocks the sun |
| **DGM1** | 1 m digital *terrain* model (bare earth) | ground level under the observer, and the reference for canopy height |
| **LSC** | raw airborne laser point cloud, ~18 pts/m² | the voxel occupancy grid — the whole reason the canopy can attenuate rather than block |
| **DOP20** | 20 cm orthophotos, via WMS | the aerial imagery, baked at 2 m and streamed at 24 cm |
| **LoD2** | CityGML building models | the 3-D buildings, drawn for orientation only |

> Datenquelle: Staatsbetrieb Geobasisinformation und Vermessung Sachsen (GeoSN),
> dl-de/by-2-0. The laser flight over Leipzig was **January 2023** — leaf-off, which
> is the single largest source of uncertainty in the canopy model.

## Vector data — OpenStreetMap

Licence: **[ODbL 1.0](https://opendatacommons.org/licenses/odbl/)**. © OpenStreetMap
contributors. Used for building footprints (separating masonry from vegetation in the
voxel grid), water, woodland, public land and paths (which ground is standable), and
place names for labelling. Any redistributed derivative of this data carries ODbL.

## Tree cadastre — Stadt Leipzig

Licence: **dl-de/by-2-0**. The Baumkataster (182,041 trees with measured height and
crown diameter). It was measured against the laser canopy and covers only ~8% of it,
so it is *not* used for occlusion — the finding is documented rather than the data.

## Software

- **[three.js](https://threejs.org/)** — MIT
- **NumPy, SciPy, Pillow, pyproj, laspy, requests, pyosmium** — BSD/MIT
- **[pvlib](https://pvlib-python.readthedocs.io/), [Skyfield](https://rhodesmill.org/skyfield/), [ephem](https://rhodesmill.org/pyephem/)** — used only to *validate* the solar and eclipse geometry against independent implementations (NREL SPA, JPL DE421/DE440s, libastro)
- **[cupy](https://cupy.dev/)** — MIT, for the GPU ray march

## What is mine

The analysis, the viewer and every script in this repository: MIT (see `LICENSE`).
The baked rasters under `viewer/data/` are derivatives of the sources above and carry
their licences — dl-de/by-2-0 for anything derived from GeoSN or the tree cadastre,
ODbL for anything derived from OpenStreetMap.
