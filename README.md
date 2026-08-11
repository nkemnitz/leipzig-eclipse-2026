# Where can you see the eclipsed sun from, in Leipzig?

**[▶ Open the viewer](https://nkemnitz.github.io/leipzig-eclipse-2026/)**

On 12 August 2026 a partial solar eclipse reaches maximum over Leipzig at
**20:10:35 CEST**, with the sun **3.4° above the horizon** and **86% obscured**.
Sunset follows at 20:38:26, with the sun still about a third covered.

At 3.4° the question is not *when* but *where*: the sight line to the sun is only
4.7 m above the ground at 50 m distance and 7.6 m at 100 m. Practically everything
in a city is in the way. This computes, for every 2 m of ground in a 16 × 18 km
area, whether the sun is actually visible — from laser-scanned geometry, not from
a list someone published.

## The thing that makes this different

A normal viewshed treats a tree as an opaque column from the ground to the crown
top. That is wrong at a low sun, and wrongly *pessimistic*: Leipzig's median crown
base is 10.7 m, so a 3.4° sight line passes **underneath** the canopy. The binary
model reported the Fockeberg — a hill built from war rubble specifically to stand
on — as 1% open.

So the occlusion model is a **voxel grid**: 6.3 billion laser returns resolved into
32 height bands of 2 m above ground, at 2 m horizontally. A ray marched through it
meets material *at its own height*, which splits the answer into three honest
classes instead of two:

| | meaning |
|---|---|
| **clear** | nothing intersects the sight line |
| **blocked** | terrain or masonry — no amount of walking helps |
| **through canopy** | only vegetation is in the way, graded by how much |

The third class is not a yes/no. A line clipping 4 m of twigs is not a line
ploughing 60 m through floodplain forest, so it carries the path length through
occupied canopy voxels, converted to transmittance with Beer–Lambert. Because
transmittance *is* the probability that one sight line misses every leaf, the map
reads directly as "what share of sight lines from here reach the sun".

At maximum eclipse: **21.4% clear, 38.3% blocked, 40.2% through canopy** — that
last 40% is the population the binary model was throwing away.

### The extinction coefficient is measured, not assumed

Over 76,715 woodland cells where the porosity measurement and the voxel column
overlap, 49.4% of laser pulses reached the ground through a mean 10.0 m of occupied
voxel, giving **k = 0.071 per metre** (per-cell median 0.073, so the aggregate is
not carried by outliers). That flight was in **January**, leaf-off, so this is a
hard *upper bound* on how much sun gets through: anywhere dark at k = 0.071 is
genuinely dark. The map uses a leaf-on estimate of twice that, and the ranking
reports both so the disagreement is visible rather than hidden.

## Where to actually go

Share of sight lines reaching the sun at 20:10, over standable public ground:

| | | |
|---|---|---|
| Silbersee | 99.6% | |
| Auensee | 94.6% | |
| Rosentalhügel | 89.2% | 1 km from the Markt |
| Völkerschlachtdenkmal | 75.8% | |
| Lene-Voigt-Park | 60.4% | |
| **Fockeberg** | **21.2%** | the intuitive choice, and mediocre |
| Markt / Augustusplatz | ~0% | 100% walled, as you would expect |

The result I did not expect: there is **wholly open standing ground 1.0 km from the
Markt**, in the Rosental. The binary model saw none of it.

Clara-Zetkin-Park is the clearest illustration of why one number per place is the
wrong shape for the answer: its centroid scores 1% (the median sight line ploughs
73 m through Auwald) while its open meadow near the Parkbühne scores 95%.

## Honest limits

- **k is a leaf-off measurement doing leaf-on work.** The 2× scaling is literature,
  not measurement. It is the largest uncertainty here.
- **Beyond ~1 km the voxel grid stops helping** — the ray climbs past its 64 m
  ceiling — and the far field falls back to the max-pooled DSM march. That is not a
  fudge: past that height only buildings and terrain are tall enough to matter, and
  for those the opaque-column model is exactly right.
- **The 2 m band quantisation over-blocks slightly.** Material fills its whole band,
  so a voxel column top sits at or above the DSM in 99.4% of occupied cells. Safe
  direction, and bounded.
- **Landmark scores describe a neighbourhood.** Where a named place has little
  standable ground, the search radius grows until it finds some — Fockeberg's 21% is
  measured 437 m from the summit. Each row's tooltip carries the walk distance.
- **Buildings and the tree cadastre are drawn for orientation only.** The sun answer
  comes from the laser data alone.

## How it is built

```
download_geosn.py   DOM1 + DGM1 1 m tiles          download_lsc.py   26 GB laser point cloud
build_dsm.py        mosaic to 2 m rasters          build_voxels.py   -> 32-band occupancy grid (~25 min)
extract_osm.py      buildings, woods, water        build_solid.py    which voxels are masonry
extract_places.py   standable ground, names
                                                   voxel_march.py    3-class GPU march, 17 timestamps (70 s)
solar.py            NOAA solar position            rank_voxel.py     ranked spots
eclipse.py          circumstances via ephem        patch_spots.py    viewer spot list
viewshed.py         DSM ray march (far field)
                                                   build_viewer.py   baked textures
build_lod2.py       CityGML -> meshes              build_voxel_tex.py  3-class textures
build_detail.py     1 m streamed tiles             build_artifact.py single-file page
export_voxels.py    voxel box -> PLY for MeshLab
```

Tests: `test_viewshed.py` (ray-march geometry, including the +2.06° UTM meridian
convergence that walks a 5 km sight line 180 m sideways if ignored),
`test_voxel_march.py` (containment invariant against the DSM — the voxel model may
only ever *open* cells, never close them), `test_shaders.py` (three silent WebGL
failure modes), `test_orientation.py`, `verify_viewer.py`, `check_detail.py`.

Run the viewer locally with any static server:

```bash
cd viewer && python -m http.server 8000
```

## Solar geometry

Positions come from the NOAA/Meeus algorithm in `solar.py`, validated to 0.005°
against NREL SPA (pvlib), libastro (pyephem) and JPL DE421/DE440s (Skyfield).
Eclipse circumstances come from circle–circle overlap of the topocentric solar and
lunar disks. Earth curvature (d²/2R) and refraction (R_eff = 7/6 R) are both
included — at 5 km the curvature drop exceeds eye height, which matters at 3.4°.

## Credits

All input data is open. See **[CREDITS.md](CREDITS.md)** — GeoSN (dl-de/by-2-0),
OpenStreetMap (ODbL), Stadt Leipzig tree cadastre (dl-de/by-2-0). Code is MIT.
