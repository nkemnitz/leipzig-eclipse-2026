"""Extract the OSM layers needed to turn a sunlit raster into usable advice:
where is water, where is publicly walkable open ground, and what is this place called.

Coordinates are absolute ETRS89/UTM33N metres (EPSG:25833), matching the raster grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import osmium
from pyproj import Transformer

PBF = Path("data/sachsen-latest.osm.pbf")
OUT = Path("data")

# Output raster extent, plus a margin so features are not clipped mid-polygon.
MINX, MAXX = 306000, 326000
MINY, MAXY = 5682000, 5698000

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
LON0, LAT0 = to_wgs.transform(MINX, MINY)
LON1, LAT1 = to_wgs.transform(MAXX, MAXY)

WATER_TAGS = [("natural", "water"), ("landuse", "reservoir"), ("landuse", "basin"),
              ("waterway", "riverbank"), ("natural", "wetland")]

# Land you can walk onto without trespassing.
PUBLIC_OPEN = {
    "leisure": {"park", "garden", "common", "recreation_ground", "dog_park",
                "nature_reserve", "playground", "pitch"},
    "landuse": {"grass", "meadow", "village_green", "recreation_ground", "cemetery"},
    "natural": {"grassland", "heath", "sand", "beach", "shingle", "bare_rock"},
    "place": {"square"},
    "amenity": {"marketplace"},
    "tourism": {"viewpoint", "picnic_site"},
}

# Open, and it will show up as "sunlit", but you have no right to stand there.
# Kept separate so it can be excluded from recommendations rather than silently
# ranked first -- the western outskirts are wall-to-wall allotments and farmland.
PRIVATE_OPEN = {
    "landuse": {"allotments", "farmland", "greenfield", "brownfield", "orchard",
                "vineyard", "industrial", "commercial", "quarry", "landfill",
                "construction", "military", "railway"},
    "leisure": {"golf_course", "stadium", "track", "sports_centre", "marina"},
}

WALKABLE_HIGHWAY = {"footway", "path", "cycleway", "pedestrian", "track", "steps",
                    "living_street", "residential", "service", "unclassified",
                    "tertiary", "secondary", "primary", "road"}

NAME_KEYS = ("place", "leisure", "tourism", "natural", "amenity", "man_made",
             "historic", "waterway", "landuse", "building")


def in_box(lat, lon):
    return LAT0 <= lat <= LAT1 and LON0 <= lon <= LON1


def ring_xy(ring):
    pts = [(n.lon, n.lat) for n in ring if n.location.valid()]
    if len(pts) < 3:
        return None
    if not any(in_box(la, lo) for lo, la in pts):
        return None
    xs, ys = to_utm.transform([p[0] for p in pts], [p[1] for p in pts])
    return [[round(x, 1), round(y, 1)] for x, y in zip(xs, ys)]


def main():
    water, public, private, paths, named = [], [], [], [], []

    fp = osmium.FileProcessor(str(PBF)).with_areas()
    for obj in fp:
        if isinstance(obj, osmium.osm.Node):
            t = obj.tags
            if "name" not in t:
                continue
            loc = obj.location
            if not loc.valid() or not in_box(loc.lat, loc.lon):
                continue
            kind = next(((k, t[k]) for k in NAME_KEYS if k in t), None)
            if kind:
                x, y = to_utm.transform(loc.lon, loc.lat)
                named.append({"n": t["name"], "k": f"{kind[0]}={kind[1]}",
                              "x": round(x, 1), "y": round(y, 1)})
            continue

        if isinstance(obj, osmium.osm.Way):
            t = obj.tags
            if t.get("highway") in WALKABLE_HIGHWAY:
                try:
                    pts = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
                except Exception:
                    continue
                if len(pts) < 2 or not any(in_box(la, lo) for lo, la in pts):
                    continue
                xs, ys = to_utm.transform([p[0] for p in pts], [p[1] for p in pts])
                paths.append([[round(x, 1), round(y, 1)] for x, y in zip(xs, ys)])
            continue

        if not isinstance(obj, osmium.osm.Area):
            continue
        t = obj.tags
        is_water = any(t.get(k) == v for k, v in WATER_TAGS)
        is_public = any(t.get(k) in vs for k, vs in PUBLIC_OPEN.items())
        is_private = any(t.get(k) in vs for k, vs in PRIVATE_OPEN.items())
        has_name = "name" in t
        if not (is_water or is_public or is_private or has_name):
            continue
        try:
            rings = [r for r in (ring_xy(r) for r in obj.outer_rings()) if r]
        except Exception:
            continue
        if not rings:
            continue
        if is_water:
            water.extend(rings)
        if is_public:
            public.extend(rings)
        if is_private:
            private.extend(rings)
        if has_name:
            kind = next(((k, t[k]) for k in NAME_KEYS if k in t), None)
            if kind:
                big = max(rings, key=len)
                cx = sum(p[0] for p in big) / len(big)
                cy = sum(p[1] for p in big) / len(big)
                named.append({"n": t["name"], "k": f"{kind[0]}={kind[1]}",
                              "x": round(cx, 1), "y": round(cy, 1)})

    for name, obj in (("water", water), ("public", public), ("private", private),
                      ("paths", paths), ("named", named)):
        (OUT / f"places_{name}.json").write_text(json.dumps(obj, separators=(",", ":")))
        print(f"  {name:10s} {len(obj):7d}")
    (OUT / "places_meta.json").write_text(json.dumps({
        "crs": "EPSG:25833", "minx": MINX, "maxx": MAXX, "miny": MINY, "maxy": MAXY,
    }, indent=2))


if __name__ == "__main__":
    main()
