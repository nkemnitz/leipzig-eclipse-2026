"""Extract buildings, trees and woodland from an OSM PBF into a local metric frame.

Output CRS is ETRS89 / UTM 33N (EPSG:25833), the official CRS for Saxony, then
shifted to a local origin so coordinates are small floats in metres.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import osmium
from pyproj import Transformer

PBF = Path("data/sachsen-latest.osm.pbf")
OUT = Path("data")

# Study area: Leipzig city plus the lakes to the south and west, which are the
# obvious candidates for an unobstructed low western horizon.
BBOX = dict(south=51.25, north=51.42, west=12.22, east=12.55)

TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
# Local origin: Leipzig Markt, snapped to a round UTM value.
ORIGIN_X, ORIGIN_Y = 315000.0, 5690000.0

METRES_PER_LEVEL = 3.2  # typical German floor-to-floor incl. slab
FEET = 0.3048

# Fallback heights (metres) when a building has no height/levels tag at all.
DEFAULT_HEIGHT = {
    "church": 20.0, "cathedral": 30.0, "chapel": 10.0,
    "garage": 3.0, "garages": 3.0, "carport": 2.5, "shed": 3.0, "hut": 3.0,
    "roof": 4.0, "greenhouse": 4.0, "kiosk": 3.5,
    "house": 9.0, "detached": 9.0, "semidetached_house": 9.0, "terrace": 11.0,
    "bungalow": 4.5, "residential": 14.0, "apartments": 16.0, "dormitory": 16.0,
    "commercial": 12.0, "retail": 8.0, "office": 18.0, "hotel": 18.0,
    "industrial": 10.0, "warehouse": 10.0, "hangar": 12.0,
    "school": 12.0, "university": 16.0, "hospital": 20.0,
    "train_station": 15.0, "transportation": 12.0,
    "civic": 14.0, "public": 14.0, "government": 16.0,
    "tower": 30.0, "water_tower": 30.0, "silo": 20.0, "storage_tank": 12.0,
    "stadium": 20.0, "sports_hall": 12.0,
    "construction": 10.0, "yes": 10.0,
}

# Woodland canopy heights. The Leipziger Auwald is genuinely tall, mature
# hardwood floodplain forest -- treating it as 10 m would badly under-shadow it.
WOOD_HEIGHT = {"wood": 25.0, "forest": 25.0, "scrub": 3.0, "orchard": 6.0}
DEFAULT_TREE_HEIGHT = 12.0  # OSM street trees rarely carry a height tag


def parse_len(value: str | None) -> float | None:
    """Parse an OSM length tag: '12', '12 m', '12.5m', "40'", '3 ft'."""
    if not value:
        return None
    v = value.strip().lower().replace(",", ".")
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*(m|metre|meter|meters|ft|feet|')?$", v)
    if not m:
        # Feet-and-inches, e.g. 40'6"
        m2 = re.match(r"^(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)?\"?$", v)
        if m2:
            ft = float(m2.group(1)) + (float(m2.group(2) or 0) / 12.0)
            return ft * FEET
        return None
    num = float(m.group(1))
    unit = m.group(2)
    if unit in ("ft", "feet", "'"):
        num *= FEET
    return num if 0.0 < num < 400.0 else None


def building_height(tags) -> tuple[float, float]:
    """Return (height, min_height) in metres."""
    h = parse_len(tags.get("height")) or parse_len(tags.get("building:height"))
    if h is None:
        levels = tags.get("building:levels") or tags.get("levels")
        try:
            n = float(str(levels).replace(",", ".")) if levels else None
        except ValueError:
            n = None
        if n is not None and 0 < n < 200:
            h = n * METRES_PER_LEVEL
            roof = parse_len(tags.get("roof:height"))
            if roof:
                h += roof
            elif tags.get("roof:shape") in ("gabled", "hipped", "pyramidal", "gambrel", "half-hipped"):
                h += 3.0  # unrecorded pitched roof
    if h is None:
        kind = tags.get("building") or tags.get("building:part") or "yes"
        h = DEFAULT_HEIGHT.get(kind, 10.0)

    mh = parse_len(tags.get("min_height"))
    if mh is None:
        lv = tags.get("building:min_level")
        try:
            mh = float(lv) * METRES_PER_LEVEL if lv else 0.0
        except (ValueError, TypeError):
            mh = 0.0
    return round(min(h, 380.0), 2), round(max(mh, 0.0), 2)


def in_bbox(lat, lon):
    return BBOX["south"] <= lat <= BBOX["north"] and BBOX["west"] <= lon <= BBOX["east"]


def project_ring(coords):
    """[(lon,lat)] -> [[x,y]] in local metres, de-duplicated and closed-open."""
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    xs, ys = TRANSFORMER.transform(lons, lats)
    ring = []
    for x, y in zip(xs, ys):
        px, py = round(x - ORIGIN_X, 2), round(y - ORIGIN_Y, 2)
        if not ring or (abs(px - ring[-1][0]) > 0.05 or abs(py - ring[-1][1]) > 0.05):
            ring.append([px, py])
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def ring_area(ring):
    a = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def main():
    if not PBF.exists():
        sys.exit(f"missing {PBF} -- download it first")

    buildings, woods, trees = [], [], []
    n_area = 0

    fp = osmium.FileProcessor(str(PBF)).with_areas()
    for obj in fp:
        if isinstance(obj, osmium.osm.Node):
            t = obj.tags
            if t.get("natural") == "tree":
                loc = obj.location
                if loc.valid() and in_bbox(loc.lat, loc.lon):
                    x, y = TRANSFORMER.transform(loc.lon, loc.lat)
                    h = parse_len(t.get("height")) or DEFAULT_TREE_HEIGHT
                    crown = parse_len(t.get("diameter_crown")) or max(3.0, h * 0.5)
                    trees.append([
                        round(x - ORIGIN_X, 1), round(y - ORIGIN_Y, 1),
                        round(h, 1), round(crown, 1),
                    ])
            continue

        if not isinstance(obj, osmium.osm.Area):
            continue

        t = obj.tags
        is_building = "building" in t or "building:part" in t
        landcover = t.get("natural") if t.get("natural") in WOOD_HEIGHT else t.get("landuse")
        is_wood = landcover in WOOD_HEIGHT
        if not (is_building or is_wood):
            continue

        n_area += 1
        try:
            outers = []
            for ring in obj.outer_rings():
                coords = [(n.lon, n.lat) for n in ring if n.location.valid()]
                if len(coords) < 3:
                    continue
                if not any(in_bbox(la, lo) for lo, la in coords):
                    continue
                r = project_ring(coords)
                if len(r) >= 3:
                    outers.append(r)
        except Exception:
            continue

        if not outers:
            continue

        if is_building:
            h, mh = building_height(t)
            for r in outers:
                if ring_area(r) < 4.0:  # drop slivers
                    continue
                buildings.append({"h": h, "m": mh, "r": r})
        else:
            hh = WOOD_HEIGHT[landcover]
            for r in outers:
                if ring_area(r) < 200.0:
                    continue
                woods.append({"h": hh, "r": r})

    meta = {
        "crs": "EPSG:25833",
        "origin": [ORIGIN_X, ORIGIN_Y],
        "bbox_wgs84": BBOX,
        "counts": {"buildings": len(buildings), "woods": len(woods), "trees": len(trees)},
        "metres_per_level": METRES_PER_LEVEL,
        "default_tree_height": DEFAULT_TREE_HEIGHT,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "buildings.json").write_text(json.dumps(buildings, separators=(",", ":")))
    (OUT / "woods.json").write_text(json.dumps(woods, separators=(",", ":")))
    (OUT / "trees.json").write_text(json.dumps(trees, separators=(",", ":")))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"areas scanned: {n_area}")
    for k, v in meta["counts"].items():
        print(f"  {k:10s} {v:8d}")
    hs = sorted(b["h"] for b in buildings)
    if hs:
        print(f"  height p50={hs[len(hs)//2]:.1f} p95={hs[int(len(hs)*0.95)]:.1f} max={hs[-1]:.1f}")


if __name__ == "__main__":
    main()
