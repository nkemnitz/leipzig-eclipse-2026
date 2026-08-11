"""Turn the horizon rasters into ranked, standable viewing spots.

Ranked on MARGIN -- how far the sun clears the skyline in degrees -- not on a
binary sunlit flag, which saturates across the whole open outskirts and then ties.

A cell only counts if you could actually stand on it: nothing built or grown at
the cell itself, not in water, and either mapped public open land or within ~10 m
of a public way. Allotments, farmland and industrial land are excluded even though
they are wide open, because you have no right to stand there.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

DATA = Path("data")
to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
MARKT_X, MARKT_Y = 317035.0, 5690878.0

LANDMARKS = [
    ("Fockeberg", "Fockeberg"), ("Rosentalhügel", "Rosentalhügel"),
    ("Scherbelberg", "Scherbelberg"), ("Völkerschlachtdenkmal", "Völkerschlachtdenkmal"),
    ("Cospudener See", "Cospudener See"), ("Bistumshöhe", "Bistumshöhe"),
    ("Kulkwitzer See", "Kulkwitzer See"), ("Auensee", "Auensee"),
    ("Silbersee", "Silbersee"), ("Markkleeberger See", "Markkleeberger See"),
    ("Augustusplatz", "Augustusplatz"), ("Markt (centre)", "Markt"),
    ("Sachsenbrücke", "Sachsenbrücke"), ("Clara-Zetkin-Park", "Clara-Zetkin-Park"),
    ("Bienitz", "Bienitz"), ("Monarchenhügel", "Monarchenhügel"),
    ("agra-Park", "agra"), ("Lene-Voigt-Park", "Lene-Voigt-Park"),
    ("Panorama Tower", "Panorama Tower"), ("Zwenkauer See", "Zwenkauer See"),
    ("Schladitzer See", "Schladitzer See"), ("Elsterbecken", "Elsterbecken"),
    ("Palmengarten", "Palmengarten"), ("Wildpark", "Wildpark"),
]

# Kinds that make a sensible place NAME, best first. Without this the nearest
# named feature is routinely a recycling firm or a kindergarten.
KIND_RANK = [
    {"natural=water", "natural=peak", "leisure=park", "leisure=nature_reserve",
     "tourism=viewpoint", "place=square", "historic=monument", "historic=memorial",
     "leisure=garden", "amenity=marketplace"},
    {"leisure=common", "leisure=recreation_ground", "leisure=dog_park",
     "leisure=playground", "leisure=pitch", "landuse=forest", "landuse=cemetery",
     "landuse=village_green", "landuse=recreation_ground", "natural=wood",
     "natural=grassland", "natural=beach", "natural=sand", "natural=scrub",
     "waterway=river", "waterway=canal", "tourism=attraction", "tourism=picnic_site"},
    {"place=suburb", "place=quarter", "place=neighbourhood", "place=locality",
     "place=village", "place=town", "place=hamlet", "place=borough"},
]
ALLOWED_KINDS = set().union(*KIND_RANK)


def rasterize(polys, shape, minx, miny, res, lines=False, width=5):
    ny, nx = shape
    img = Image.new("1", (nx, ny), 0)
    d = ImageDraw.Draw(img)
    for p in polys:
        pts = [((x - minx) / res, (y - miny) / res) for x, y in p]
        if len(pts) < 2:
            continue
        if lines:
            d.line(pts, fill=1, width=width)
        else:
            d.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)  # row 0 = miny = south, matching the grid


def load_json(name):
    """Read one of the extracted OSM layer files."""
    return json.loads((DATA / name).read_text())


def main():
    h = np.load(DATA / "horizon.npz")
    labels = [str(s) for s in h["labels"]]
    hz = {lab: h[lab] for lab in labels}
    alts = {lab: float(h["alt"][i]) for i, lab in enumerate(labels)}
    azs = {lab: float(h["az"][i]) for i, lab in enumerate(labels)}
    ny, nx = hz["20:10"].shape

    hm = json.loads((DATA / "horizon_meta.json").read_text())
    minx, maxx, miny, maxy = hm["extent"]
    res = hm["res"]

    z = np.load(DATA / "sunlit.npz")
    times = [str(t) for t in z["times"]]
    obsc = z["obsc"]
    masks = np.stack([np.unpackbits(p, count=ny * nx).reshape(ny, nx).astype(bool)
                      for p in z["packed"]])
    last = np.where(masks, np.arange(len(times), dtype=np.int16)[:, None, None], -1).max(axis=0)
    weighted = (masks * obsc[:, None, None]).sum(axis=0) * 5.0

    gmeta = json.loads((DATA / "grid_meta.json").read_text())
    j0 = int((minx - gmeta["minx"]) / res)
    i0 = int((miny - gmeta["miny"]) / res)
    obj = np.load(DATA / "obj2m.npy")[i0:i0 + ny, j0:j0 + nx]
    dtm = np.load(DATA / "dtm2m.npy")[i0:i0 + ny, j0:j0 + nx]

    water = rasterize(load_json("places_water.json"), (ny, nx), minx, miny, res)
    public = rasterize(load_json("places_public.json"), (ny, nx), minx, miny, res)
    private = rasterize(load_json("places_private.json"), (ny, nx), minx, miny, res)
    paths = rasterize(load_json("places_paths.json"), (ny, nx), minx, miny, res,
                      lines=True, width=5)

    standable = (obj < 1.0) & ~water & (public | paths) & ~(private & ~public & ~paths)
    print(f"standable: {standable.sum():,} cells ({100*standable.mean():.1f}%)")
    margin = alts["20:10"] - hz["20:10"]
    print(f"of standable, sun above skyline at max: {100*(margin[standable] > 0).mean():.1f}%")

    named = [p for p in load_json("places_named.json") if p["k"] in ALLOWED_KINDS]
    px = np.array([p["x"] for p in named])
    py = np.array([p["y"] for p in named])
    prank = np.array([next((r for r, s in enumerate(KIND_RANK) if p["k"] in s), 3)
                      for p in named])
    print(f"{len(named)} place-like named features for labelling")

    def name_at(x, y):
        d = np.hypot(px - x, py - y)
        near = np.where(d < 900)[0]
        if not len(near):
            k = int(np.argmin(d))
            return named[k]["n"], named[k]["k"], float(d[k])
        cost = prank[near] + d[near] / 400.0
        k = int(near[np.argmin(cost)])
        return named[k]["n"], named[k]["k"], float(d[k])

    def report(i, j):
        x = minx + (j + 0.5) * res
        y = miny + (i + 0.5) * res
        lon, lat = to_wgs.transform(x, y)
        li = int(last[i, j])
        nm, kind, dist = name_at(x, y)
        return {
            "name": nm, "kind": kind, "name_dist_m": round(dist),
            "lat": round(lat, 6), "lon": round(lon, 6),
            "utm_x": round(x, 1), "utm_y": round(y, 1),
            "ground_m": round(float(dtm[i, j]), 1),
            "km_from_markt": round(float(np.hypot(x - MARKT_X, y - MARKT_Y)) / 1000, 2),
            "horizon_deg": {lab: round(float(hz[lab][i, j]), 2) for lab in labels},
            "margin_deg": {lab: round(alts[lab] - float(hz[lab][i, j]), 2) for lab in labels},
            "last_visible": times[li] if li >= 0 else None,
            "visible_at_max": bool(masks[times.index("20:10")][i, j]),
            "weighted_min": round(float(weighted[i, j]), 1),
        }

    # Best standable cell per 400 m block, ranked by margin at maximum eclipse.
    B = int(400 / res)
    score = np.where(standable, margin, -99.0).astype(np.float32)
    nbi, nbj = ny // B, nx // B
    blocks = score[: nbi * B, : nbj * B].reshape(nbi, B, nbj, B).transpose(0, 2, 1, 3)
    bmax = blocks.max(axis=(2, 3))
    barg = blocks.reshape(nbi, nbj, B * B).argmax(axis=2)

    cands = []
    for bi in range(nbi):
        for bj in range(nbj):
            if bmax[bi, bj] <= 0:
                continue
            off = int(barg[bi, bj])
            cands.append((float(bmax[bi, bj]), bi * B + off // B, bj * B + off % B))
    cands.sort(reverse=True)
    print(f"{len(cands)} blocks where the sun clears the skyline at maximum")

    top, seen = [], set()
    for sc, i, j in cands:
        r = report(i, j)
        if r["name"] in seen:
            continue
        seen.add(r["name"])
        top.append(r)
        if len(top) >= 30:
            break

    lm = []
    for label, needle in LANDMARKS:
        hits = [p for p in load_json("places_named.json")
                if needle.lower() in p["n"].lower()]
        if not hits:
            continue
        p = min(hits, key=lambda q: np.hypot(q["x"] - MARKT_X, q["y"] - MARKT_Y))
        j = int((p["x"] - minx) / res)
        i = int((p["y"] - miny) / res)
        if not (0 <= i < ny and 0 <= j < nx):
            lm.append({"landmark": label, "osm_name": p["n"], "outside_grid": True})
            continue
        rr = int(400 / res)
        i1, i2 = max(0, i - rr), min(ny, i + rr)
        j1, j2 = max(0, j - rr), min(nx, j + rr)
        sub = np.where(standable[i1:i2, j1:j2], margin[i1:i2, j1:j2], -99.0)
        oi, oj = np.unravel_index(int(sub.argmax()), sub.shape)
        r = report(i1 + oi, j1 + oj)
        r["landmark"] = label
        r["osm_name"] = p["n"]
        lm.append(r)

    out = {
        "sun": [{"t": t, "az": round(float(a), 2), "alt": round(float(e), 2),
                 "obsc_pct": round(float(o) * 100, 1)}
                for t, a, e, o in zip(times, z["az"], z["alt"], obsc)],
        "key_azimuths": {lab: {"az": azs[lab], "alt": alts[lab]} for lab in labels},
        "landmarks": lm, "top_spots": top,
    }
    (DATA / "ranked_spots.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    def row(r):
        m = r["margin_deg"]
        return (f"{m['20:10']:+6.2f} {m['20:30']:+6.2f} {r['horizon_deg']['20:10']:6.2f} "
                f"{str(r['last_visible']):>6s} {r['ground_m']:5.0f} {r['km_from_markt']:5.1f} "
                f" {r['lat']:.5f},{r['lon']:.5f}")

    hdr = (f"{'':30s} {'marg':>6s} {'marg':>6s} {'skyln':>6s} {'until':>6s} "
           f"{'elev':>5s} {'km':>5s}  coords")
    hdr2 = (f"{'':30s} {'20:10':>6s} {'20:30':>6s} {'20:10':>6s}")
    print("\n=== NAMED LANDMARKS (best standable cell within 400 m) ===")
    print(hdr); print(hdr2)
    for r in sorted(lm, key=lambda q: -(q.get("margin_deg", {}).get("20:10", -99))):
        if r.get("outside_grid"):
            print(f"{r['landmark']:30s} {'outside study area':>20s}")
        else:
            print(f"{r['landmark']:30s} {row(r)}")

    print("\n=== TOP RANKED PUBLIC SPOTS (by margin at maximum eclipse) ===")
    print(hdr); print(hdr2)
    for n, r in enumerate(top[:25], 1):
        print(f"{n:2d} {r['name'][:27]:27s} {row(r)}")


if __name__ == "__main__":
    main()
