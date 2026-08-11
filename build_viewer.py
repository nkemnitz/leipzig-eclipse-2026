"""Bake the computed rasters into textures + metadata for the browser viewer.

Everything the page draws comes from the same DSM and the same ray-march that
produced the recommendations, so the 3-D view and the numbers cannot disagree.

Outputs (viewer/data/):
  height.png    16-bit grayscale, the DOM1 surface -- displaces the mesh
  ortho.jpg     GeoSN 20 cm orthophoto mosaic, downsampled
  ground.png    RGBA, 17 sunlit bitplanes for an observer standing on the ground
  surface.png   RGBA, 17 sunlit bitplanes for the surface itself (3-D shading)
  info.png      RGBA: margin@20:10, margin@20:30, horizon@20:10, standable
  meta.json     extents, sun track, ranked spots, per-spot skyline profiles
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import io
import json
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

import profiles

Image.MAX_IMAGE_PIXELS = None
DATA = Path("data")
OUT = Path("viewer/data")

DOWN = 4                      # 2 m grid -> 8 m textures
# Orthophoto: GeoSN DOP20 is 20 cm at source, so viewer sharpness is purely a
# question of how finely we bake it. 1 km WMS tiles at 512 px give 2 m/px over the
# whole 16x18 km, which is 8192x9216 -- past the max texture size on plenty of
# GPUs, so it is stored as a 2x2 grid of quadrant textures and selected in-shader.
ORTHO_TILE_KM = 1
ORTHO_PX_PER_TILE = 512       # -> 2 m/px
ORTHO_QUADS = 2
WMS = "https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest"

H0, HSCALE = 80.0, 50.0       # height encoding: (h - H0) * HSCALE -> uint16
MARGIN_LO, MARGIN_HI = -20.0, 5.0
# Skyline can sit BELOW the horizontal on a hill, so the range starts negative --
# clipping it at 0 made the panel's skyline and margin fail to reconcile.
HORIZON_LO, HORIZON_HI = -5.0, 60.0

to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)


def block_reduce(a, f, how="max"):
    ny, nx = a.shape
    a = a[: ny // f * f, : nx // f * f].reshape(ny // f, f, nx // f, f)
    if how == "max":
        return a.max(axis=(1, 3))
    if how == "min":
        return a.min(axis=(1, 3))
    if how == "mean":
        return a.mean(axis=(1, 3))
    return a.mean(axis=(1, 3)) > 0.5   # majority


def fetch_ortho(minx, maxx, miny, maxy):
    """Stitch a DOP20 mosaic from tiled WMS GetMap requests."""
    nx = (maxx - minx) // (ORTHO_TILE_KM * 1000)
    ny = (maxy - miny) // (ORTHO_TILE_KM * 1000)
    W, H = nx * ORTHO_PX_PER_TILE, ny * ORTHO_PX_PER_TILE
    print(f"ortho: {nx}x{ny} WMS tiles -> {W}x{H} px ({(maxx-minx)/W:.1f} m/px)")
    canvas = Image.new("RGB", (W, H), (60, 65, 60))

    def one(args):
        ix, iy = args
        x0 = minx + ix * ORTHO_TILE_KM * 1000
        y0 = miny + iy * ORTHO_TILE_KM * 1000
        x1, y1 = x0 + ORTHO_TILE_KM * 1000, y0 + ORTHO_TILE_KM * 1000
        url = (f"{WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=sn_dop_020"
               f"&STYLES=&CRS=EPSG:25833&BBOX={x0},{y0},{x1},{y1}"
               f"&WIDTH={ORTHO_PX_PER_TILE}&HEIGHT={ORTHO_PX_PER_TILE}&FORMAT=image/jpeg")
        for _ in range(3):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                    return ix, iy, Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                pass
        return ix, iy, None

    done = 0
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        for ix, iy, img in ex.map(one, [(a, b) for a in range(nx) for b in range(ny)]):
            done += 1
            if img is None:
                continue
            # WMS y increases north; image rows increase downward
            canvas.paste(img, (ix * ORTHO_PX_PER_TILE,
                               (ny - 1 - iy) * ORTHO_PX_PER_TILE))
    print(f"  fetched {done} tiles")
    # The mosaic is assembled north-up (WMS tiles are north-up and are pasted
    # top-down), but every analysis raster in this project is written south-up
    # (row 0 = miny). Flip so all textures share one row convention -- otherwise
    # the imagery is mirrored against the terrain and the shadow masks.
    return canvas.transpose(Image.FLIP_TOP_BOTTOM)


def pack_bitplanes(masks):
    """(T, ny, nx) bool -> RGB uint8 image with plane k in bit k of R,G,B.

    Deliberately RGB, not RGBA, and capped at 24 planes. Canvas getImageData
    premultiplies, so any pixel whose alpha is 0 comes back with R=G=B=0 -- and
    with only 17 planes the alpha byte is always 0, which silently wiped every
    sunlit flag on load. Never store data in the alpha channel of a texture that
    is read back through a canvas.
    """
    t, ny, nx = masks.shape
    assert t <= 24, "only 24 bitplanes fit in RGB"
    acc = np.zeros((ny, nx), dtype=np.uint32)
    for k in range(t):
        acc |= (masks[k].astype(np.uint32) << np.uint32(k))
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for b in range(3):
        rgb[:, :, b] = ((acc >> np.uint32(8 * b)) & np.uint32(0xFF)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gmeta = json.loads((DATA / "grid_meta.json").read_text())
    res = gmeta["res"]
    minx, maxx = gmeta["out_minx"], gmeta["out_maxx"]
    miny, maxy = gmeta["out_miny"], gmeta["out_maxy"]

    j0 = int((minx - gmeta["minx"]) / res)
    i0 = int((miny - gmeta["miny"]) / res)
    ny_o = int((maxy - miny) / res)
    nx_o = int((maxx - minx) / res)

    dsm_full = np.load(DATA / "dsm2m.npy", mmap_mode="r")
    dtm_full = np.load(DATA / "dtm2m.npy", mmap_mode="r")
    dsm = np.asarray(dsm_full[i0:i0 + ny_o, j0:j0 + nx_o], dtype=np.float32)
    dtm = np.asarray(dtm_full[i0:i0 + ny_o, j0:j0 + nx_o], dtype=np.float32)

    # --- height texture (max-reduced, matching the shadow model's convention) ---
    # Encoded as RGB8 hi/lo bytes, NOT a 16-bit PNG: browsers decode 16-bit PNGs
    # down to 8 bits when drawn to a canvas, which would quantise the terrain to
    # ~1 m steps and corrupt the click-to-query readout.
    # ONE display surface, not two: bare earth wherever a building stands (so the
    # LoD2 solids sit on real ground) and the DOM1 surface everywhere else (so
    # tree canopy — the thing that actually blocks the sun here — is still there).
    # A separate canopy mesh masked per-fragment over a coarser grid left floating
    # slabs wherever a quad straddled a treeline.
    obj_full = np.asarray(np.load(DATA / "obj2m.npy", mmap_mode="r")
                          [i0:i0 + ny_o, j0:j0 + nx_o])
    from rank_spots import rasterize as _rast, load_json as _lj
    om = json.loads((DATA / "meta.json").read_text())
    bx, by = om["origin"]
    bldg = _rast([[[x + bx, y + by] for x, y in p["r"]]
                  for p in _lj("buildings.json")], (ny_o, nx_o), minx, miny, res)
    # Dilate: OSM footprints and LoD2 solids do not agree cell-for-cell, and an
    # undilated mask leaves a collar of roof-height ground around every building.
    from scipy.ndimage import binary_dilation
    bldg_d = binary_dilation(bldg, iterations=4)
    # Ground mesh is bare earth: it keeps the orthophoto crisp and lets the LoD2
    # solids sit on real ground. Draping the DOM1 surface here instead turns every
    # tree into a spike and smears the imagery over it.
    hs = block_reduce(dtm, DOWN, "min")
    enc = np.clip((hs - H0) * HSCALE, 0, 65535).astype(np.uint16)
    th, tw = enc.shape

    # --- canopy blanket -------------------------------------------------------
    # Drawn as its own surface so the ground underneath stays clean. The height is
    # deliberately a SMOOTHED, DILATED blanket rather than the raw DOM1: a mask
    # applied per-fragment over a coarser mesh makes any quad that straddles a
    # treeline slant from treetop down to bare earth, and those wedges survive as
    # floating slabs. Dilating the height past the mask, then eroding the mask,
    # guarantees every drawn fragment sits well inside the flat part of the blanket.
    from scipy.ndimage import binary_erosion, maximum_filter, uniform_filter
    veg = (obj_full > 2.0) & ~bldg_d
    veg_small = block_reduce(veg.astype(np.float32), DOWN, "mean") > 0.35
    dsm_small = block_reduce(dsm, DOWN, "max")
    base = np.where(veg_small, dsm_small, hs).astype(np.float32)
    blanket = uniform_filter(maximum_filter(base, size=5), size=5)
    cmask = binary_erosion(veg_small, iterations=1)
    cenc = np.clip((blanket - H0) * HSCALE, 0, 65535).astype(np.uint16)
    crgb = np.zeros((th, tw, 3), dtype=np.uint8)
    crgb[:, :, 0] = (cenc >> 8).astype(np.uint8)
    crgb[:, :, 1] = (cenc & 0xFF).astype(np.uint8)
    crgb[:, :, 2] = cmask * 255
    Image.fromarray(crgb, mode="RGB").save(OUT / "canopy.png", optimize=True)
    print(f"canopy.png {tw}x{th}  cover {100*cmask.mean():.0f}%  "
          f"mean height above ground {np.mean(blanket[cmask] - hs[cmask]):.1f} m")

    rgb = np.zeros((th, tw, 3), dtype=np.uint8)
    rgb[:, :, 0] = (enc >> 8).astype(np.uint8)
    rgb[:, :, 1] = (enc & 0xFF).astype(np.uint8)
    rgb[:, :, 2] = (block_reduce(veg.astype(np.float32), DOWN, "mean") > 0.4) * 255
    Image.fromarray(rgb, mode="RGB").save(OUT / "height.png", optimize=True)
    back = ((rgb[:, :, 0].astype(np.uint32) << 8) | rgb[:, :, 1]) / HSCALE + H0
    print(f"height.png {tw}x{th}  {hs.min():.1f}..{hs.max():.1f} m  "
          f"roundtrip max err {np.abs(back - hs).max():.4f} m  "
          f"veg {100*(rgb[:,:,2]>0).mean():.0f}%")

    # Bare-earth terrain for the ground mesh, reduced with MIN so the drawn ground
    # never rises above true ground inside a block -- that is what buries LoD2
    # buildings. Costs a few decimetres of ridge height on steep spoil heaps.
    ts = block_reduce(dtm, DOWN, "min")
    tenc = np.clip((ts - H0) * HSCALE, 0, 65535).astype(np.uint16)
    trgb = np.zeros((th, tw, 3), dtype=np.uint8)
    trgb[:, :, 0] = (tenc >> 8).astype(np.uint8)
    trgb[:, :, 1] = (tenc & 0xFF).astype(np.uint8)
    Image.fromarray(trgb, mode="RGB").save(OUT / "terrain.png", optimize=True)
    print(f"terrain.png {tw}x{th}  {ts.min():.1f}..{ts.max():.1f} m")

    terr = block_reduce(dtm, DOWN, "mean").astype(np.float32)
    np.save(OUT / "terrain_small.npy", terr)

    # --- sunlit bitplanes ---
    z = np.load(DATA / "sunlit.npz")
    times = [str(t) for t in z["times"]]
    az, alt, obsc = z["az"], z["alt"], z["obsc"]

    def unpack(key):
        return np.stack([
            np.unpackbits(p, count=ny_o * nx_o).reshape(ny_o, nx_o).astype(bool)
            for p in z[key]])

    ground = unpack("packed")
    surface = unpack("packed_surface")
    # Ground mask answers "could I stand somewhere in this 8 m square and see the
    # sun", so it reduces with ANY -- the same best-cell convention as the margin
    # channel. Majority here made a lit hilltop ringed by trees report "no".
    g_small = np.stack([block_reduce(m.astype(np.float32), DOWN, "max") > 0.5
                        for m in ground])
    # Surface mask drives 3-D shading, where majority looks right.
    s_small = np.stack([block_reduce(m.astype(np.float32), DOWN, "mean") > 0.5
                        for m in surface])
    pack_bitplanes(g_small).save(OUT / "ground.png", optimize=True)
    pack_bitplanes(s_small).save(OUT / "surface.png", optimize=True)
    print(f"ground.png/surface.png {tw}x{th}  {len(times)} planes")

    # --- info texture: margins, skyline height, standable ---
    h = np.load(DATA / "horizon.npz")
    labels = [str(s) for s in h["labels"]]
    alts = {lab: float(h["alt"][i]) for i, lab in enumerate(labels)}
    m10 = alts["20:10"] - h["20:10"]
    m30 = alts["20:30"] - h["20:30"]
    hz10 = h["20:10"]

    from rank_spots import rasterize, load_json
    water = rasterize(load_json("places_water.json"), (ny_o, nx_o), minx, miny, res)
    public = rasterize(load_json("places_public.json"), (ny_o, nx_o), minx, miny, res)
    paths = rasterize(load_json("places_paths.json"), (ny_o, nx_o), minx, miny, res,
                      lines=True, width=5)
    obj = np.asarray(np.load(DATA / "obj2m.npy", mmap_mode="r")[i0:i0 + ny_o, j0:j0 + nx_o])
    standable = (obj < 1.0) & ~water & (public | paths)

    def q(a, lo, hi):
        return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    # Margin uses max over the block (the best cell you could stand on); the
    # skyline channel must therefore use MIN, not mean, or the panel reports a
    # skyline and a margin that cannot both be true of the same square metre.
    info = np.zeros((th, tw, 3), dtype=np.uint8)
    info[:, :, 0] = q(block_reduce(m10, DOWN, "max"), MARGIN_LO, MARGIN_HI)
    info[:, :, 1] = q(block_reduce(m30, DOWN, "max"), MARGIN_LO, MARGIN_HI)
    info[:, :, 2] = q(block_reduce(hz10, DOWN, "min"), HORIZON_LO, HORIZON_HI)
    Image.fromarray(info, mode="RGB").save(OUT / "info.png", optimize=True)
    # "standable" rides in terrain.png's spare blue channel -- see pack_bitplanes
    # for why it must not live in an alpha channel.
    stand_small = (block_reduce(standable.astype(np.float32), DOWN, "mean") > 0.25)
    trgb[:, :, 2] = stand_small * 255
    Image.fromarray(trgb, mode="RGB").save(OUT / "terrain.png", optimize=True)
    print(f"info.png {tw}x{th}  standable {100*stand_small.mean():.0f}%")

    # --- orthophoto ---
    ortho_path = OUT / "ortho.jpg"
    quad_paths = [OUT / f"ortho_{qx}_{qy}.jpg"
                  for qy in range(ORTHO_QUADS) for qx in range(ORTHO_QUADS)]
    if not all(p.exists() for p in quad_paths) or not ortho_path.exists():
        full = fetch_ortho(minx, maxx, miny, maxy)
        W, H = full.size
        qw, qh = W // ORTHO_QUADS, H // ORTHO_QUADS
        for qy in range(ORTHO_QUADS):
            for qx in range(ORTHO_QUADS):
                # row 0 of the flipped mosaic is SOUTH, so qy indexes south->north
                box = (qx * qw, qy * qh, (qx + 1) * qw, (qy + 1) * qh)
                full.crop(box).save(OUT / f"ortho_{qx}_{qy}.jpg",
                                    quality=84, optimize=True, progressive=True)
        # keep a single low-res copy for the alignment test and the inlined build
        full.resize((W // 2, H // 2), Image.LANCZOS).save(
            ortho_path, quality=80, optimize=True, progressive=True)
        del full
    qsz = Image.open(quad_paths[0]).size
    tot = sum(p.stat().st_size for p in quad_paths) / 1e6
    print(f"ortho quadrants {ORTHO_QUADS}x{ORTHO_QUADS} of {qsz[0]}x{qsz[1]} px "
          f"= {(maxx-minx)/(qsz[0]*ORTHO_QUADS):.2f} m/px, {tot:.1f} MB total")

    # Alignment test: the imagery must agree with the independently-derived OSM
    # water mask. If the ortho row convention were flipped, "water" pixels would
    # land on rooftops and the blue-minus-red contrast would collapse.
    oi = np.asarray(Image.open(ortho_path).convert("RGB").resize((tw, th),
                                                                 Image.BILINEAR),
                    dtype=np.float32)
    wsmall = block_reduce(water.astype(np.float32), DOWN, "mean") > 0.8
    if wsmall.sum() > 1000:
        br = oi[:, :, 2] - oi[:, :, 0]
        wet, dry = float(br[wsmall].mean()), float(br[~wsmall].mean())
        # Measured: correct orientation gives a gap of ~37, a vertically flipped
        # ortho still gives ~6 (Leipzig's lakes are roughly N-S symmetric), so the
        # threshold has to sit well above that to actually discriminate.
        gap = wet - dry
        print(f"ortho alignment: blue-red over OSM water {wet:+.1f} vs land {dry:+.1f} "
              f"(gap {gap:.1f}) -> {'OK' if gap > 15 else 'MISALIGNED'}")
        if gap <= 15:
            raise SystemExit("ortho does not align with the water mask - check row order")

    # --- skyline profiles for the recommended spots ---
    ranked = json.loads((DATA / "ranked_spots.json").read_text())
    spots = []
    seen = set()
    for src, tag in ((ranked.get("landmarks", []), "landmark"),
                     (ranked.get("top_spots", []), "ranked")):
        for r in src:
            if r.get("outside_grid") or "lat" not in r:
                continue
            key = (round(r["lat"], 4), round(r["lon"], 4))
            if key in seen:
                continue
            seen.add(key)
            spots.append({**r, "tag": tag,
                          "label": r.get("landmark") or r.get("name", "?")})

    print(f"computing skyline profiles for {len(spots)} spots ...")
    gm = {"res": res, "minx": gmeta["minx"], "miny": gmeta["miny"]}
    dsm_all = np.load(DATA / "dsm2m.npy", mmap_mode="r")
    for s in spots:
        if "utm_x" not in s or "utm_y" not in s:
            continue
        x, y = s["utm_x"], s["utm_y"]
        try:
            azs, elev = profiles.profile(dsm_all, gm, x, y,
                                         ground_z=s["ground_m"], n_az=181)
        except ValueError:
            continue
        s["profile_az0"], s["profile_az1"] = float(azs[0]), float(azs[-1])
        s["profile"] = [round(float(v), 2) for v in elev]

    meta = {
        "extent": [minx, maxx, miny, maxy],
        "tex": [tw, th],
        "res_m": res * DOWN,
        "height": {"h0": H0, "scale": HSCALE},
        "info": {"margin_lo": MARGIN_LO, "margin_hi": MARGIN_HI,
                 "horizon_lo": HORIZON_LO, "horizon_hi": HORIZON_HI},
        "times": times,
        "sun": [{"t": t, "az": round(float(a), 3), "alt": round(float(e), 3),
                 "obsc": round(float(o) * 100, 1)}
                for t, a, e, o in zip(times, az, alt, obsc)],
        "eclipse": {"c1": "19:17:30", "max": "20:10:30", "sunset": "20:38:27",
                    "max_obscuration": 86.0},
        "spots": spots,
        "attribution": "Elevation & imagery: GeoSN (dl-de/by-2-0). Vector data: OpenStreetMap contributors (ODbL).",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, separators=(",", ":"), ensure_ascii=False))
    print(f"meta.json {(OUT/'meta.json').stat().st_size/1e6:.2f} MB, {len(spots)} spots")

    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    print(f"\nviewer/data total {total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
