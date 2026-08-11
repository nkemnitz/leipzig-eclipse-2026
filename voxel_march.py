"""Three-class line of sight to the sun, marched through the voxel grid on the GPU.

The DSM march can only answer yes/no, and it answers it with the wrong model: it
treats a tree as an opaque column from the ground to the crown top. At a 3.4 deg
sun the sight line is 4.7 m up at 50 m and 7.6 m at 100 m, while the median crown
base in Leipzig's woodland is 10.7 m -- so the line passes UNDER the crowns and
the column model wrongly blocks it. That single error is why the old ranking put
the Fockeberg at 1% open.

Marching occupancy resolved in height fixes it and, in doing so, splits the answer
into three honest classes:

    0  CLEAR      nothing intersects the sight line
    1  BLOCKED    terrain, a building, or a far-field obstruction is in the way
    2  THROUGH CANOPY  the only thing in the way is vegetation

Class 2 is not a yes/no -- a sight line that clips 4 m of twigs is not the same as
one that ploughs 60 m through a wood. So it carries the path length through
occupied canopy voxels, converted to a transmittance with Beer-Lambert.

Near field vs far field. The voxel grid is 32 bands of 2 m above ground, so it
stops being useful once the ray climbs past 64 m AGL, at d = 64/tan(alt). Beyond
that only buildings and terrain are tall enough to matter, and for those the
opaque-column model is exactly right -- so the far field falls back to the
max-pooled DSM march, which is the same code path the binary map has always used.
The handover distance is therefore not a fudge: it is the distance past which the
two models agree by construction.

    python voxel_march.py            # all 17 timestamps
    python voxel_march.py --at 20:10 # one, with a breakdown
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from viewshed import BANDS, DMAX, EYE_H, R_EFF, grid_convergence_deg, ray_offsets

DATA = Path("data")
OUT = Path("data/voxout")

VOX_DMAX = 1800.0      # the voxel grid only carries a 2 km margin
NB_TOP = 64.0          # 32 bands x 2 m: above this the voxel grid says nothing

# Extinction per metre of occupied canopy voxel, Beer-Lambert.
#
# K_LEAFOFF is MEASURED, not assumed: over 76,715 woodland cells in the Fockeberg
# tile, 49.4% of laser pulses reached the ground through a mean 10.0 m of occupied
# voxel, giving -ln(0.494)/10.0 = 0.071 (per-cell median 0.073, so the aggregate is
# not being carried by outliers). That flight was in January. Leaf-on canopy is
# roughly twice as extinctive, hence K_AUGUST. Reporting both is the point: the
# leaf-off number is a hard UPPER BOUND on how much sun gets through, so anywhere
# that is dark even at K_LEAFOFF is genuinely dark.
K_LEAFOFF = 0.0708
K_AUGUST = 0.1416

CEST = _dt.timezone(_dt.timedelta(hours=2))


def _kernels(cp):
    """One fused kernel per march; fusing avoids ~2 GB of temporaries per step."""
    near = cp.ElementwiseKernel(
        in_params=("uint32 occ_s, bool solid_s, float32 gnd_s, float32 gnd_p, "
                   "float32 bias, float32 seg, float32 vh, int32 nb"),
        out_params="bool blocked, float32 vegpath",
        operation=r"""
            if (!blocked) {
                // height of the sight line above the ground under the sample.
                // bias = d*tan(alt) + d^2/(2R_eff), i.e. rise plus earth curvature.
                float hr = gnd_p + %f + bias - gnd_s;
                int b = (int)floorf(hr / vh);
                if (b < 0) {
                    blocked = true;                 // the ray is under the terrain
                } else if (b < nb) {
                    if ((occ_s >> b) & 1u) {
                        if (solid_s) blocked = true;  // roof or wall
                        else vegpath += seg;          // crown: attenuates, not blocks
                    }
                }
            }
        """ % EYE_H,
        name="voxel_step",
    )
    far = cp.ElementwiseKernel(
        in_params="float32 h_s, float32 bias",
        out_params="float32 best",
        operation="best = fmaxf(best, h_s - bias);",
        name="dsm_step",
    )
    return near, far


def maxpool_upsample_gpu(cp, a, f):
    if f == 1:
        return a
    ny, nx = a.shape
    py, px = (-ny) % f, (-nx) % f
    if py or px:
        a = cp.pad(a, ((0, py), (0, px)), mode="edge")
    h, w = a.shape
    pooled = a.reshape(h // f, f, w // f, f).max(axis=(1, 3))
    return cp.repeat(cp.repeat(pooled, f, axis=0), f, axis=1)[:ny, :nx]


class Marcher:
    def __init__(self, verbose=True):
        import cupy as cp
        self.cp = cp
        self.near_k, self.far_k = _kernels(cp)

        self.gmeta = json.loads((DATA / "grid_meta.json").read_text())
        self.vmeta = json.loads((DATA / "voxels/meta.json").read_text())
        g, v = self.gmeta, self.vmeta
        self.res = float(g["res"])
        assert v["res"] == self.res

        # reported region, identical to the binary map's so the two are comparable
        self.oj0 = int((g["out_minx"] - g["minx"]) / self.res)
        self.oi0 = int((g["out_miny"] - g["miny"]) / self.res)
        self.nx = int((g["out_maxx"] - g["out_minx"]) / self.res)
        self.ny = int((g["out_maxy"] - g["out_miny"]) / self.res)
        # ... expressed in voxel-grid indices
        self.vj0 = int((g["out_minx"] - v["minx"]) / self.res)
        self.vi0 = int((g["out_miny"] - v["miny"]) / self.res)

        if verbose:
            print(f"reported region {self.ny} x {self.nx} cells @ {self.res:g} m")

        dtm = np.load(DATA / "dtm2m.npy", mmap_mode="r")
        dsm = np.load(DATA / "dsm2m.npy", mmap_mode="r")
        # bare earth over the VOXEL extent, for the near-field march
        dj = int((v["minx"] - g["minx"]) / self.res)
        di = int((v["miny"] - g["miny"]) / self.res)
        gnd = np.ascontiguousarray(dtm[di:di + v["ny"], dj:dj + v["nx"]]).astype(np.float32)
        med = float(np.nanmedian(gnd))
        self.gnd = cp.asarray(np.nan_to_num(gnd, nan=med))
        self.occ = cp.asarray(np.load(DATA / "voxels/occ.npy"))
        self.solid = cp.asarray(np.load(DATA / "voxels/solid.npy"))

        # full DSM + eye height, for the far-field march
        d = np.ascontiguousarray(dsm).astype(np.float32)
        self.dsm = cp.asarray(np.nan_to_num(d, nan=med))
        t = np.ascontiguousarray(dtm).astype(np.float32)
        self.eye = cp.asarray(np.nan_to_num(t, nan=med) + np.float32(EYE_H))
        self.dsm_shape = self.dsm.shape

        self.conv = grid_convergence_deg((g["out_minx"] + g["out_maxx"]) / 2,
                                         (g["out_miny"] + g["out_maxy"]) / 2)
        self.vh = np.float32(v["vh"])
        self.nb = np.int32(v["bands"])
        if verbose:
            free, total = cp.cuda.Device(0).mem_info
            print(f"grid convergence {self.conv:+.3f} deg; "
                  f"{(total-free)/1e9:.1f} GB on the GPU")

    # ---------------------------------------------------------------- near field
    def near(self, az_grid, alt):
        cp = self.cp
        ny, nx, res = self.ny, self.nx, self.res
        vi0, vj0 = self.vi0, self.vj0
        tan_alt = np.tan(np.radians(alt))
        dvox = min(NB_TOP / max(tan_alt, 1e-4), VOX_DMAX)

        blocked = cp.zeros((ny, nx), dtype=cp.bool_)
        vegpath = cp.zeros((ny, nx), dtype=cp.float32)
        di, dj, dist = ray_offsets(az_grid, res, dvox, step=res)
        gnd_p = self.gnd[vi0:vi0 + ny, vj0:vj0 + nx]

        # segment length credited to each sample: the ray advances one cell per
        # unique offset, so the slant length between consecutive samples
        seg = np.diff(np.concatenate([[0.0], dist])) / np.cos(np.radians(alt))

        for k in range(len(di)):
            si, sj = vi0 + int(di[k]), vj0 + int(dj[k])
            if si < 0 or sj < 0 or si + ny > self.occ.shape[0] or sj + nx > self.occ.shape[1]:
                break                      # ran out of voxel margin; far field takes over
            d = float(dist[k])
            bias = np.float32(d * tan_alt + d * d / (2.0 * R_EFF))
            self.near_k(self.occ[si:si + ny, sj:sj + nx],
                        self.solid[si:si + ny, sj:sj + nx],
                        self.gnd[si:si + ny, sj:sj + nx], gnd_p,
                        bias, np.float32(seg[k]), self.vh, self.nb,
                        blocked, vegpath)
        return blocked, vegpath, dvox, len(di)

    # ----------------------------------------------------------------- far field
    def far(self, az_grid, alt, dmin):
        """Threshold height S: an observer at p is blocked iff eye(p) < S(p)."""
        cp = self.cp
        res = self.res
        tan_alt = np.tan(np.radians(alt))
        best = cp.full(self.dsm_shape, -1e30, dtype=cp.float32)
        nyg, nxg = self.dsm_shape
        nsteps = 0

        for f, d0, d1 in BANDS:
            if d1 <= dmin or d0 >= DMAX:
                continue
            level = maxpool_upsample_gpu(cp, self.dsm, f)
            di, dj, dist = ray_offsets(az_grid, res, min(d1, DMAX),
                                       step=f * res, dmin=max(d0, dmin))
            for k in range(len(di)):
                i, j = int(di[k]), int(dj[k])
                i0d, i1d = max(0, -i), min(nyg, nyg - i)
                j0d, j1d = max(0, -j), min(nxg, nxg - j)
                if i0d >= i1d or j0d >= j1d:
                    continue
                d = float(dist[k])
                bias = np.float32(d * tan_alt + d * d / (2.0 * R_EFF))
                self.far_k(level[i0d + i:i1d + i, j0d + j:j1d + j], bias,
                           best[i0d:i1d, j0d:j1d])
                nsteps += 1
            del level
            cp.get_default_memory_pool().free_all_blocks()

        sub = best[self.oi0:self.oi0 + self.ny, self.oj0:self.oj0 + self.nx]
        eye = self.eye[self.oi0:self.oi0 + self.ny, self.oj0:self.oj0 + self.nx]
        return eye < sub, nsteps

    # -------------------------------------------------------------------- public
    def classify(self, az_true, alt, verbose=False):
        az = az_true + self.conv
        nb_blocked, vegpath, dvox, n1 = self.near(az, alt)
        far_blocked, n2 = self.far(az, alt, dvox)
        cls = self.cp.where(nb_blocked | far_blocked, 1,
                            self.cp.where(vegpath > 0.0, 2, 0)).astype(self.cp.uint8)
        if verbose:
            print(f"    voxel march to {dvox:.0f} m ({n1} steps), "
                  f"DSM march {dvox:.0f}-{DMAX:.0f} m ({n2} steps)")
        return cls, vegpath


def summarise(cp, cls, vegpath, k):
    n = cls.size
    clear = int((cls == 0).sum())
    blocked = int((cls == 1).sum())
    canopy = int((cls == 2).sum())
    trans = cp.where(cls == 2, cp.exp(-k * vegpath), 0.0)
    # sun actually reaching the observer, 0..1
    vis = cp.where(cls == 0, 1.0, trans)
    return dict(clear=100 * clear / n, blocked=100 * blocked / n,
                canopy=100 * canopy / n, mean_vis=float(vis.mean()) * 100,
                canopy_half=100 * float((trans > 0.5).sum()) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="only this HH:MM")
    ap.add_argument("--down", type=int, default=4, help="viewer downsample factor")
    a = ap.parse_args()

    from eclipse import circumstances
    from solar import LEIPZIG_LAT, LEIPZIG_LON, sun_position

    m = Marcher()
    cp = m.cp
    OUT.mkdir(parents=True, exist_ok=True)

    times = [_dt.datetime(2026, 8, 12, 19, 20, tzinfo=CEST) + _dt.timedelta(minutes=5 * k)
             for k in range(17)]
    sp = sun_position(times, LEIPZIG_LAT, LEIPZIG_LON)
    circ = circumstances()
    by_min = {t.astimezone(CEST).strftime("%H:%M"): o for t, o, *_ in circ["samples"]}

    labels = [t.strftime("%H:%M") for t in times]
    keep = [i for i, l in enumerate(labels) if (a.at is None or l == a.at)]
    if not keep:
        ap.error(f"no such timestamp; have {labels}")

    D = a.down
    ny_d, nx_d = m.ny // D, m.nx // D
    cls_small = np.zeros((len(times), ny_d, nx_d), np.uint8)
    trans_small = np.zeros((len(times), ny_d, nx_d), np.uint8)
    rows = []

    print(f"\n{'time':>6} {'az':>7} {'alt':>6} {'obsc':>6} "
          f"{'clear':>7} {'blocked':>8} {'canopy':>7} {'sun through':>12}")
    for i in keep:
        az = float(sp["azimuth"][i])
        alt = float(sp["apparent_altitude"][i])
        obsc = by_min.get(labels[i], 0.0)
        if alt <= -0.5:
            # Sun below the horizon: there is no direct sight line from anywhere,
            # so every cell is blocked. Skipping these timestamps left their planes
            # at zero, and zero reads as "not a wall" -- which painted the entire
            # city as merely canopy-attenuated at 20:40, after sunset.
            cls_small[i] = 255
            trans_small[i] = 0
            rows.append(dict(t=labels[i], az=az, alt=alt, obsc=obsc, clear=0.0,
                             blocked=100.0, canopy=0.0, mean_vis=0.0, canopy_half=0.0))
            print(f"{labels[i]:>6} {az:7.2f} {alt:6.2f} {obsc*100:5.1f}% "
                  f"{'--- sun below the horizon: all blocked ---':>44}", flush=True)
            continue
        cls, vegpath = m.classify(az, alt, verbose=a.at is not None)
        s = summarise(cp, cls, vegpath, K_AUGUST)
        print(f"{labels[i]:>6} {az:7.2f} {alt:6.2f} {obsc*100:5.1f}% "
              f"{s['clear']:6.1f}% {s['blocked']:7.1f}% {s['canopy']:6.1f}% "
              f"{s['mean_vis']:11.1f}%", flush=True)
        rows.append(dict(t=labels[i], az=az, alt=alt, obsc=obsc, **s))

        # Downsample for the viewer by AVERAGING, not by taking the worst cell.
        # Transmittance is already a probability that one sight line gets through,
        # so its mean over a block is the same kind of quantity at a coarser scale;
        # a min would turn an 8 m pixel into "the unluckiest 2 m inside it" and
        # paint every park uniformly black. The wall fraction is kept separately
        # because "no sun ever" and "sun through a lot of leaves" are different
        # answers and must not average into each other.
        trans = cp.where(cls == 2, cp.exp(-K_AUGUST * vegpath),
                         cp.where(cls == 0, 1.0, 0.0)).astype(cp.float32)
        blk = lambda a: a[:ny_d * D, :nx_d * D].reshape(ny_d, D, nx_d, D).mean(axis=(1, 3))
        cls_small[i] = cp.asnumpy((blk((cls == 1).astype(cp.float32)) * 255).astype(cp.uint8))
        trans_small[i] = cp.asnumpy((blk(trans) * 255).astype(cp.uint8))

        if labels[i] in ("20:10", "20:30", "19:45"):
            np.savez_compressed(
                OUT / f"vox_{labels[i].replace(':','')}.npz",
                cls=cp.asnumpy(cls), vegpath=cp.asnumpy(vegpath.astype(cp.float16)),
                az=np.float32(az), alt=np.float32(alt))

    if a.at is None:
        assert len(rows) == len(times), (
            f"only {len(rows)} of {len(times)} timestamps produced a frame; an "
            "unwritten plane reads as all-clear in the viewer")
    np.savez_compressed(
        OUT / "vox_timeline.npz", wall=cls_small, trans=trans_small,
        times=np.array(labels), down=np.int32(D),
        extent=np.array([m.gmeta["out_minx"], m.gmeta["out_maxx"],
                         m.gmeta["out_miny"], m.gmeta["out_maxy"]], np.int64),
        res=np.float32(m.res * D), k_august=np.float32(K_AUGUST),
        k_leafoff=np.float32(K_LEAFOFF))
    (OUT / "vox_meta.json").write_text(json.dumps({
        "classes": {"0": "clear", "1": "blocked (terrain/building/far field)",
                    "2": "through canopy"},
        "k_august": K_AUGUST, "k_leafoff": K_LEAFOFF,
        "vox_dmax": VOX_DMAX, "band_top_m": NB_TOP,
        "grid_convergence_deg": m.conv,
        "extent": [m.gmeta["out_minx"], m.gmeta["out_maxx"],
                   m.gmeta["out_miny"], m.gmeta["out_maxy"]],
        "res": m.res, "viewer_res": m.res * D,
        "rows": rows,
    }, indent=2))
    print(f"\nsaved {OUT}/vox_timeline.npz")


if __name__ == "__main__":
    main()
