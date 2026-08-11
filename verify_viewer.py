"""Load the viewer in headless Chromium, capture console errors, and screenshot it.

WebGL in headless needs SwiftShader; without the ANGLE flags the page silently
falls back to no context and every screenshot comes back black.
"""

from __future__ import annotations

import http.server
import socketserver
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path("viewer")
PORT = 8731
SHOTS = Path("out/shots")

FLAGS = [
    "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist", "--enable-webgl", "--disable-gpu-sandbox",
]


def serve():
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT), **k)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.TCPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    srv = serve()
    logs, errors, failed = [], [], []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=FLAGS)
        page = browser.new_page(viewport={"width": 1600, "height": 950},
                                device_scale_factor=1)
        # SwiftShader renders this scene in software; a frame costs seconds, and
        # each new full-screen layer costs more. The default 30 s screenshot
        # timeout started failing purely on frame time, not on anything broken.
        page.set_default_timeout(180000)
        page.on("console", lambda m: (logs.append(f"[{m.type}] {m.text}"),
                                      errors.append(m.text) if m.type == "error" else None))
        page.on("pageerror", lambda e: errors.append(f"PAGEERROR {e}"))
        page.on("requestfailed",
                lambda r: failed.append(f"{r.url.split('/')[-1]} {r.failure}"))

        page.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="load", timeout=120000)

        gl = page.evaluate("""() => {
            const c = document.createElement('canvas');
            const g = c.getContext('webgl2') || c.getContext('webgl');
            if (!g) return 'NO WEBGL';
            const d = g.getExtension('WEBGL_debug_renderer_info');
            return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'webgl ok';
        }""")
        print(f"WebGL renderer: {gl}")

        try:
            page.wait_for_selector("#load", state="detached", timeout=180000)
            print("data loaded")
        except Exception:
            print("!! loading overlay never cleared")
            print("   overlay text:", page.inner_text("#load")[:400])

        page.wait_for_timeout(4000)
        page.screenshot(path=str(SHOTS / "01_default.png"))

        for mode, name in ((1, "02_visible"),):
            page.click(f'#modes button[data-mode="{mode}"]')
            page.wait_for_timeout(1200)
            page.screenshot(path=str(SHOTS / f"{name}.png"))

        page.click('#modes button[data-mode="0"]')
        # fly to the top-ranked spot and grab the readout
        page.click("#spots .spot")
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SHOTS / "04_spot.png"))
        print("\n--- readout panel ---")
        print(page.inner_text("#readout")[:700])

        # The readout is sampled from the textures on the CPU; the sidebar value
        # comes straight from meta.json. If the raster row convention is wrong
        # they disagree wildly, so assert they match.
        import re as _re
        listed = page.inner_text("#spots .spot")
        panel = page.inner_text("#readout")
        m_list = _re.search(r"([+-]\d+\.\d+)\s*°", listed)
        m_panel = _re.search(r"clears it by, at max\s*([+-]\d+\.\d+)°", panel)
        if m_list and m_panel:
            a, b = float(m_list.group(1)), float(m_panel.group(1))
            ok = abs(a - b) < 0.6      # 8 m texture vs 2 m source
            print(f"\nconsistency: sidebar {a:+.2f}deg vs readout {b:+.2f}deg -> "
                  f"{'OK' if ok else 'MISMATCH (raster orientation?)'}")
            if not ok:
                errors.append(f"readout/sidebar mismatch {a} vs {b}")
            # A positive margin at maximum MUST mean the sunlit bitplane says yes.
            # These come from different textures, so disagreement means one of them
            # decoded wrong (this is how the alpha-premultiply bug showed up).
            vis = "YES" if "Sun visible at maximum\nYES" in panel else "no"
            agree = (b > 0) == (vis == "YES")
            print(f"margin {b:+.2f} vs 'visible at maximum' = {vis} -> "
                  f"{'OK' if agree else 'CONTRADICTION'}")
            if not agree:
                errors.append(f"margin {b} contradicts visible-at-max={vis}")
        else:
            errors.append("could not parse margin from readout")

        # The time slider must actually change the analysis layers, not just mode 0.
        page.click('#modes button[data-mode="1"]')
        page.wait_for_timeout(600)
        shot_a = page.screenshot()
        page.evaluate("document.querySelector('#time').value = 0;"
                      "document.querySelector('#time').dispatchEvent(new Event('input'))")
        page.wait_for_timeout(900)
        shot_b = page.screenshot()
        changed = shot_a != shot_b
        print(f"'Can I see the sun?' responds to the time slider -> "
              f"{'OK' if changed else 'STATIC'}")
        if not changed:
            errors.append("mode 1 does not respond to the time slider")

        # The canopy layer is interpolated between three baked key times rather
        # than read from a per-timestamp plane, so its time response is a separate
        # code path that would fail silently by showing 20:10 forever.
        w = page.evaluate("() => window.__dbg.uniforms.uKeyW.value.toArray()")
        print(f"canopy key weights at 19:20 -> {[round(v,3) for v in w]}")
        if abs(sum(w) - 1.0) > 1e-3:
            errors.append(f"canopy key weights do not sum to 1: {w}")
        page.evaluate("document.querySelector('#tomax').click()")
        page.click('#modes button[data-mode="0"]')

        # Close-up on the centre so building walls can be inspected for shading noise
        # Leipzig Markt, UTM (317035, 5690878) -> world, so the shot lands on the
        # dense centre where wall shading can actually be judged.
        page.evaluate("""() => {
            const d = window.__dbg, m = d.meta;
            const [MINX, MAXX, MINY, MAXY] = m.extent;
            const W = MAXX - MINX, D = MAXY - MINY;
            const wx = 317035 - MINX - W / 2, wz = -(5690878 - MINY - D / 2);
            d.controls.target.set(wx, 240, wz);
            d.camera.position.set(wx + 520, 480, wz + 620);
            d.controls.update();
        }""")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(SHOTS / "06_walls.png"))

        # Streamed detail tiles run a SEPARATE material from the wide view, so a
        # uniform can be bound in one and missing in the other and only the close
        # view breaks. Drop the camera onto the Fockeberg until detail loads, then
        # shoot both analysis modes there.
        page.evaluate("""() => {
            const d = window.__dbg, m = d.meta;
            const [MINX, MAXX, MINY, MAXY] = m.extent;
            const W = MAXX - MINX, D = MAXY - MINY;
            const wx = 316183 - MINX - W / 2, wz = -(5688395 - MINY - D / 2);
            d.controls.target.set(wx, 160, wz);
            d.camera.position.set(wx + 260, 300, wz + 300);
            d.controls.update();
            if (d.updateDetail) d.updateDetail();
        }""")
        page.wait_for_timeout(6000)
        ntiles = page.evaluate("() => window.__dbg.detail ? window.__dbg.detail.size : 0")
        print(f"detail tiles loaded at the Fockeberg: {ntiles}")
        if not ntiles:
            errors.append("no detail tiles loaded; the close-up modes are unverified")
        for mode, name in ((1, "22_detail_visible"),):
            page.click(f'#modes button[data-mode="{mode}"]')
            page.wait_for_timeout(2500)
            page.screenshot(path=str(SHOTS / f"{name}.png"))
        page.click('#modes button[data-mode="0"]')

        print("\n--- first spots ---")
        print(page.inner_text("#spots")[:400])
        print("\n--- time label ---", page.inner_text("#timebox")[:160].replace("\n", " | "))

        # verify the mesh actually has geometry and is not a flat plane
        # Count LoD2 triangles that actually made it into the scene
        stats = page.evaluate("""() => {
            const r = {spots: document.querySelectorAll('#spots .spot').length,
                       tiles: 0, tris: 0, meshes: 0};
            if (window.__dbg) {
                for (const g of window.__dbg.tiles.values()) {
                    if (!g) continue;
                    r.tiles++;
                    for (const m of g.children) {
                        r.meshes++;
                        r.tris += m.geometry.getAttribute('position').count / 3;
                    }
                }
            }
            return r;
        }""")
        print("stats:", stats)
        if stats.get("tiles", 0) == 0:
            errors.append("no LoD2 building tiles loaded")
        browser.close()

    srv.shutdown()
    print(f"\nconsole errors: {len(errors)}")
    for e in errors[:12]:
        print("  ERR", e[:200])
    print(f"failed requests: {len(failed)}")
    for f in failed[:12]:
        print("  REQ", f[:200])
    print("\nscreenshots:")
    for p in sorted(SHOTS.glob("*.png")):
        print(f"  {p} {p.stat().st_size/1000:.0f} kB")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
