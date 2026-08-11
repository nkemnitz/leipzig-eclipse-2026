"""Fold the viewer into ONE self-contained HTML file.

Needed because a published page cannot fetch sibling assets or a CDN, so three.js,
OrbitControls, the app and every texture must live inside the document. Textures
are re-encoded smaller here; the full-resolution build stays in viewer/ for local use.

three.module.js is an ES module ending in `export { A, B, ... }`. Stripping that
export leaves every name in module scope, so re-creating `const THREE = {A, B, ...}`
from the same list lets the unmodified app code keep using `THREE.Mesh` etc.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

from PIL import Image

V = Path("viewer")
OUT = Path("out/leipzig_eclipse_viewer.html")

# Smaller than the local build so the whole page fits comfortably when inlined.
TEX_W = 1000
ORTHO_W = 1800
ORTHO_Q = 72


def b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def shrink_png(path: Path, w: int, nearest: bool) -> bytes:
    im = Image.open(path)
    h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.NEAREST if nearest else Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def shrink_jpg(path: Path, w: int, q: int) -> bytes:
    im = Image.open(path).convert("RGB")
    h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
    return buf.getvalue()


def inline_three() -> str:
    src = (V / "lib/three.module.js").read_text(encoding="utf-8")
    m = list(re.finditer(r"^export\s*\{([^}]*)\}\s*;?\s*$", src, re.M))
    if not m:
        raise SystemExit("could not find three.js export block")
    last = m[-1]
    names = []
    for part in last.group(1).split(","):
        part = part.strip()
        if not part:
            continue
        # handle "X as Y"
        names.append(part.split(" as ")[-1].strip())
    body = src[: last.start()] + src[last.end():]
    ns = "const THREE = {" + ", ".join(names) + "};"
    return body + "\n" + ns + "\n"


def inline_controls() -> str:
    """OrbitControls, isolated in a closure.

    three.js and OrbitControls both declare module-private top-level names (e.g.
    `_ray`). Concatenated into one module scope that is a redeclaration error and
    nothing runs, so each borrower gets its own closure; they can still read
    three.js's names from the enclosing module scope.
    """
    src = (V / "lib/OrbitControls.js").read_text(encoding="utf-8")
    src = re.sub(r"^import[^;]*;\s*$", "", src, flags=re.M)          # drop `import ... from 'three'`
    src = re.sub(r"^export\s*\{[^}]*\}\s*;?\s*$", "", src, flags=re.M)
    src = re.sub(r"^export\s+(class|const|function)", r"\1", src, flags=re.M)
    return "const OrbitControls = (function(){\n" + src + "\nreturn OrbitControls;\n})();\n"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d = V / "data"
    meta = json.loads((d / "meta.json").read_text())

    assets = {
        n: b64(shrink_png(d / f"{n}.png", TEX_W, nearest=True), "image/png")
        # vox_wall is packed bitplanes and vox_trans is a transmittance byte per
        # channel; both must be resampled NEAREST like the other data textures,
        # since interpolating a bit or a class boundary invents values.
        for n in ("height", "ground", "surface", "info", "terrain", "canopy",
                  "vox_wall", "vox_trans")
    }
    # The local build uses four 4096x4608 quadrants (2 m/px); inlined they would
    # blow the page budget, so each is shrunk but the 2x2 layout is kept so the
    # quadrant-selecting shader needs no changes.
    for q in ("0_0", "1_0", "0_1", "1_1"):
        assets[f"ortho_{q}"] = b64(
            shrink_jpg(d / f"ortho_{q}.jpg", ORTHO_W // 2, ORTHO_Q), "image/jpeg")
    for k, v in assets.items():
        print(f"  {k:8s} {len(v)/1e6:5.2f} MB (base64)")

    app = (V / "app.js").read_text(encoding="utf-8")
    app = re.sub(r"^import[^;]*;\s*$", "", app, flags=re.M)
    # Same isolation for the app; it uses top-level await, so an async IIFE.
    app_wrapped = "(async () => {\n%s\n})();\n"
    # swap network loads for the inlined data: URIs
    app = app.replace("await (await fetch('./data/meta.json')).json()", "window.__META")
    app = app.replace("loadImageData(`./data/${n}.png`)", "loadImageData(window.__ASSETS[n])")
    app = app.replace("_tl.loadAsync(`./data/ortho_${q}.jpg`)",
                      "_tl.loadAsync(window.__ASSETS[`ortho_${q}`])")
    # No LoD2 tiles are inlined (43 MB); the page degrades to terrain-only.
    app = app.replace("await (await fetch('./data/lod2/manifest.json')).json()", "null")

    html = (V / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'<script type="importmap">.*?</script>', "", html, flags=re.S)
    html = html.replace('<script type="module" src="./app.js"></script>', "")
    # keep only the <body> content; the publisher supplies the document skeleton
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    head_style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)

    doc = f"""<title>Leipzig eclipse viewshed — 12 Aug 2026</title>
<style>{head_style}</style>
{body}
<script>
window.__META = {json.dumps(meta, separators=(",", ":"), ensure_ascii=False)};
window.__ASSETS = {json.dumps(assets)};
</script>
<script type="module">
{inline_three()}
{inline_controls()}
{app_wrapped % app}
</script>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"\n{OUT}  {OUT.stat().st_size/1e6:.2f} MB")
    if OUT.stat().st_size > 15_500_000:
        print("  !! over the 16 MB publish limit -- lower TEX_W / ORTHO_W")


if __name__ == "__main__":
    main()
