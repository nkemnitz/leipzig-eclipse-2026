import * as THREE from 'three';
import { OrbitControls } from './lib/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ---------------------------------------------------------------- data loading
async function loadImageData(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { data: ctx.getImageData(0, 0, c.width, c.height).data, w: c.width, h: c.height };
}

function rawTexture(imgdata) {
  // Byte-exact texture: nearest filtering, no colour-space conversion, so the
  // packed bitplanes survive the trip to the GPU intact.
  const t = new THREE.DataTexture(new Uint8Array(imgdata.data), imgdata.w, imgdata.h,
    THREE.RGBAFormat, THREE.UnsignedByteType);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.colorSpace = THREE.NoColorSpace;
  t.flipY = false;
  t.needsUpdate = true;
  return t;
}

const meta = await (await fetch('./data/meta.json')).json();
const [MINX, MAXX, MINY, MAXY] = meta.extent;
const WIDTH_M = MAXX - MINX, DEPTH_M = MAXY - MINY;

let heightImg, groundImg, surfaceImg, infoImg, terrainImg, canopyImg, orthoTex, orthoQuads;
let voxWallImg, voxTransImg;
try {
  [heightImg, groundImg, surfaceImg, infoImg, terrainImg, canopyImg,
   voxWallImg, voxTransImg] = await Promise.all(
    ['height', 'ground', 'surface', 'info', 'terrain', 'canopy',
     'vox_wall', 'vox_trans'].map((n) => loadImageData(`./data/${n}.png`)));
  const _tl = new THREE.TextureLoader();
  orthoQuads = await Promise.all(['0_0','1_0','0_1','1_1'].map(
    (q) => _tl.loadAsync(`./data/ortho_${q}.jpg`)));
  orthoTex = orthoQuads[0];
} catch (e) {
  $('#load').innerHTML = `<div id="err">Could not load viewer data.<br>` +
    `Run <code>python build_viewer.py</code>, then serve this folder over HTTP ` +
    `(<code>python -m http.server</code>) — opening index.html via file:// is blocked ` +
    `by the browser.<br><br>${e}</div>`;
  throw e;
}
for (const t of orthoQuads) {
  t.colorSpace = THREE.SRGBColorSpace;
  t.flipY = false;
  t.anisotropy = 8;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;   // no bleeding across the seam
}

const TW = heightImg.w, TH = heightImg.h;
const { h0: H0, scale: HSCALE } = meta.height;

// Decode the hi/lo byte pair back into metres, once, on the CPU. Used for the
// mesh, for raycast readouts and for the camera framing.
const heights = new Float32Array(TW * TH);        // DOM1 surface (roofs + canopy)
const terrain = new Float32Array(TW * TH);        // DGM1 bare earth
const canopyH = new Float32Array(TW * TH);        // smoothed canopy blanket
const canopyMask = new Uint8Array(TW * TH);
for (let i = 0; i < TW * TH; i++) {
  heights[i] = ((heightImg.data[i * 4] << 8) | heightImg.data[i * 4 + 1]) / HSCALE + H0;
  terrain[i] = ((terrainImg.data[i * 4] << 8) | terrainImg.data[i * 4 + 1]) / HSCALE + H0;
  canopyH[i] = ((canopyImg.data[i * 4] << 8) | canopyImg.data[i * 4 + 1]) / HSCALE + H0;
  canopyMask[i] = canopyImg.data[i * 4 + 2] > 127 ? 1 : 0;
}

// ------------------------------------------------------------------ scene setup
const app = $('#app');
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
scene.fog = new THREE.Fog(0x0d1117, WIDTH_M * 0.9, WIDTH_M * 2.4);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 20, WIDTH_M * 6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;   // never go under the ground
// Zoom toward whatever is under the pointer rather than toward the stored pivot,
// which is what makes close-in navigation feel anchored to the map.
controls.zoomToCursor = true;
controls.target.set(0, 0, 0);

// World frame: +X east, +Z south (so north is -Z), +Y up.
function worldFromUTM(x, y) { return [x - MINX - WIDTH_M / 2, -(y - MINY - DEPTH_M / 2)]; }
function utmFromWorld(wx, wz) { return [wx + MINX + WIDTH_M / 2, -wz + MINY + DEPTH_M / 2]; }

// The rasters are written with ROW 0 = SOUTH (miny), matching the analysis grid,
// so image row index is v directly -- NOT (1-v). Getting this backwards mirrors
// every lookup across the city and silently returns another place's numbers.
function texRow(v) { return clamp(Math.floor(v * TH), 0, TH - 1); }

function sampleTex(img, wx, wz) {
  const [x, y] = utmFromWorld(wx, wz);
  const u = (x - MINX) / (MAXX - MINX), v = (y - MINY) / (MAXY - MINY);
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  const px = clamp(Math.floor(u * TW), 0, TW - 1);
  const i = (texRow(v) * TW + px) * 4;
  return [img.data[i], img.data[i + 1], img.data[i + 2], img.data[i + 3]];
}
function terrainAt(wx, wz) {
  const [x, y] = utmFromWorld(wx, wz);
  const u = (x - MINX) / (MAXX - MINX), v = (y - MINY) / (MAXY - MINY);
  return terrain[texRow(v) * TW + clamp(Math.floor(u * TW), 0, TW - 1)];
}
function heightAt(wx, wz) {
  const [x, y] = utmFromWorld(wx, wz);
  const u = (x - MINX) / (MAXX - MINX), v = (y - MINY) / (MAXY - MINY);
  return heights[texRow(v) * TW + clamp(Math.floor(u * TW), 0, TW - 1)];
}

// ------------------------------------------------------------------- material
const uniforms = {
  uOrtho: { value: orthoQuads[0] },
  uOrtho10: { value: orthoQuads[1] },
  uOrtho01: { value: orthoQuads[2] },
  uOrtho11: { value: orthoQuads[3] },
  uGround: { value: rawTexture(groundImg) },
  uSurface: { value: rawTexture(surfaceImg) },
  uInfo: { value: rawTexture(infoImg) },
  uTerrain: { value: rawTexture(terrainImg) },
  uVoxWall: { value: rawTexture(voxWallImg) },
  uVoxTrans: { value: rawTexture(voxTransImg) },
  // Weights over the three baked key times (19:45 / 20:10 / 20:30). Transmittance
  // falls smoothly and monotonically as the sun sinks, so interpolating between
  // three keys costs far less accuracy than the 8 m grid already spends.
  uKeyW: { value: new THREE.Vector3(0, 1, 0) },
  // Declared here, not assigned later: `shared` is built before that point and
  // would capture undefined, leaving the canopy material's uniform unbound.
  uCanopyDetail: { value: 0 },
  uPlaneSel: { value: new THREE.Vector4(1, 0, 0, 0) },
  uBitPow: { value: 1 },
  uSunDir: { value: new THREE.Vector3(0, 1, 0) },
  uMode: { value: 0 },
  uMarginLo: { value: meta.info.margin_lo },
  uMarginHi: { value: meta.info.margin_hi },
  uHorizonLo: { value: meta.info.horizon_lo },
  uHorizonHi: { value: meta.info.horizon_hi },
  uSunAlt: { value: 3.5 },
  uMarker: { value: new THREE.Vector2(1e9, 1e9) },
};


// ---------------------------------------------------------------------- i18n
// The page is about a German city and will mostly be read there, so DE is a peer
// of EN, not an afterthought: every string lives here and the toggle swaps them
// live, including the ones built at runtime (legend, readout, sun status).
const I18N = {
  en: {
    max:'max', title:'Where the eclipsed sun is visible',
    subtitle:'Leipzig · 12 Aug 2026 · computed from GeoSN 1 m DOM1',
    surface:'Surface colour', mode0:'Aerial + shadow', mode1:'Can I see the sun?',
    exag:'Vertical exaggeration', detail:'Terrain mesh resolution',
    low:'Low', med:'Medium', high:'High',
    landmarks:'Landmarks', buildingsBtn:'Buildings (LoD2)',
    detailNote:'How finely the ground surface is drawn. Close in, 1 m aerial imagery and '
      +'elevation stream in automatically regardless.',
    canopyDetail:'Tree canopy', cdCoarse:'Coarse', cdFine:'Detailed (1 m)',
    canopyNote:'Detailed canopy adds a second 1 m mesh per streamed tile. It looks sharper '
      +'up close and costs noticeably more memory.',
    landmarkNote:"Buildings are Leipzig's LoD2 model (2021), drawn for orientation only. "
      +"The sun answer comes from the 2023 1 m laser surface.",
    spots:'Best spots (sight lines to the sun)', clickTitle:'Click the map',
    disclaimer:'A hobby project, offered as-is with no guarantees. It is a geometry '
      +'calculation from open data, not advice \u2014 check the sky, not this page, and never '
      +'look at the sun without proper eye protection.',
    navHelp:'Mouse: drag to orbit, right-drag to pan, scroll to zoom. Touch: one finger to '
      +'orbit, two to pan and pinch to zoom. Tap or click the map to read out that point and '
      +'anchor the view there; pick a spot from the list to fly to it.',
    sortPct:'By sight lines', sortKm:'By distance',
    tabInfo:'Settings', tabPoint:'Point', tabLegend:'Legend',
    walkTo:'best area is', wallShare:'terrain/building',
    clickHint:'Pick any point to see whether the sun clears its skyline.',
    skyHint:"Skyline (white) vs the sun's path (yellow). Where the yellow line is "
      +"above the white ridge, the sun is visible.",
    loading:'Loading terrain…',
    mh0:'Aerial imagery, lit where the sun still reaches that surface at the chosen time.',
    mh1:'Share of sight lines from that spot that reach the sun, marched through the laser '
      +'point cloud resolved in height. Dark red is terrain or masonry, or canopy so dense it '
      +'amounts to the same. Amber is marginal — a few metres either way changes the answer. '
      +'Follows the time slider.',
    lgLit:'sunlit surface', lgShadow:'in shadow',
    lgCanSee:'you can see the sun from here', lgBlocked:'skyline blocks it',
    lgVegOpen:'most sight lines reach the sun',
    lgVegMid:'marginal — a few metres either way changes it',
    lgVegDense:'canopy too dense to see through',
    lgVegWall:'terrain or building — no sight line at all',
    lgNotStand:'not open ground (dimmed)',
    lgBuilding:'building (LoD2)',
    selected:'Selected point', standable:'open ground you can stand on',
    notStandable:'not open ground (building, canopy or water)',
    kSkyline:'Skyline height (WNW)', kClears:'Sun clears it by, at max', kAt:'…at 20:30',
    kVisMax:'Sun visible at maximum', kUntil:'Visible until', kGround:'Ground elevation',
    kCoords:'Coordinates', yes:'YES', no:'no',
    sunUp:'up', azimuth:'azimuth', maxEclipse:'maximum eclipse',
    beforeC1:'before first contact', covered:'covered',
  },
  de: {
    max:'Max', title:'Wo die verfinsterte Sonne sichtbar ist',
    subtitle:'Leipzig · 12. Aug. 2026 · berechnet aus GeoSN 1 m DOM1',
    surface:'Oberflächenfarbe', mode0:'Luftbild + Schatten', mode1:'Sehe ich die Sonne?',
    exag:'Überhöhung', detail:'Geländeauflösung',
    low:'Niedrig', med:'Mittel', high:'Hoch',
    landmarks:'Orientierung', buildingsBtn:'Gebäude (LoD2)',
    detailNote:'Wie fein die Geländeoberfläche gezeichnet wird. Aus der Nähe werden Luftbild '
      +'und Höhen in 1 m ohnehin automatisch nachgeladen.',
    canopyDetail:'Kronendach', cdCoarse:'Grob', cdFine:'Detailliert (1 m)',
    canopyNote:'Das detaillierte Kronendach fügt je Kachel ein zweites 1-m-Gitter hinzu – '
      +'schärfer aus der Nähe, aber deutlich speicherhungriger.',
    landmarkNote:'Gebäude stammen aus dem LoD2-Modell der Stadt Leipzig (2021), Bäume '
      +'aus dem Baumkataster – beide dienen nur der Orientierung. Die Sonnenberechnung '
      +'nutzt das 1-m-Laseroberflächenmodell von 2023.',
    spots:'Beste Standorte (Sichtlinien zur Sonne)', clickTitle:'Karte anklicken',
    disclaimer:'Ein Hobbyprojekt, ohne Gew\u00e4hr. Es ist eine Geometrieberechnung aus '
      +'offenen Daten, keine Empfehlung \u2013 verlass dich auf den Himmel, nicht auf diese '
      +'Seite, und schau nie ohne geeigneten Augenschutz in die Sonne.',
    navHelp:'Maus: ziehen zum Drehen, rechts ziehen zum Verschieben, scrollen zum Zoomen. '
      +'Touch: ein Finger dreht, zwei verschieben und zoomen. Tippen bzw. klicken liest den '
      +'Punkt aus und verankert die Ansicht; ein Standort aus der Liste fliegt dorthin.',
    sortPct:'Nach Sichtlinien', sortKm:'Nach Entfernung',
    tabInfo:'Einstellungen', tabPoint:'Punkt', tabLegend:'Legende',
    walkTo:'beste Fläche liegt', wallShare:'Gelände/Gebäude',
    clickHint:'Beliebigen Punkt wählen, um zu sehen, ob die Sonne dort über den Horizont reicht.',
    skyHint:'Horizontlinie (weiß) und Sonnenbahn (gelb). Wo die gelbe Linie über der '
      +'weißen Kante liegt, ist die Sonne sichtbar.',
    loading:'Gelände wird geladen…',
    mh0:'Luftbild, beleuchtet dort, wo die Sonne die Oberfläche zur gewählten Zeit noch erreicht.',
    mh1:'Anteil der Sichtlinien von diesem Punkt, die die Sonne erreichen – berechnet aus '
      +'der Laser-Punktwolke, in der Höhe aufgelöst. Dunkelrot ist Gelände oder Mauerwerk, '
      +'oder ein Kronendach, das genauso dicht ist. Bernstein heißt grenzwertig: wenige Meter '
      +'entscheiden. Folgt dem Zeitregler.',
    lgLit:'besonnte Fläche', lgShadow:'im Schatten',
    lgCanSee:'Sonne von hier sichtbar', lgBlocked:'Horizont verdeckt sie',
    lgVegOpen:'die meisten Sichtlinien erreichen die Sonne',
    lgVegMid:'grenzwertig – wenige Meter entscheiden',
    lgVegDense:'Kronendach zu dicht, um durchzusehen',
    lgVegWall:'Gelände oder Gebäude – gar keine Sichtlinie',
    lgNotStand:'kein offenes Gelände (abgedunkelt)',
    lgBuilding:'Gebäude (LoD2)',
    selected:'Gewählter Punkt', standable:'offenes, begehbares Gelände',
    notStandable:'kein offenes Gelände (Gebäude, Baumkrone oder Wasser)',
    kSkyline:'Horizonthöhe (WNW)', kClears:'Sonne darüber, bei Maximum', kAt:'…um 20:30',
    kVisMax:'Sonne bei Maximum sichtbar', kUntil:'Sichtbar bis', kGround:'Geländehöhe',
    kCoords:'Koordinaten', yes:'JA', no:'nein',
    sunUp:'hoch', azimuth:'Azimut', maxEclipse:'Maximum der Finsternis',
    beforeC1:'vor dem ersten Kontakt', covered:'bedeckt',
  },
};
let LANG = (navigator.language || 'en').toLowerCase().startsWith('de') ? 'de' : 'en';
const T = (k) => (I18N[LANG][k] !== undefined ? I18N[LANG][k] : I18N.en[k]);

function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const v = T(el.dataset.i18n);
    if (v !== undefined) el.textContent = v;
  });
  $('#lang-en').classList.toggle('on', LANG === 'en');
  $('#lang-de').classList.toggle('on', LANG === 'de');
  $('#modehelp').textContent = T('mh' + uniforms.uMode.value);
  buildLegend();
  if (typeof renderSpots === 'function') renderSpots();
  setTime(ti);
  if (current) describe(current.wx, current.wz, current.label);
}

// The legend used to be a fixed list whose colours had drifted away from what the
// shader actually draws. It is now generated FROM the active mode, so it can only
// ever show swatches that are really on screen.
function buildLegend() {
  const rows = [];
  if (uniforms.uMode.value === 1) {
    // swatches are the shader's own constants; keep them in step with classColour
    rows.push(['#47e06b', T('lgVegOpen')], ['#c2800f', T('lgVegMid')],
              ['#703d17', T('lgVegDense')], ['#4a1f33', T('lgVegWall')],
              ['#4a5468', T('lgNotStand')]);
  } else {
    rows.push(['#f2b544', T('lgLit')], ['#38445c', T('lgShadow')]);
  }
  if (lod2 && buildingsOn) rows.push(['#b9a893', T('lgBuilding')]);
  $('#legend').innerHTML = rows.map(([c, t]) =>
    `<div class="lrow"><span class="sw" style="background:${c}"></span><span>${t}</span></div>`
  ).join('');
}

const CLASS_GLSL = /* glsl */`
  // Transmittance is the PROBABILITY that a sight line misses every leaf, so the
  // ramp must punish the low end. 2% through 40 m of Auwald is not "amber, worth
  // a try" -- it is "you will not see the sun", and it should read almost like the
  // wall it effectively is. Amber is reserved for the genuinely marginal band
  // where walking a few metres changes the answer. (An earlier sqrt() did the
  // opposite: it lifted 5% to a fifth of the way to green.)
  vec3 classColour(float wall, float tr){
    // The two hopeless classes are separated by HUE, not just lightness: masonry
    // is cool plum, dense canopy is warm brown. They mean different things -- one
    // will never change, the other is a leaf-on/leaf-off judgement -- so they must
    // be told apart at a glance, or the third class was pointless.
    vec3 cWall = vec3(0.29, 0.12, 0.20);   // terrain or masonry: hopeless
    vec3 cDark = vec3(0.44, 0.24, 0.09);   // canopy dense enough to be a wall
    vec3 cMid  = vec3(0.76, 0.50, 0.12);   // marginal: a few metres changes it
    vec3 cOpen = vec3(0.28, 0.88, 0.42);   // most sight lines clear
    if (wall > 0.5) return cWall;
    float t = clamp(tr, 0.0, 1.0);
    return t < 0.30 ? mix(cDark, cMid, t / 0.30)
                    : mix(cMid,  cOpen, (t - 0.30) / 0.70);
  }
`;

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    varying vec2 vUv;
    varying vec3 vNormalW;
    varying vec3 vWorld;
    void main(){
      vUv = uv;
      vNormalW = normalize(normalMatrix * normal);
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorld = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }`,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv; varying vec3 vNormalW; varying vec3 vWorld;
    uniform sampler2D uOrtho, uSurface, uInfo, uTerrain, uCover;
    uniform sampler2D uVoxWall, uVoxTrans; uniform vec3 uKeyW;
    uniform vec4 uPlaneSel; uniform float uBitPow;
    uniform vec3 uSunDir; uniform int uMode;
    uniform float uMarginLo, uMarginHi, uHorizonLo, uHorizonHi, uSunAlt;
    uniform vec2 uMarker;


    // The 2 m/px mosaic is 8192x9216 -- larger than many GPUs allow in one
    // texture -- so it lives as four quadrants selected here.
    uniform sampler2D uOrtho10, uOrtho01, uOrtho11;
    vec3 orthoAt(vec2 uv){
      vec2 q = step(vec2(0.5), uv);
      vec2 l = clamp(uv * 2.0 - q, 0.0005, 0.9995);
      if (q.y < 0.5) return q.x < 0.5 ? texture2D(uOrtho, l).rgb : texture2D(uOrtho10, l).rgb;
      return q.x < 0.5 ? texture2D(uOrtho01, l).rgb : texture2D(uOrtho11, l).rgb;
    }
    // plane k of the packed bitplane texture -> 0.0 or 1.0
    float plane(sampler2D t, vec2 uv){
      float b = floor(dot(texture2D(t, uv), uPlaneSel) * 255.0 + 0.5);
      return mod(floor(b / uBitPow), 2.0);
    }
    ${CLASS_GLSL}
    vec3 ramp(float t){                       // blue -> green -> yellow -> red
      t = clamp(t, 0.0, 1.0);
      vec3 c1=vec3(0.16,0.24,0.44), c2=vec3(0.15,0.62,0.35),
           c3=vec3(0.93,0.79,0.28), c4=vec3(0.85,0.25,0.20);
      if(t<0.33) return mix(c1,c2,t/0.33);
      if(t<0.66) return mix(c2,c3,(t-0.33)/0.33);
      return mix(c3,c4,(t-0.66)/0.34);
    }
    void main(){
      if (texture2D(uCover, vUv).r > 0.5) discard;   // streamed detail covers this
      vec3 base = orthoAt(vUv);
      vec4 info = texture2D(uInfo, vUv);
      vec3 N = normalize(vNormalW);
      float ndl = max(dot(N, uSunDir), 0.0);
      float litSurf = plane(uSurface, vUv);

      // Low sun -> strong warm key light, cool sky fill in shadow. The fill is
      // deliberately generous: at 3.4 deg almost nothing is lit, and a
      // physically-faithful ambient renders the city as an unreadable black slab.
      vec3 sunCol = vec3(1.20, 0.88, 0.58);
      vec3 skyCol = vec3(0.42, 0.48, 0.62);
      vec3 col = base * (skyCol * 1.25 + sunCol * (litSurf * ndl) * 2.40);

      if(uMode == 1){
        // The single analysis layer. There is no separate binary map any more:
        // the binary one called a stand of trees a wall, and at a 3.4 deg sun the
        // sight line runs UNDER the crowns, so it was answering a different
        // question than the one on the button.
        vec3 tint = classColour(plane(uVoxWall, vUv),
                                dot(texture2D(uVoxTrans, vUv).rgb, uKeyW));
        float standable = step(0.5, texture2D(uTerrain, vUv).b);
        col = mix(col * 0.55, mix(col*0.5, tint, 0.74), max(standable, 0.35));
      } else {
        col += sunCol * litSurf * ndl * 0.10;
      }

      // marker ring at the queried point
      float d = distance(vWorld.xz, uMarker);
      float ring = smoothstep(90.0, 70.0, d) - smoothstep(60.0, 40.0, d);
      col = mix(col, vec3(0.35,0.75,1.0), clamp(ring, 0.0, 1.0) * 0.9);

      gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
    }`,
});

uniforms.uVeg = { value: rawTexture(heightImg) };

// Shared entries are the SAME {value} objects, so a time change updates every
// material at once without bookkeeping.
// Every sampler a shader DECLARES must appear here. An unbound sampler2D silently
// falls back to texture unit 0 -- whatever happened to be bound last -- instead of
// erroring, so a missing entry renders plausible nonsense. uGround was missing:
// the detail tiles decoded their own R/G height bytes as if they were the sunlit
// bitplanes, drawing terrain contours in "Can I see the sun?" at high zoom.
const shared = {
  uSurface: uniforms.uSurface,
  uTerrain: uniforms.uTerrain, uVoxWall: uniforms.uVoxWall,
  uVoxTrans: uniforms.uVoxTrans, uKeyW: uniforms.uKeyW,
  uPlaneSel: uniforms.uPlaneSel, uCanopyDetail: uniforms.uCanopyDetail,
  uBitPow: uniforms.uBitPow, uSunDir: uniforms.uSunDir, uMode: uniforms.uMode,
};


const LIT_GLSL = /* glsl */`
  uniform vec4 uPlaneSel; uniform float uBitPow;
  float planeAt(sampler2D t, vec2 uv){
    float b = floor(dot(texture2D(t, uv), uPlaneSel) * 255.0 + 0.5);
    return mod(floor(b / uBitPow), 2.0);
  }`;


uniforms.uCanopy = { value: rawTexture(canopyImg) };
const canopyMaterial = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: /* glsl */`
    varying vec2 vUv; varying vec3 vNormalW;
    void main(){
      vUv = uv; vNormalW = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv; varying vec3 vNormalW;
    uniform sampler2D uCanopy, uOrtho, uOrtho10, uOrtho01, uOrtho11, uSurface, uGround, uInfo, uCover;
    uniform int uCanopyDetail;
    uniform vec4 uPlaneSel; uniform float uBitPow;
    vec3 orthoAt(vec2 uv){
      vec2 q = step(vec2(0.5), uv);
      vec2 l = clamp(uv * 2.0 - q, 0.0005, 0.9995);
      if (q.y < 0.5) return q.x < 0.5 ? texture2D(uOrtho, l).rgb : texture2D(uOrtho10, l).rgb;
      return q.x < 0.5 ? texture2D(uOrtho01, l).rgb : texture2D(uOrtho11, l).rgb;
    }
    uniform vec3 uSunDir; uniform int uMode;
    float planeC(sampler2D t, vec2 uv){
      float b = floor(dot(texture2D(t, uv), uPlaneSel) * 255.0 + 0.5);
      return mod(floor(b / uBitPow), 2.0);
    }
    void main(){
      // Yield to a streamed tile only when that tile carries its own canopy. With
      // canopy detail set to Coarse the tile supplies terrain alone, so discarding
      // here deleted the blanket and replaced it with nothing -- the canopy
      // visibly "unloaded" wherever you looked closely.
      if (uCanopyDetail == 1 && texture2D(uCover, vUv).r > 0.5) discard;
      if (texture2D(uCanopy, vUv).b < 0.5) discard;   // eroded canopy mask
      float lit = planeC(uSurface, vUv);
      float ndl = max(dot(normalize(vNormalW), uSunDir), 0.0);
      vec3 leaf = mix(vec3(0.11,0.17,0.10), orthoAt(vUv), 0.5);
      vec3 col = leaf * (vec3(0.36,0.44,0.54) * 1.2
                 + vec3(1.20,0.88,0.58) * lit * ndl * 2.3);
      if (uMode == 1){
        float litG = planeC(uGround, vUv);
        col = mix(col*0.5, litG > 0.5 ? vec3(0.25,0.85,0.35) : vec3(0.75,0.20,0.18), 0.45);
      }
      gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
    }`,
});

// ---------------------------------------------------------------- LoD2 buildings
function buildingMaterial(tint) {
  return new THREE.ShaderMaterial({
    uniforms: { ...shared, uTint: { value: new THREE.Color(tint) },
                uMinXZ: { value: new THREE.Vector2(WIDTH_M, DEPTH_M) } },
    vertexShader: /* glsl */`
      varying vec3 vWorld;
      void main(){
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorld = wp.xyz;
        gl_Position = projectionMatrix * viewMatrix * wp;
      }`,
    fragmentShader: /* glsl */`
      precision highp float;
      varying vec3 vWorld;
      uniform sampler2D uSurface; uniform vec3 uSunDir, uTint;
      uniform vec2 uMinXZ; uniform int uMode;
      ${LIT_GLSL}
      void main(){
        // Flat normal straight from the fragment's own derivatives -- LoD2 is
        // faceted anyway, and this halves the geometry payload (no normals stored).
        // Orient it toward the CAMERA, never by testing N.y: on a vertical wall
        // N.y is ~0, so float noise flips the normal per fragment and the wall
        // dithers between lit and ambient. dot(N,V) is far from zero on anything
        // you can actually see, so this is stable.
        vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
        if (dot(N, cameraPosition - vWorld) < 0.0) N = -N;
        vec2 uv = vec2((vWorld.x + uMinXZ.x * 0.5) / uMinXZ.x,
                       (-vWorld.z + uMinXZ.y * 0.5) / uMinXZ.y);
        float lit = planeAt(uSurface, uv);
        float ndl = max(dot(N, uSunDir), 0.0);
        vec3 col = uTint * (vec3(0.34,0.40,0.52) * 1.05
                   + vec3(1.20,0.88,0.58) * lit * ndl * 2.3);
        if (uMode == 1) col = mix(col, vec3(0.20,0.22,0.30), 0.55);  // recede
        gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
      }`,
    side: THREE.DoubleSide,   // the -Z scale inverts winding; don't cull on it
  });
}
const roofMat = buildingMaterial(0xb9a893);
const wallMat = buildingMaterial(0x8d8578);

const buildingGroup = new THREE.Group();
scene.add(buildingGroup);
let lod2 = null, loadedTiles = new Map(), buildingsOn = true;

try {
  lod2 = await (await fetch('./data/lod2/manifest.json')).json();
} catch (e) { lod2 = null; }

async function loadTile(key) {
  if (loadedTiles.has(key)) return;
  loadedTiles.set(key, null);                       // reserve slot
  const entry = lod2.tiles[key];
  const [tx, ty] = key.split('_').map(Number);
  const grp = new THREE.Group();
  for (const cls of ['roof', 'wall']) {
    if (!entry[cls]) continue;
    const buf = await (await fetch(`./data/lod2/${entry[cls].file}`)).arrayBuffer();
    const arr = new Int16Array(buf);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Int16BufferAttribute(arr, 3));
    const m = new THREE.Mesh(g, cls === 'roof' ? roofMat : wallMat);
    // De-quantise via the mesh transform, so the int16 buffer goes to the GPU as-is
    m.scale.set(lod2.xy_scale, lod2.z_scale * exag, lod2.xy_scale);
    const ox = lod2.minx + tx * lod2.tile, oy = lod2.miny + ty * lod2.tile;
    const [wx, wz] = worldFromUTM(ox, oy);
    m.position.set(wx, lod2.z0 * exag, wz);
    m.scale.z *= -1;                                 // UTM north -> world -Z
    grp.add(m);
  }
  loadedTiles.set(key, grp);
  buildingGroup.add(grp);
}

function updateBuildings() {
  if (!lod2 || !buildingsOn) return;
  const t = controls.target;
  const near = [];
  for (const key of Object.keys(lod2.tiles)) {
    const [tx, ty] = key.split('_').map(Number);
    const [wx, wz] = worldFromUTM(lod2.minx + (tx + 0.5) * lod2.tile,
                                  lod2.miny + (ty + 0.5) * lod2.tile);
    const d = Math.hypot(wx - t.x, wz - t.z);
    if (d < 3500) near.push([d, key]);
  }
  near.sort((a, b) => a[0] - b[0]);
  for (const [, key] of near.slice(0, 60)) loadTile(key);
}


// ------------------------------------------------------------- streamed detail
// Baked mosaics cap out at whatever fits one texture. Streaming instead means the
// only limit is the source: DOP20 imagery at 0.2 m/px straight from the GeoSN WMS
// (it sends CORS headers, so the browser can use it as a texture directly) and
// DOM1 geometry at its native 1 m from pre-baked 500 m tiles.
const WMS = 'https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest';
const DETAIL_PX = 2048;              // per 500 m tile -> 0.24 m/px
const DETAIL_RADIUS = 1400;          // metres from the camera target
const DETAIL_MAX = 12;

let detailMan = null, detailOn = true;
const detailGroup = new THREE.Group();
scene.add(detailGroup);
const detailTiles = new Map();
let canopyDetail = false;   // 1 m canopy meshes: opt-in, see loadDetail
// Streamed aerial imagery, kept per tile independently of the meshes. Toggling
// the canopy rebuilds a tile's geometry, and without this cache that rebuild also
// discarded the WMS texture -- so switching canopy detail visibly dropped the
// imagery back to the baked 2 m mosaic until every tile had refetched.
const detailTex = new Map();
// Where a detail tile is loaded, the coarse 8 m ground and canopy must stop
// drawing. The coarse canopy is a smoothed MAX, so it sits above the real crowns
// and otherwise hides the streamed tiles entirely, leaving only stray high-res
// fragments poking through. One texel per 500 m tile, sampled by global uv.
const COVER_NX = 32, COVER_NY = 36;
const coverData = new Uint8Array(COVER_NX * COVER_NY * 4);
const coverTex = new THREE.DataTexture(coverData, COVER_NX, COVER_NY,
  THREE.RGBAFormat, THREE.UnsignedByteType);
coverTex.magFilter = coverTex.minFilter = THREE.NearestFilter;
coverTex.generateMipmaps = false; coverTex.flipY = false;
coverTex.colorSpace = THREE.NoColorSpace; coverTex.needsUpdate = true;
uniforms.uCover = { value: coverTex };
function setCover(gx, gy, on) {
  if (gx < 0 || gy < 0 || gx >= COVER_NX || gy >= COVER_NY) return;
  coverData[(gy * COVER_NX + gx) * 4] = on ? 255 : 0;
  coverTex.needsUpdate = true;
}
const INLINED = typeof window.__ASSETS !== 'undefined';   // published page: no network
try {
  if (!INLINED) detailMan = await (await fetch('./data/detail/manifest.json')).json();
} catch (e) { detailMan = null; }

const detailMaterial = (map, packed, isCanopy) => new THREE.ShaderMaterial({
  uniforms: { ...shared, uMap: { value: map }, uPacked: { value: packed },
              uIsCanopy: { value: isCanopy ? 1 : 0 }, uInfo: uniforms.uInfo,
              uExtent: { value: new THREE.Vector4(MINX, MINY, WIDTH_M, DEPTH_M) } },
  vertexShader: `
    varying vec2 vUv; varying vec3 vNormalW; varying vec3 vWorld;
    void main(){
      vUv = uv; vNormalW = normalize(normalMatrix * normal);
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorld = wp.xyz;
      gl_Position = projectionMatrix * viewMatrix * wp;
    }`,
  fragmentShader: `
    precision highp float;
    varying vec2 vUv; varying vec3 vNormalW; varying vec3 vWorld;
    uniform sampler2D uMap, uPacked, uSurface, uInfo, uTerrain;
    uniform sampler2D uVoxWall, uVoxTrans; uniform vec3 uKeyW;
    uniform int uIsCanopy;
    uniform vec4 uPlaneSel; uniform float uBitPow; uniform vec4 uExtent;
    uniform vec3 uSunDir; uniform int uMode;
    ${CLASS_GLSL}
    float planeD(sampler2D t, vec2 uv){
      float b = floor(dot(texture2D(t, uv), uPlaneSel) * 255.0 + 0.5);
      return mod(floor(b / uBitPow), 2.0);
    }
    void main(){
      float b = floor(texture2D(uPacked, vUv).b * 255.0 + 0.5);
      // Canopy draws only inside its mask. Terrain draws EVERYWHERE -- punching a
      // matching hole in it leaves gaps you can see the sky through, because the
      // canopy sits higher and does not cover the hole in screen space at an
      // oblique angle. Depth testing hides the ground under the canopy for free.
      if (uIsCanopy == 1 && b < 128.0) discard;
      // masks are global rasters, so index them by world position, not local uv
      vec2 guv = vec2((vWorld.x + uExtent.z * 0.5) / uExtent.z,
                      (-vWorld.z + uExtent.w * 0.5) / uExtent.w);
      float lit = planeD(uSurface, guv);
      float ndl = max(dot(normalize(vNormalW), uSunDir), 0.0);
      vec3 base = texture2D(uMap, vUv).rgb;
      if (uIsCanopy == 1) base = mix(vec3(0.16,0.23,0.14), base, 0.7);
      vec3 col = base * (vec3(0.42,0.48,0.62) * 1.25
                 + vec3(1.20,0.88,0.58) * lit * ndl * 2.40);
      float standable = step(0.5, texture2D(uTerrain, guv).b);
      if (uMode == 1){
        vec3 tint = classColour(planeD(uVoxWall, guv),
                                dot(texture2D(uVoxTrans, guv).rgb, uKeyW));
        col = mix(col * 0.55, mix(col*0.5, tint, 0.74), max(standable, 0.35));
      }
      gl_FragColor = vec4(pow(col, vec3(0.4545)), 1.0);
    }`,
  polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
});

async function loadDetail(gx, gy) {
  if (detailTiles.has(gx + '_' + gy)) return;
  detailTiles.set(gx + '_' + gy, null);
  const T = detailMan.tile_m;
  const x0 = detailMan.minx + gx * T, y0 = detailMan.miny + gy * T;

  const img = await loadImageData(`./data/detail/${gx}_${gy}.png`);
  const N = img.w;                                   // 500 px at 1 m
  const seg = 250;                                   // 2 m mesh
  const nv = seg + 1;

  // Terrain and canopy are two independent surfaces, as in the coarse layer.
  // B packs the canopy: high bit = drawn mask, low 7 bits = height above ground.
  function makeGeom(asCanopy) {
    const pos = new Float32Array(nv * nv * 3), uvs = new Float32Array(nv * nv * 2);
    for (let j = 0; j < nv; j++) {
      for (let i = 0; i < nv; i++) {
        const u = i / seg, v = j / seg, k = j * nv + i;
        const px = Math.min(N - 1, Math.round(u * (N - 1)));
        const py = Math.min(N - 1, Math.round(v * (N - 1)));
        const o = (py * N + px) * 4;
        let h = ((img.data[o] << 8) | img.data[o + 1]) / detailMan.scale + detailMan.h0;
        if (asCanopy) h += (img.data[o + 2] & 127);
        const [wx, wz] = worldFromUTM(x0 + u * T, y0 + v * T);
        pos[k*3] = wx; pos[k*3+1] = h + (asCanopy ? 0.3 : 0.15);
        pos[k*3+2] = wz;
        uvs[k*2] = u; uvs[k*2+1] = v;
      }
    }
    const idx = new Uint32Array(seg * seg * 6);
    let q = 0;
    for (let j = 0; j < seg; j++) for (let i = 0; i < seg; i++) {
      const a = j*nv+i, b = a+1, c = a+nv, d = c+1;
      idx[q++]=a; idx[q++]=b; idx[q++]=c; idx[q++]=b; idx[q++]=d; idx[q++]=c;
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    return g;
  }

  const url = `${WMS}?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=sn_dop_020`
    + `&STYLES=&CRS=EPSG:25833&BBOX=${x0},${y0},${x0+T},${y0+T}`
    + `&WIDTH=${DETAIL_PX}&HEIGHT=${DETAIL_PX}&FORMAT=image/jpeg`;
  let tex = detailTex.get(gx + '_' + gy);
  if (!tex) {
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');
    try { tex = await loader.loadAsync(url); }
    catch (e) { detailTiles.delete(gx + '_' + gy); return; }
    detailTex.set(gx + '_' + gy, tex);
  }
  tex.colorSpace = THREE.SRGBColorSpace;
  // WMS returns a NORTH-UP image, unlike the baked mosaic (which build_viewer
  // explicitly flips to south-up). Geometry v=0 is the tile's south edge, so this
  // one needs flipY=true. Getting it wrong mirrors each tile north-south, which
  // reads as streets running over hilltops. See test_orientation.py.
  tex.flipY = true;
  tex.anisotropy = 8;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;

  const packed = rawTexture(img);
  // The 1 m canopy surface is by far the most expensive thing per tile -- a second
  // full mesh plus a second material -- and the coarse blanket already reads fine
  // from any distance you would actually judge a viewpoint at. So it is opt-in.
  const grp = new THREE.Group();
  grp.add(new THREE.Mesh(makeGeom(false), detailMaterial(tex, packed, false)));
  if (canopyDetail) {
    const cm = new THREE.Mesh(makeGeom(true), detailMaterial(tex, packed, true));
    cm.userData.isCanopy = true;
    grp.add(cm);
  }
  grp.scale.y = exag;
  detailTiles.set(gx + '_' + gy, grp);
  detailGroup.add(grp);
  setCover(gx, gy, true);
}

function updateDetail() {
  if (!detailMan || !detailOn) return;
  const T = detailMan.tile_m;
  const [tx, ty] = utmFromWorld(controls.target.x, controls.target.z);
  const near = [];
  for (const key of detailMan.tiles) {
    const [gx, gy] = key.split('_').map(Number);
    const cx = detailMan.minx + (gx + 0.5) * T, cy = detailMan.miny + (gy + 0.5) * T;
    const d = Math.hypot(cx - tx, cy - ty);
    if (d < DETAIL_RADIUS) near.push([d, gx, gy]);
  }
  near.sort((a, b) => a[0] - b[0]);
  for (const [, gx, gy] of near.slice(0, DETAIL_MAX)) loadDetail(gx, gy);
  // drop far tiles so memory does not grow without bound while roaming
  for (const [key, m] of detailTiles) {
    if (!m) continue;
    const [gx, gy] = key.split('_').map(Number);
    const cx = detailMan.minx + (gx + 0.5) * T, cy = detailMan.miny + (gy + 0.5) * T;
    if (Math.hypot(cx - tx, cy - ty) > DETAIL_RADIUS * 2.2) {
      detailGroup.remove(m);
      setCover(gx, gy, false);
      for (const c of m.children) { c.geometry.dispose(); c.material.dispose(); }
      detailTiles.delete(key);
    }
  }
}


// ------------------------------------------------------------------- geometry
let mesh = null, canopy = null;
function buildSurface(segX, src, reduce) {
  const segZ = Math.round(segX * TH / TW);
  const nx = segX + 1, nz = segZ + 1;
  const pos = new Float32Array(nx * nz * 3);
  const uvs = new Float32Array(nx * nz * 2);
  for (let j = 0; j < nz; j++) {
    const v = j / segZ;
    for (let i = 0; i < nx; i++) {
      const u = i / segX;
      const k = j * nx + i;
      // reduce over the block the vertex stands for, so features survive
      // decimation instead of flickering in and out as detail changes
      const x0 = Math.floor(u * (TW - 1)), y0 = Math.floor(v * (TH - 1));
      const bx = Math.max(1, Math.floor(TW / segX)), by = Math.max(1, Math.floor(TH / segZ));
      let hmx = reduce === 'max' ? -1e9 : (reduce === 'min' ? 1e9 : 0), cnt = 0;
      for (let dy = 0; dy < by; dy++) {
        const yy = Math.min(TH - 1, y0 + dy);
        for (let dx = 0; dx < bx; dx++) {
          const h = src[yy * TW + Math.min(TW - 1, x0 + dx)];
          if (reduce === 'max') { if (h > hmx) hmx = h; }
          else if (reduce === 'min') { if (h < hmx) hmx = h; }
          else { hmx += h; cnt++; }
        }
      }
      if (reduce === 'mean') hmx /= (cnt || 1);
      pos[k * 3] = (u - 0.5) * WIDTH_M;
      pos[k * 3 + 1] = hmx;
      // v runs south->north (row 0 = south) but world +Z is south, so negate:
      // without this the terrain is mirrored against the imagery draped on it.
      pos[k * 3 + 2] = -(v - 0.5) * DEPTH_M;
      uvs[k * 2] = u;
      uvs[k * 2 + 1] = v;
    }
  }
  const idx = (nx * nz > 65535) ? new Uint32Array(segX * segZ * 6) : new Uint16Array(segX * segZ * 6);
  let p = 0;
  for (let j = 0; j < segZ; j++) {
    for (let i = 0; i < segX; i++) {
      // winding matches the negated-Z mapping above so computeVertexNormals()
      // yields +Y normals; reversing one without the other makes the whole
      // surface backfacing and unlit
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      idx[p++] = a; idx[p++] = b; idx[p++] = c;
      idx[p++] = b; idx[p++] = d; idx[p++] = c;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

function buildMesh(segX) {
  if (mesh) { mesh.geometry.dispose(); scene.remove(mesh); }
  // ONE surface: the baked display height is bare earth under buildings (so the
  // LoD2 solids sit on real ground) and the DOM1 surface elsewhere, so trees are
  // part of the terrain instead of a second masked mesh that shed floating slabs.
  if (canopy) { canopy.geometry.dispose(); scene.remove(canopy); }
  mesh = new THREE.Mesh(buildSurface(segX, terrain, 'min'), material);
  scene.add(mesh);
  // Canopy blanket: heights are already dilated past the mask, so no quad slopes
  // down to bare earth and there are no floating wedges at treeline edges.
  canopy = new THREE.Mesh(buildSurface(segX, canopyH, 'max'), canopyMaterial);
  scene.add(canopy);
}

// Vertical exaggeration is applied through the mesh scale so shading normals stay
// consistent with what you see. Buildings must scale by the SAME factor about the
// same origin, or they float above / sink into the terrain as you drag the slider.
let exag = 2;
function applyExag() {
  mesh.scale.y = exag;
  if (canopy) canopy.scale.y = exag;
  for (const m of detailTiles.values()) if (m) m.scale.y = exag;
  for (const grp of loadedTiles.values()) {
    if (!grp) continue;
    for (const m of grp.children) {
      m.scale.y = lod2.z_scale * exag;
      m.position.y = lod2.z0 * exag;
    }
  }
}
buildMesh(1000);
applyExag();
updateBuildings();
updateDetail();

// -------------------------------------------------------------------- sun/time
let ti = meta.sun.findIndex((s) => s.t === '20:10');
if (ti < 0) ti = Math.floor(meta.sun.length / 2);

function sunVector(azDeg, altDeg) {
  const a = azDeg * Math.PI / 180, e = altDeg * Math.PI / 180;
  // east, north, up -> world x, -z, y
  return new THREE.Vector3(Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a))
    .normalize();
}

// The canopy layer is baked at three key times; everything between is interpolated.
const VOX_KEYS = ['19:45', '20:10', '20:30'].map((t) => meta.sun.findIndex((s) => s.t === t));
function keyWeights(k) {
  const [a, b, c] = VOX_KEYS;
  const w = new THREE.Vector3(0, 0, 0);
  if (k <= a) w.x = 1;
  else if (k >= c) w.z = 1;
  else if (k <= b) { const t = (k - a) / (b - a); w.x = 1 - t; w.y = t; }
  else { const t = (k - b) / (c - b); w.y = 1 - t; w.z = t; }
  return w;
}

function setTime(k) {
  ti = clamp(k, 0, meta.sun.length - 1);
  const s = meta.sun[ti];
  uniforms.uPlaneSel.value.set(...[0, 1, 2, 3].map((b) => (Math.floor(ti / 8) === b ? 1 : 0)));
  uniforms.uBitPow.value = Math.pow(2, ti % 8);
  uniforms.uKeyW.value.copy(keyWeights(ti));
  uniforms.uSunDir.value.copy(sunVector(s.az, s.alt));
  uniforms.uSunAlt.value = s.alt;
  $('#tlabel').textContent = s.t;
  const phase = s.obsc < 0.5 ? T('beforeC1')
    : (s.t === '20:10' ? T('maxEclipse') : `${s.obsc.toFixed(0)}% ${T('covered')}`);
  $('#tmeta').innerHTML = `${LANG === 'de' ? 'Sonne' : 'sun'} ${s.alt.toFixed(1)}° `
    + `${T('sunUp')} · ${T('azimuth')} ${s.az.toFixed(0)}° · <b>${phase}</b>`;
  $('#time').value = ti;
  drawSky();
}

// ------------------------------------------------------------------- readout
let current = null;
function fmtDeg(v) { return (v >= 0 ? '+' : '') + v.toFixed(2) + '°'; }

function describe(wx, wz, label) {
  const info = sampleTex(infoImg, wx, wz);
  if (!info) return;
  const [x, y] = utmFromWorld(wx, wz);
  const m10 = info[0] / 255 * (meta.info.margin_hi - meta.info.margin_lo) + meta.info.margin_lo;
  const m30 = info[1] / 255 * (meta.info.margin_hi - meta.info.margin_lo) + meta.info.margin_lo;
  const hz = info[2] / 255 * (meta.info.horizon_hi - meta.info.horizon_lo)
    + meta.info.horizon_lo;
  const standable = sampleTex(terrainImg, wx, wz)[2] > 127;

  // last timestamp with the sun visible from the ground here
  const g = sampleTex(groundImg, wx, wz);
  let lastVis = null, visAtMax = false;
  for (let k = 0; k < meta.sun.length; k++) {
    const byte = g[Math.floor(k / 8)];
    const bit = (byte >> (k % 8)) & 1;
    if (bit) { lastVis = meta.sun[k].t; if (meta.sun[k].t === '20:10') visAtMax = true; }
  }
  uniforms.uMarker.value.set(wx, wz);

  const cls = m10 > 1 ? 'good' : (m10 > 0 ? 'warn' : 'bad');
  $('#rname').textContent = label || T('selected');
  $('#rsub').innerHTML = standable
    ? `<span class="good">${T('standable')}</span>`
    : `<span class="warn">${T('notStandable')}</span>`;
  $('#rkv').innerHTML = `
    <div class="kv"><span>${T('kSkyline')}</span><span>${hz.toFixed(2)}°</span></div>
    <div class="kv"><span>${T('kClears')}</span>
      <span class="${cls}">${fmtDeg(m10)}</span></div>
    <div class="kv"><span>${T('kAt')}</span><span class="${m30 > 0 ? 'good' : 'bad'}">${fmtDeg(m30)}</span></div>
    <div class="kv"><span>${T('kVisMax')}</span>
      <span class="${visAtMax ? 'good' : 'bad'}">${visAtMax ? T('yes') : T('no')}</span></div>
    <div class="kv"><span>${T('kUntil')}</span><span>${lastVis || '—'}</span></div>
    <div class="kv"><span>${T('kGround')}</span><span>${terrainAt(wx, wz).toFixed(0)} m</span></div>
    <div class="kv"><span>${T('kCoords')}</span><span>${utmToLatLon(x, y)}</span></div>`;
  current = { wx, wz, label, profile: null };
  drawSky();
}

// UTM33N -> WGS84, inline so the page has no dependencies
function utmToLatLon(x, y) {
  const k0 = 0.9996, a = 6378137, f = 1 / 298.257222101, e2 = f * (2 - f);
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const M = y / k0, mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64));
  const p1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
    + (21 * e1 * e1 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu);
  const C1 = (e2 / (1 - e2)) * Math.cos(p1) ** 2;
  const T1 = Math.tan(p1) ** 2;
  const N1 = a / Math.sqrt(1 - e2 * Math.sin(p1) ** 2);
  const R1 = a * (1 - e2) / Math.pow(1 - e2 * Math.sin(p1) ** 2, 1.5);
  const D = (x - 500000) / (N1 * k0);
  const lat = p1 - (N1 * Math.tan(p1) / R1) * (D * D / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * e2 / (1 - e2)) * D ** 4 / 24
    + (61 + 90 * T1 + 45 * T1 * T1 - 252 * e2 / (1 - e2)) * D ** 6 / 720);
  const lon = (D - (1 + 2 * T1 + C1) * D ** 3 / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * e2 / (1 - e2) + 24 * T1 * T1) * D ** 5 / 120)
    / Math.cos(p1);
  const la = lat * 180 / Math.PI, lo = 15 + lon * 180 / Math.PI;
  return `${la.toFixed(5)}, ${lo.toFixed(5)}`;
}

// --------------------------------------------------------------- skyline chart
function drawSky() {
  const cv = $('#sky'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, W, H);

  const AZ0 = 255, AZ1 = 305, ALT0 = -2, ALT1 = 16;
  const px = (az) => (az - AZ0) / (AZ1 - AZ0) * W;
  const py = (alt) => H - (alt - ALT0) / (ALT1 - ALT0) * H;

  ctx.strokeStyle = '#20262e'; ctx.lineWidth = 1; ctx.font = '9px sans-serif';
  ctx.fillStyle = '#6a7480';
  for (let alt = 0; alt <= 15; alt += 5) {
    ctx.beginPath(); ctx.moveTo(0, py(alt)); ctx.lineTo(W, py(alt)); ctx.stroke();
    ctx.fillText(alt + '°', 3, py(alt) - 2);
  }
  for (let az = 260; az <= 300; az += 10) {
    ctx.beginPath(); ctx.moveTo(px(az), 0); ctx.lineTo(px(az), H); ctx.stroke();
    ctx.fillText(az + '°', px(az) + 2, H - 3);
  }

  const prof = current && current.profile;
  if (prof) {
    ctx.beginPath();
    const n = prof.values.length;
    for (let i = 0; i < n; i++) {
      const az = prof.az0 + (prof.az1 - prof.az0) * i / (n - 1);
      const X = px(az), Y = py(prof.values[i]);
      i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
    }
    ctx.lineTo(px(prof.az1), H); ctx.lineTo(px(prof.az0), H); ctx.closePath();
    ctx.fillStyle = '#e6edf322'; ctx.fill();
    ctx.strokeStyle = '#e6edf3'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  // sun track
  ctx.strokeStyle = '#f2b544'; ctx.lineWidth = 2; ctx.beginPath();
  meta.sun.forEach((s, i) => { i ? ctx.lineTo(px(s.az), py(s.alt)) : ctx.moveTo(px(s.az), py(s.alt)); });
  ctx.stroke();
  meta.sun.forEach((s, i) => {
    // marker size tracks obscuration, so the deepest phase reads at a glance
    const r = i === ti ? 5.5 : 1.8 + 2.4 * (s.obsc / 100);
    ctx.beginPath(); ctx.arc(px(s.az), py(s.alt), r, 0, 7);
    ctx.fillStyle = i === ti ? '#fff2c4' : '#f2b54499'; ctx.fill();
  });
  const s = meta.sun[ti];
  ctx.fillStyle = '#8b949e';
  ctx.fillText(`${s.t}  ${s.obsc.toFixed(0)}% covered`, px(s.az) + 8, py(s.alt) - 8);
}

// ------------------------------------------------------------------ interaction
const ray = new THREE.Raycaster();
renderer.domElement.addEventListener('pointerdown', (e) => { drag = false; });
let drag = false;
renderer.domElement.addEventListener('pointermove', () => { drag = true; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (drag) return;
  const r = renderer.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(m, camera);
  // Raycast the streamed detail tiles too, not just the coarse mesh. Zoomed in,
  // the 1 m tiles are what you are actually looking at; hitting only the 8 m mesh
  // anchors you to a surface up to a storey away from the one on screen.
  const targets = [mesh];
  for (const g of detailTiles.values()) if (g) targets.push(...g.children);
  const hit = ray.intersectObjects(targets, false)[0];
  if (!hit) return;
  describe(hit.point.x, hit.point.z, null);
  // On a phone the readout sheet is closed, so a tap would compute an answer the
  // user never sees. Surface it.
  if (isPhone.matches) showPanel('readout');
  // Anchor the orbit/zoom pivot on the clicked ground. Without this the pivot
  // stays wherever the last flyTo left it, so zooming from anywhere else drives
  // the camera toward a point hanging in the air.
  controls.target.copy(hit.point);
  controls.update();
});

function flyTo(spot) {
  const [wx, wz] = worldFromUTM(spot.utm_x, spot.utm_y);
  const h = heightAt(wx, wz) * exag;
  // stand back to the east-south-east so the camera looks toward the WNW sun
  const target = new THREE.Vector3(wx, h, wz);
  const dist = 1400;
  const az = (meta.sun[ti].az + 180) * Math.PI / 180;
  camera.position.set(wx + Math.sin(az) * dist, h + 480, wz - Math.cos(az) * dist);
  controls.target.copy(target);
  controls.update();
  updateBuildings();
  current = { wx, wz, profile: spot.profile ? {
    az0: spot.profile_az0, az1: spot.profile_az1, values: spot.profile } : null };
  describe(wx, wz, spot.label);
  if (spot.profile) {
    current.profile = { az0: spot.profile_az0, az1: spot.profile_az1, values: spot.profile };
    drawSky();
  }
}

// spot list
const spotsEl = $('#spots');
// Ranked by the share of sight lines that reach the sun, not by margin in degrees.
// Margin came from the opaque-column DSM, which called a stand of trees a wall --
// it listed the Fockeberg at +3.4 deg while the canopy layer beside it showed the
// same hill attenuated. Clicking flies to the best standing area, not to the
// nominal marker, so the number and the place you land on are the same place.
let spotSort = 'pct';
function renderSpots() {
  const rows = meta.spots.slice();
  // Entries with no voxel score sort last either way rather than pretending to a 0.
  rows.sort((a, b) => {
    if (spotSort === 'km') return (a.km_from_markt ?? 1e9) - (b.km_from_markt ?? 1e9);
    return ((b.vox ? b.vox.best : -1) - (a.vox ? a.vox.best : -1));
  });
  spotsEl.innerHTML = '';
  for (const s of rows.slice(0, 40)) {
    const v = s.vox;
    const el = document.createElement('div');
    el.className = 'spot';
    const right = spotSort === 'km'
      ? (s.km_from_markt != null ? s.km_from_markt.toFixed(1) + ' km' : '')
      : (v ? Math.round(v.best) + '%' : '');
    el.innerHTML = `<b>${s.label}</b><i>${right}</i>`;
    if (v) {
      el.title = `${Math.round(v.best)}% · ${T('walkTo')} ${v.walk_m} m · `
        + `${s.km_from_markt != null ? s.km_from_markt.toFixed(1) + ' km · ' : ''}`
        + `${T('wallShare')} ${v.wall}%`;
    }
    el.onclick = () => {
      flyTo(v ? { ...s, utm_x: v.best_x, utm_y: v.best_y } : s);
      if (isPhone.matches) showPanel(null);   // else you land behind the sheet
    };
    spotsEl.appendChild(el);
  }
}
renderSpots();
$('#sort').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('#sort').children].forEach((c) => c.classList.toggle('on', c === b));
  spotSort = b.dataset.sort;
  renderSpots();
};
$('#attrib').textContent = meta.attribution;

// controls
$('#time').max = meta.sun.length - 1;
$('#time').oninput = (e) => setTime(+e.target.value);
$('#tomax').onclick = () => setTime(meta.sun.findIndex((s) => s.t === '20:10'));
let playing = null;
$('#play').onclick = () => {
  if (playing) { clearInterval(playing); playing = null; $('#play').textContent = '▶'; return; }
  $('#play').textContent = '❚❚';
  playing = setInterval(() => setTime((ti + 1) % meta.sun.length), 700);
};
$('#modes').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('#modes').children].forEach((c) => c.classList.toggle('on', c === b));
  uniforms.uMode.value = +b.dataset.mode;
  $('#modehelp').textContent = T('mh' + uniforms.uMode.value);
  buildLegend();
};
// Mobile sheets. Only one may be open, and tapping its own tab closes it, so the
// map is never permanently buried under a panel on a small screen.
function showPanel(name) {
  for (const id of ['hud', 'readout', 'legend']) {
    $('#' + id).classList.toggle('open', id === name);
  }
  [...$('#mtabs').children].forEach((c) => c.classList.toggle('on', c.dataset.panel === name));
}
$('#mtabs').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  showPanel($('#' + b.dataset.panel).classList.contains('open') ? null : b.dataset.panel);
};
const isPhone = matchMedia('(max-width:760px), (pointer:coarse) and (max-width:1000px)');

$('#lang-en').onclick = () => { LANG = 'en'; applyLang(); };
$('#lang-de').onclick = () => { LANG = 'de'; applyLang(); };
$('#canopy').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('#canopy').children].forEach((c) => c.classList.toggle('on', c === b));
  canopyDetail = b.dataset.cd === '1';
  uniforms.uCanopyDetail.value = canopyDetail ? 1 : 0;
  // Already-streamed tiles were built with the old setting, so drop them and let
  // updateDetail re-stream; otherwise the toggle only affects tiles you visit next.
  for (const [k, g] of detailTiles) {
    // Geometry only: the WMS texture lives in detailTex and is reused, so the
    // imagery never flickers back to the coarse mosaic on a canopy toggle.
    if (g) { detailGroup.remove(g); g.traverse((o) => o.geometry && o.geometry.dispose()); }
    const [gx, gy] = k.split('_').map(Number);
    setCover(gx, gy, false);
  }
  detailTiles.clear();
  updateDetail();
};
$('#quality').onclick = (e) => {
  const b = e.target.closest('button'); if (!b) return;
  [...$('#quality').children].forEach((c) => c.classList.toggle('on', c === b));
  buildMesh(+b.dataset.q); applyExag();
};
const bBtn = $('#buildings');
if (!lod2) { bBtn.disabled = true; bBtn.textContent = 'Buildings (not built)'; }
else {
  bBtn.classList.add('on');
  bBtn.onclick = () => {
    buildingsOn = !buildingsOn;
    bBtn.classList.toggle('on', buildingsOn);
    buildingGroup.visible = buildingsOn;
    if (buildingsOn) updateBuildings();
    buildLegend();
  };
}
// Stream in tiles around wherever the user just moved to.
controls.addEventListener('end', () => { updateBuildings(); updateDetail(); });
$('#exag').oninput = (e) => {
  exag = +e.target.value; $('#exlabel').textContent = exag.toFixed(1) + '×'; applyExag();
};

// initial camera: south-east of the centre, looking WNW toward the setting sun
camera.position.set(WIDTH_M * 0.30, 4200, DEPTH_M * 0.42);
controls.update();
applyLang();
setTime(ti);
$('#load').remove();

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

window.__dbg = { tiles: loadedTiles, scene, meta, camera, controls,
                 get ground(){return mesh}, get canopy(){return canopy},
                 buildings: buildingGroup, detail: detailTiles,
                 get detailMan(){return detailMan}, updateDetail,
                 uniforms };   // used by verify_viewer.py
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
