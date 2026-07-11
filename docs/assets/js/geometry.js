// Client-side port of src/geometry.py -- NACA generation, point-cloud
// construction, and panel-method force integration, so the Tool page can
// run entirely in the browser (no Python backend) for GitHub Pages hosting.
//
// Differences from the Python version, and why they're safe:
// - resampleClose() uses arc-length-parameterized piecewise-linear
//   resampling instead of scipy's periodic B-spline (splprep/splev). Slightly
//   less smooth at high curvature, but the airfoil surfaces here are already
//   densely sampled (NACA generator emits 400 points), so the difference is
//   sub-pixel at any point cloud density we use.
// - Only 4-digit NACA codes are supported (the common case, and what the
//   Tool page's UI targets); 5-digit camber-line math is more involved and
//   deferred.
// - Point-in-polygon and point-to-boundary distance are hand-rolled
//   (ray-casting, and min distance to each edge segment) instead of shapely,
//   since shapely is a native/C library with no browser equivalent.
'use strict';

// ---------------------------------------------------------------------
// air properties (matches src/geometry.py exactly)
// ---------------------------------------------------------------------

const AIR_T = 298.15;
const P_REF = 1.01325e5;
const MOL = 28.965338e-3;

function airKinematicViscosity(T = AIR_T) {
  return -3.400747e-6 + 3.452139e-8 * T + 1.00881778e-10 * T ** 2 - 1.363528e-14 * T ** 3;
}
function airDensity(T = AIR_T) {
  return (P_REF * MOL) / (8.3144621 * T);
}

const ENVELOPE_RE = [2e6, 6e6];
const ENVELOPE_AOA = [-5.0, 15.0];

function checkEnvelope(reynolds, aoaDeg) {
  const warnings = [];
  if (reynolds < ENVELOPE_RE[0] || reynolds > ENVELOPE_RE[1]) {
    warnings.push(`Reynolds ${reynolds.toExponential(2)} is outside the training envelope [${ENVELOPE_RE}]`);
  }
  if (aoaDeg < ENVELOPE_AOA[0] || aoaDeg > ENVELOPE_AOA[1]) {
    warnings.push(`AoA ${aoaDeg} deg is outside the training envelope [${ENVELOPE_AOA}]`);
  }
  return warnings;
}

// ---------------------------------------------------------------------
// seedable RNG (mulberry32) -- deterministic point clouds per airfoil,
// same spirit as the Python side's np.random.default_rng(seed)
// ---------------------------------------------------------------------

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randUniform(rng, lo, hi) {
  return lo + rng() * (hi - lo);
}
function randInt(rng, n) {
  return Math.floor(rng() * n);
}
function randExponential(rng, scale) {
  return -Math.log(1 - rng()) * scale;
}

// ---------------------------------------------------------------------
// NACA 4-digit generator (direct port of airfrans.naca_generator)
// ---------------------------------------------------------------------

function thicknessDist(t, xs, CTE = true) {
  const a = CTE ? -0.1036 : -0.1015;
  return xs.map((x) => 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a * x ** 4));
}

function camberLine4Digit(m, p, xs) {
  const yc = new Array(xs.length).fill(0);
  const dyc = new Array(xs.length).fill(0);
  if (p === 0) {
    for (let i = 0; i < xs.length; i++) dyc[i] = -2 * m * xs[i];
    return [yc, dyc];
  }
  if (p === 1) {
    for (let i = 0; i < xs.length; i++) dyc[i] = 2 * m * (1 - xs[i]);
    return [yc, dyc];
  }
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i];
    if (x < p) {
      yc[i] = (m / p ** 2) * (2 * p * x - x ** 2);
      dyc[i] = ((2 * m) / p ** 2) * (p - x);
    } else {
      yc[i] = (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x ** 2);
      dyc[i] = ((2 * m) / (1 - p) ** 2) * (p - x);
    }
  }
  return [yc, dyc];
}

/** code: 4-digit NACA string, e.g. "2412". Returns closed loop [[x,y],...] chord=1. */
function nacaAirfoil4Digit(code, nbSamples = 300, CTE = true) {
  code = code.trim();
  if (code.length !== 4) throw new Error('only 4-digit NACA codes are supported client-side');
  const m = parseInt(code[0], 10) / 100;
  const p = parseInt(code[1], 10) / 10;
  const t = parseInt(code.slice(2), 10) / 100;

  const xs = [];
  for (let i = 0; i <= nbSamples; i++) {
    const beta = (Math.PI * (nbSamples - i)) / nbSamples; // linspace(pi, 0)
    xs.push((1 - Math.cos(beta)) / 2);
  }
  const [yc, dyc] = camberLine4Digit(m, p, xs);
  const yt = thicknessDist(t, xs, CTE);
  const theta = dyc.map(Math.atan);

  const upper = [];
  const lower = [];
  for (let i = 0; i < xs.length; i++) {
    const xu = xs[i] - yt[i] * Math.sin(theta[i]);
    const yu = yc[i] + yt[i] * Math.cos(theta[i]);
    const xl = xs[i] + yt[i] * Math.sin(theta[i]);
    const yl = yc[i] - yt[i] * Math.cos(theta[i]);
    upper.push([xu, yu]);
    lower.push([xl, yl]);
  }
  const loop = upper.concat(lower.slice(0, -1).reverse());
  loop[0] = [1, 0];
  loop[loop.length - 1] = [1, 0];
  return loop;
}

function parseDatText(text) {
  const pts = [];
  for (const line of text.split('\n')) {
    const parts = line.trim().split(/\s+/);
    if (parts.length !== 2) continue;
    const x = parseFloat(parts[0]);
    const y = parseFloat(parts[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
  }
  return pts;
}

// ---------------------------------------------------------------------
// resample + chord-normalize (arc-length piecewise-linear, see file header)
// ---------------------------------------------------------------------

function resampleClose(coords, nPoints = 300) {
  let xs = coords.map((p) => p[0]);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const chord = xmax - xmin;
  let pts = coords.map(([x, y]) => [(x - xmin) / chord, y / chord]);

  const first = pts[0];
  const last = pts[pts.length - 1];
  if (Math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-9) {
    pts = pts.concat([first]);
  }

  // cumulative arc length
  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  }
  const total = cum[cum.length - 1];

  const out = [];
  for (let i = 0; i < nPoints; i++) {
    const target = (i / nPoints) * total;
    let j = 1;
    while (j < cum.length && cum[j] < target) j++;
    j = Math.min(j, cum.length - 1);
    const segLen = cum[j] - cum[j - 1] || 1e-12;
    const f = (target - cum[j - 1]) / segLen;
    const x = pts[j - 1][0] + f * (pts[j][0] - pts[j - 1][0]);
    const y = pts[j - 1][1] + f * (pts[j][1] - pts[j - 1][1]);
    out.push([x, y]);
  }
  out.push(out[0]); // close
  return { coords: out, chord };
}

/** INWARD unit normals (into the solid) -- matches the AirfRANS training
 * convention verified in src/geometry.py; getting this backwards corrupts
 * every surface-adjacent prediction. */
function surfaceNormals(coords) {
  const pts = coords.slice(0, -1);
  const n = pts.length;
  const normals = [];
  let cx = 0,
    cy = 0;
  for (const [x, y] of pts) {
    cx += x;
    cy += y;
  }
  cx /= n;
  cy /= n;

  for (let i = 0; i < n; i++) {
    const prev = pts[(i - 1 + n) % n];
    const next = pts[(i + 1) % n];
    const tx = next[0] - prev[0];
    const ty = next[1] - prev[1];
    let nx = ty;
    let ny = -tx;
    const norm = Math.hypot(nx, ny) || 1e-12;
    nx /= norm;
    ny /= norm;
    const toCenterX = pts[i][0] - cx;
    const toCenterY = pts[i][1] - cy;
    const sign = -Math.sign(nx * toCenterX + ny * toCenterY); // negative = inward
    normals.push([nx * (sign || 1), ny * (sign || 1)]);
  }
  normals.push(normals[0]);
  return normals;
}

// ---------------------------------------------------------------------
// point-in-polygon (ray casting) and distance-to-boundary
// ---------------------------------------------------------------------

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 2; i < poly.length - 1; j = i++) {
    const xi = poly[i][0],
      yi = poly[i][1];
    const xj = poly[j][0],
      yj = poly[j][1];
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function distPointToSegment(px, py, ax, ay, bx, by) {
  const abx = bx - ax,
    aby = by - ay;
  const apx = px - ax,
    apy = py - ay;
  const lenSq = abx * abx + aby * aby || 1e-18;
  let t = (apx * abx + apy * aby) / lenSq;
  t = Math.max(0, Math.min(1, t));
  const cx = ax + t * abx,
    cy = ay + t * aby;
  return Math.hypot(px - cx, py - cy);
}

function distToBoundary(x, y, poly) {
  let best = Infinity;
  for (let i = 0; i < poly.length - 1; i++) {
    const d = distPointToSegment(x, y, poly[i][0], poly[i][1], poly[i + 1][0], poly[i + 1][1]);
    if (d < best) best = d;
  }
  return best;
}

// ---------------------------------------------------------------------
// point cloud generation (mirrors generate_geometry_cloud + apply_inflow)
// ---------------------------------------------------------------------

function generateGeometryCloud(surfaceCoords, nVolume = 12000, domain = [[-2, 4], [-1.5, 1.5]], seed = 0) {
  const rng = mulberry32(seed);
  const surfPts = surfaceCoords.slice(0, -1);
  const surfNorm = surfaceNormals(surfaceCoords).slice(0, -1);
  const nSurf = surfPts.length;

  const nNear = Math.floor(nVolume * 0.6);
  const nearPts = [];
  for (let i = 0; i < nNear; i++) {
    const idx = randInt(rng, nSurf);
    const offset = randExponential(rng, 0.03);
    // surfNorm is inward; step outward into the fluid.
    nearPts.push([surfPts[idx][0] - surfNorm[idx][0] * offset, surfPts[idx][1] - surfNorm[idx][1] * offset]);
  }
  const nFar = nVolume - nNear;
  const farPts = [];
  for (let i = 0; i < nFar; i++) {
    farPts.push([randUniform(rng, domain[0][0], domain[0][1]), randUniform(rng, domain[1][0], domain[1][1])]);
  }

  const cand = [];
  for (const [x, y] of nearPts.concat(farPts)) {
    if (x < domain[0][0] || x > domain[0][1] || y < domain[1][0] || y > domain[1][1]) continue;
    if (pointInPolygon(x, y, surfaceCoords)) continue;
    cand.push([x, y]);
  }
  const sdfCand = cand.map(([x, y]) => distToBoundary(x, y, surfaceCoords));

  const position = surfPts.concat(cand);
  const sdf = new Array(nSurf).fill(0).concat(sdfCand);
  const normals = surfNorm.concat(cand.map(() => [0, 0]));
  const isSurface = new Array(nSurf).fill(true).concat(cand.map(() => false));

  return { position, sdf, normals, isSurface, surfaceNormals: surfNorm };
}

function applyInflow(geo, reynolds, aoaDeg, T = AIR_T) {
  const nu = airKinematicViscosity(T);
  const inletSpeed = reynolds * nu;
  const aoaRad = (aoaDeg * Math.PI) / 180;
  const inletVx = inletSpeed * Math.cos(aoaRad);
  const inletVy = inletSpeed * Math.sin(aoaRad);

  const n = geo.position.length;
  const x = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) {
    x[i * 7 + 0] = geo.position[i][0];
    x[i * 7 + 1] = geo.position[i][1];
    x[i * 7 + 2] = inletVx;
    x[i * 7 + 3] = inletVy;
    x[i * 7 + 4] = geo.sdf[i];
    x[i * 7 + 5] = geo.normals[i][0];
    x[i * 7 + 6] = geo.normals[i][1];
  }
  return { x, n, position: geo.position, isSurface: geo.isSurface, surfaceNormals: geo.surfaceNormals, inletSpeed, aoaRad };
}

// ---------------------------------------------------------------------
// force integration (panel method, mirrors integrate_forces exactly)
// ---------------------------------------------------------------------

function integrateForces(surfacePos, surfaceNormalVecs, predSurface, offsetPred, eps, inletSpeed, aoaRad, T = AIR_T) {
  const RHO = airDensity(T);
  const NU = airKinematicViscosity(T);
  const M = surfacePos.length;

  const segLen = [];
  for (let i = 0; i < M; i++) {
    const j = (i + 1) % M;
    segLen.push(Math.hypot(surfacePos[j][0] - surfacePos[i][0], surfacePos[j][1] - surfacePos[i][1]));
  }
  const panelLen = [];
  for (let i = 0; i < M; i++) {
    const prev = (i - 1 + M) % M;
    panelLen.push(0.5 * (segLen[i] + segLen[prev]));
  }

  let FpX = 0,
    FpY = 0,
    FvX = 0,
    FvY = 0;
  for (let i = 0; i < M; i++) {
    const pressure = predSurface[i][2];
    const nx = surfaceNormalVecs[i][0],
      ny = surfaceNormalVecs[i][1];
    // normal is inward: -p*n_outward = -p*(-n_inward) = +p*n_inward
    FpX += pressure * nx * panelLen[i];
    FpY += pressure * ny * panelLen[i];

    const tx = -ny,
      ty = nx;
    const duDnX = (offsetPred[i][0] - predSurface[i][0]) / eps;
    const duDnY = (offsetPred[i][1] - predSurface[i][1]) / eps;
    const duTDn = duDnX * tx + duDnY * ty;
    FvX += NU * duTDn * tx * panelLen[i];
    FvY += NU * duTDn * ty * panelLen[i];
  }
  FpX *= RHO;
  FpY *= RHO;
  FvX *= RHO;
  FvY *= RHO;
  const Fx = FpX + FvX;
  const Fy = FpY + FvY;

  const cosA = Math.cos(aoaRad),
    sinA = Math.sin(aoaRad);
  const Fd = cosA * Fx + sinA * Fy;
  const Fl = -sinA * Fx + cosA * Fy;
  const q = 0.5 * RHO * inletSpeed ** 2;
  return { cd: Fd / q, cl: Fl / q };
}

window.AirfoilGeometry = {
  nacaAirfoil4Digit,
  parseDatText,
  resampleClose,
  surfaceNormals,
  generateGeometryCloud,
  applyInflow,
  integrateForces,
  checkEnvelope,
  airDensity,
  airKinematicViscosity,
};
