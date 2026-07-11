// Fully client-side Tool page: geometry (geometry.js) + ONNX Runtime Web
// running the exported MLP (assets/model.onnx) + Canvas2D field rendering.
// No backend calls at all -- this is what makes the site GitHub-Pages-postable.
'use strict';

const G = window.AirfoilGeometry;
let NORM_STATS = null;
let SESSION = null;

const els = {
  nacaCode: document.getElementById('naca-code'),
  datFile: document.getElementById('dat-file'),
  nacaInput: document.getElementById('naca-input'),
  datInput: document.getElementById('dat-input'),
  reynolds: document.getElementById('reynolds'),
  reynoldsValue: document.getElementById('reynolds-value'),
  aoa: document.getElementById('aoa'),
  aoaValue: document.getElementById('aoa-value'),
  warnings: document.getElementById('warnings'),
  clValue: document.getElementById('cl-value'),
  cdValue: document.getElementById('cd-value'),
  canvas: document.getElementById('field-canvas'),
  status: document.getElementById('canvas-status'),
  colorbarMin: document.getElementById('colorbar-min'),
  colorbarMax: document.getElementById('colorbar-max'),
};

const SEQ_STOPS_LIGHT = [
  [0.0, [232, 240, 253]],
  [1 / 6, [158, 197, 244]],
  [2 / 6, [109, 167, 236]],
  [3 / 6, [57, 135, 229]],
  [4 / 6, [37, 106, 191]],
  [5 / 6, [24, 79, 149]],
  [1.0, [13, 54, 107]],
];
const SEQ_STOPS_DARK = [
  [0.0, [16, 25, 43]],
  [1 / 6, [28, 58, 99]],
  [2 / 6, [39, 87, 144]],
  [3 / 6, [57, 135, 229]],
  [4 / 6, [85, 152, 231]],
  [5 / 6, [134, 182, 239]],
  [1.0, [205, 226, 251]],
];
function seqStops() {
  const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  return isDark ? SEQ_STOPS_DARK : SEQ_STOPS_LIGHT;
}
function seqColor(t, stops) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  const last = stops[stops.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

async function init() {
  els.status.textContent = 'loading model…';
  const [stats, session] = await Promise.all([
    fetch('assets/norm_stats.json').then((r) => r.json()),
    ort.InferenceSession.create('assets/model.onnx'),
  ]);
  NORM_STATS = stats;
  SESSION = session;
  els.status.textContent = '';
  await predict();
}

/** x: Float32Array of shape (n,7), row-major. Returns Float32Array (n,4), un-normalized. */
async function runModel(x, n) {
  const xm = NORM_STATS.x_mean,
    xs = NORM_STATS.x_std;
  const norm = new Float32Array(n * 7);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 7; j++) {
      norm[i * 7 + j] = (x[i * 7 + j] - xm[j]) / xs[j];
    }
  }
  const tensor = new ort.Tensor('float32', norm, [n, 7]);
  const out = await SESSION.run({ point_features: tensor });
  const predNorm = out.predictions.data;

  const ym = NORM_STATS.y_mean,
    ys = NORM_STATS.y_std;
  const pred = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 4; j++) {
      pred[i * 4 + j] = predNorm[i * 4 + j] * ys[j] + ym[j];
    }
  }
  return pred;
}

function getSource() {
  return document.querySelector('input[name="source"]:checked').value;
}
function getField() {
  return document.querySelector('input[name="field"]:checked').value;
}

let debounceTimer = null;
function schedulePredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, 200);
}

els.reynolds.addEventListener('input', () => {
  els.reynoldsValue.textContent = (els.reynolds.value / 1e6).toFixed(1) + 'e6';
  schedulePredict();
});
els.aoa.addEventListener('input', () => {
  els.aoaValue.textContent = (+els.aoa.value).toFixed(1) + '°';
  schedulePredict();
});
document.querySelectorAll('input[name="source"]').forEach((r) =>
  r.addEventListener('change', () => {
    els.nacaInput.style.display = r.value === 'naca' && r.checked ? '' : 'none';
    els.datInput.style.display = r.value === 'dat' && r.checked ? '' : 'none';
    if (r.checked) schedulePredict();
  })
);
document.querySelectorAll('input[name="field"]').forEach((r) => r.addEventListener('change', () => schedulePredict()));
els.nacaCode.addEventListener('input', () => schedulePredict());
els.datFile.addEventListener('change', () => schedulePredict());

let requestSeq = 0;

async function predict() {
  if (!SESSION) return;
  const mySeq = ++requestSeq;
  const source = getSource();
  const reynolds = +els.reynolds.value;
  const aoa = +els.aoa.value;
  const warnings = G.checkEnvelope(reynolds, aoa);

  let rawCoords;
  if (source === 'naca') {
    const code = els.nacaCode.value.trim();
    if (!code) return;
    if (code.length !== 4) {
      els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>Only 4-digit NACA codes are supported in this fully client-side build.</span></div>`;
      return;
    }
    try {
      rawCoords = G.nacaAirfoil4Digit(code, 200);
    } catch (e) {
      els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>${e.message}</span></div>`;
      return;
    }
  } else {
    const file = els.datFile.files[0];
    if (!file) return;
    const text = await file.text();
    rawCoords = G.parseDatText(text);
    warnings.push('Arbitrary .dat geometry -- accuracy degrades outside the NACA 4/5-digit family the model trained on.');
    if (rawCoords.length < 10) {
      els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>Could not parse enough points from that .dat file.</span></div>`;
      return;
    }
  }

  els.status.textContent = 'predicting…';
  try {
    const { coords } = G.resampleClose(rawCoords, 300);
    const geo = G.generateGeometryCloud(coords, 9000, [[-2, 4], [-1.5, 1.5]], 0);
    const cloud = G.applyInflow(geo, reynolds, aoa);

    const pred = await runModel(cloud.x, cloud.n);
    if (mySeq !== requestSeq) return;

    // surface subset for force integration
    const surfIdx = [];
    for (let i = 0; i < cloud.n; i++) if (cloud.isSurface[i]) surfIdx.push(i);
    const surfacePos = surfIdx.map((i) => cloud.position[i]);
    const predSurface = surfIdx.map((i) => [pred[i * 4], pred[i * 4 + 1], pred[i * 4 + 2], pred[i * 4 + 3]]);
    const surfaceNormalVecs = cloud.surfaceNormals;

    const eps = 0.01;
    const nSurf = surfacePos.length;
    const offsetX = new Float32Array(nSurf * 7);
    const inletVx = cloud.x[2],
      inletVy = cloud.x[3];
    for (let i = 0; i < nSurf; i++) {
      const [nx, ny] = surfaceNormalVecs[i];
      offsetX[i * 7 + 0] = surfacePos[i][0] - nx * eps;
      offsetX[i * 7 + 1] = surfacePos[i][1] - ny * eps;
      offsetX[i * 7 + 2] = inletVx;
      offsetX[i * 7 + 3] = inletVy;
      offsetX[i * 7 + 4] = eps;
      offsetX[i * 7 + 5] = 0;
      offsetX[i * 7 + 6] = 0;
    }
    const offsetPredFlat = await runModel(offsetX, nSurf);
    const offsetPred = [];
    for (let i = 0; i < nSurf; i++) offsetPred.push([offsetPredFlat[i * 4], offsetPredFlat[i * 4 + 1], offsetPredFlat[i * 4 + 2], offsetPredFlat[i * 4 + 3]]);
    if (mySeq !== requestSeq) return;

    const { cd, cl } = G.integrateForces(surfacePos, surfaceNormalVecs, predSurface, offsetPred, eps, cloud.inletSpeed, cloud.aoaRad);

    render({ cl, cd, warnings, position: cloud.position, pred, n: cloud.n });
  } catch (e) {
    console.error(e);
    els.status.textContent = '';
    els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>${e.message || e}</span></div>`;
  }
}

function render({ cl, cd, warnings, position, pred, n }) {
  els.status.textContent = '';
  els.clValue.textContent = cl.toFixed(3);
  els.cdValue.textContent = cd.toFixed(4);
  els.warnings.innerHTML = warnings.map((w) => `<div class="warning-banner"><span class="icon">!</span><span>${w}</span></div>`).join('');

  const field = getField();
  const values = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    values[i] = field === 'pressure' ? pred[i * 4 + 2] : Math.hypot(pred[i * 4], pred[i * 4 + 1]);
  }
  drawField(position, values);
}

function drawField(positions, values) {
  const canvas = els.canvas;
  const ctx = canvas.getContext('2d');
  const W = canvas.width,
    H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x, y] of positions) {
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  let vmin = Infinity, vmax = -Infinity;
  for (const v of values) {
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  els.colorbarMin.textContent = vmin.toFixed(1);
  els.colorbarMax.textContent = vmax.toFixed(1);

  const margin = 20;
  const scale = Math.min((W - 2 * margin) / (xmax - xmin), (H - 2 * margin) / (ymax - ymin));
  const cx = W / 2 - ((xmax + xmin) / 2) * scale;
  const cy = H / 2 + ((ymax + ymin) / 2) * scale;

  const stops = seqStops();
  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    const px = cx + x * scale;
    const py = cy - y * scale;
    const t = (values[i] - vmin) / (vmax - vmin || 1);
    ctx.fillStyle = seqColor(t, stops);
    ctx.fillRect(px, py, 2, 2);
  }
}

init();
