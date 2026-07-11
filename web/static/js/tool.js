// Tool page: form handling, live prediction, canvas field rendering.
// Grayscale sequential colormap -- monochrome theme, so magnitude is encoded
// as pure luminance instead of hue (matches the CSS colorbar gradient's
// --seq-0..--seq-6 custom properties). Grayscale ramps are inherently
// colorblind-safe since there's no hue to confuse. Mode-aware: light mode
// runs light->dark for low->high, dark mode runs dark->light, matching each
// surface's own contrast direction.
const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const SEQ_STOPS = isDark
  ? [
      [0.0, [20, 20, 20]], // --seq-0 #141414
      [1 / 6, [48, 48, 48]], // --seq-1 #303030
      [2 / 6, [92, 92, 92]], // --seq-2 #5c5c5c
      [3 / 6, [137, 137, 137]], // --seq-3 #898989
      [4 / 6, [182, 182, 182]], // --seq-4 #b6b6b6
      [5 / 6, [224, 224, 224]], // --seq-5 #e0e0e0
      [1.0, [255, 255, 255]], // --seq-6 #ffffff
    ]
  : [
      [0.0, [242, 242, 242]], // --seq-0 #f2f2f2
      [1 / 6, [207, 207, 207]], // --seq-1 #cfcfcf
      [2 / 6, [163, 163, 163]], // --seq-2 #a3a3a3
      [3 / 6, [118, 118, 118]], // --seq-3 #767676
      [4 / 6, [74, 74, 74]], // --seq-4 #4a4a4a
      [5 / 6, [35, 35, 35]], // --seq-5 #232323
      [1.0, [0, 0, 0]], // --seq-6 #000000
    ];

function seqColor(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < SEQ_STOPS.length - 1; i++) {
    const [t0, c0] = SEQ_STOPS[i];
    const [t1, c1] = SEQ_STOPS[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + f * (c1[0] - c0[0]));
      const g = Math.round(c0[1] + f * (c1[1] - c0[1]));
      const b = Math.round(c0[2] + f * (c1[2] - c0[2]));
      return `rgb(${r},${g},${b})`;
    }
  }
  const last = SEQ_STOPS[SEQ_STOPS.length - 1][1];
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

const els = {
  modelSelect: document.getElementById("model-select"),
  nacaCode: document.getElementById("naca-code"),
  datFile: document.getElementById("dat-file"),
  nacaInput: document.getElementById("naca-input"),
  datInput: document.getElementById("dat-input"),
  reynolds: document.getElementById("reynolds"),
  reynoldsValue: document.getElementById("reynolds-value"),
  aoa: document.getElementById("aoa"),
  aoaValue: document.getElementById("aoa-value"),
  warnings: document.getElementById("warnings"),
  clValue: document.getElementById("cl-value"),
  cdValue: document.getElementById("cd-value"),
  canvas: document.getElementById("field-canvas"),
  status: document.getElementById("canvas-status"),
  colorbarMin: document.getElementById("colorbar-min"),
  colorbarMax: document.getElementById("colorbar-max"),
};

let debounceTimer = null;
let requestSeq = 0;

function fmtReynolds(v) {
  return (v / 1e6).toFixed(1) + "e6";
}

els.reynolds.addEventListener("input", () => {
  els.reynoldsValue.textContent = fmtReynolds(+els.reynolds.value);
  schedulePredict();
});
els.aoa.addEventListener("input", () => {
  els.aoaValue.textContent = (+els.aoa.value).toFixed(1) + "°";
  schedulePredict();
});
document.querySelectorAll('input[name="source"]').forEach((r) =>
  r.addEventListener("change", () => {
    els.nacaInput.style.display = r.value === "naca" && r.checked ? "" : "none";
    els.datInput.style.display = r.value === "dat" && r.checked ? "" : "none";
    if (r.checked) schedulePredict();
  })
);
document.querySelectorAll('input[name="field"]').forEach((r) => r.addEventListener("change", () => schedulePredict()));
els.nacaCode.addEventListener("input", () => schedulePredict());
els.datFile.addEventListener("change", () => schedulePredict());
els.modelSelect.addEventListener("change", () => schedulePredict());

function schedulePredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, 200);
}

function getSource() {
  return document.querySelector('input[name="source"]:checked').value;
}
function getField() {
  return document.querySelector('input[name="field"]:checked').value;
}

async function loadModels() {
  const res = await fetch("/api/models");
  const models = await res.json();
  els.modelSelect.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label + (m.tool_default ? " (default)" : "");
    if (m.tool_default) opt.selected = true;
    els.modelSelect.appendChild(opt);
  }
}

async function predict() {
  const source = getSource();
  const body = {
    model_id: els.modelSelect.value,
    source,
    reynolds: +els.reynolds.value,
    aoa: +els.aoa.value,
  };
  if (source === "naca") {
    if (!els.nacaCode.value.trim()) return;
    body.naca_code = els.nacaCode.value.trim();
  } else {
    const file = els.datFile.files[0];
    if (!file) return;
    body.dat_text = await file.text();
  }

  const mySeq = ++requestSeq;
  els.status.textContent = "predicting…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (mySeq !== requestSeq) return; // a newer request superseded this one
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      els.status.textContent = "";
      els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>${err.detail || "prediction failed"}</span></div>`;
      return;
    }
    const data = await res.json();
    if (mySeq !== requestSeq) return;
    render(data);
  } catch (e) {
    if (mySeq !== requestSeq) return;
    els.status.textContent = "";
    els.warnings.innerHTML = `<div class="warning-banner"><span class="icon">!</span><span>${e}</span></div>`;
  }
}

function render(data) {
  els.status.textContent = "";
  els.clValue.textContent = data.cl.toFixed(3);
  els.cdValue.textContent = data.cd.toFixed(4);

  els.warnings.innerHTML = data.warnings
    .map((w) => `<div class="warning-banner"><span class="icon">!</span><span>${w}</span></div>`)
    .join("");

  const field = getField();
  let values;
  if (field === "pressure") {
    values = data.pressure;
  } else {
    values = data.velocity.map(([vx, vy]) => Math.sqrt(vx * vx + vy * vy));
  }

  drawField(data.position, values);
}

function drawField(positions, values) {
  const canvas = els.canvas;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  let xmin = Infinity,
    xmax = -Infinity,
    ymin = Infinity,
    ymax = -Infinity;
  for (const [x, y] of positions) {
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  let vmin = Infinity,
    vmax = -Infinity;
  for (const v of values) {
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  els.colorbarMin.textContent = vmin.toFixed(1);
  els.colorbarMax.textContent = vmax.toFixed(1);

  const margin = 20;
  const scale = Math.min((W - 2 * margin) / (xmax - xmin), (H - 2 * margin) / (ymax - ymin));
  const cx = W / 2 - ((xmax + xmin) / 2) * scale;
  const cy = H / 2 + ((ymax + ymin) / 2) * scale; // flip y (screen y grows downward)

  for (let i = 0; i < positions.length; i++) {
    const [x, y] = positions[i];
    const px = cx + x * scale;
    const py = cy - y * scale;
    const t = (values[i] - vmin) / (vmax - vmin || 1);
    ctx.fillStyle = seqColor(t);
    ctx.fillRect(px, py, 2, 2);
  }
}

loadModels().then(predict);
