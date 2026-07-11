const treeEl = document.getElementById("file-tree");
const crumbsEl = document.getElementById("breadcrumbs");
const previewEl = document.getElementById("file-preview");
const toolbarEl = document.getElementById("preview-toolbar");
const downloadLink = document.getElementById("download-link");
const previewMeta = document.getElementById("preview-meta");

// Plain text glyphs, not emoji -- emoji render in their own fixed colors
// regardless of CSS, which would break the monochrome theme.
const ICON_DIR = "▸";
const ICON_FILE = "·";

function fmtSize(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}

function renderBreadcrumbs(path) {
  const parts = path ? path.split("/") : [];
  let acc = "";
  const crumbs = [`<a href="#" data-path="">repo root</a>`];
  for (const p of parts) {
    acc = acc ? acc + "/" + p : p;
    crumbs.push(`<a href="#" data-path="${acc}">${p}</a>`);
  }
  crumbsEl.innerHTML = crumbs.join(" / ");
  crumbsEl.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      loadDir(a.dataset.path);
    })
  );
}

async function loadDir(path) {
  renderBreadcrumbs(path);
  treeEl.innerHTML = `<span class="loading">Loading…</span>`;
  const res = await fetch("/api/files?path=" + encodeURIComponent(path));
  if (!res.ok) {
    treeEl.innerHTML = `<span class="loading">Failed to load directory.</span>`;
    return;
  }
  const data = await res.json();
  if (data.type !== "dir") {
    // path pointed at a file; fall back to its parent dir
    const parent = path.split("/").slice(0, -1).join("/");
    return loadDir(parent);
  }
  treeEl.innerHTML = "";
  for (const entry of data.entries) {
    const row = document.createElement("div");
    row.className = "file-row";
    row.innerHTML = `<span class="icon">${entry.is_dir ? ICON_DIR : ICON_FILE}</span><span>${entry.name}</span><span class="size">${fmtSize(entry.size)}</span>`;
    row.addEventListener("click", () => (entry.is_dir ? loadDir(entry.path) : loadFile(entry.path)));
    treeEl.appendChild(row);
  }
  if (data.entries.length === 0) {
    treeEl.innerHTML = `<span class="loading">Empty directory.</span>`;
  }
}

async function loadFile(path) {
  previewEl.innerHTML = `<span class="loading">Loading…</span>`;
  toolbarEl.style.display = "";
  downloadLink.href = "/api/files/download?path=" + encodeURIComponent(path);
  const res = await fetch("/api/files?path=" + encodeURIComponent(path));
  const data = await res.json();
  previewMeta.textContent = path + "  ·  " + fmtSize(data.size);
  if (data.too_large) {
    previewEl.innerHTML = `<span class="loading">File is ${fmtSize(data.size)} -- too large to preview. Use Download.</span>`;
  } else if (data.binary) {
    previewEl.innerHTML = `<span class="loading">Binary file -- use Download.</span>`;
  } else {
    previewEl.textContent = data.content;
  }
}

loadDir("");
