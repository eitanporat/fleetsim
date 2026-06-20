import { pickInterestingOp, renderArchitectureSvg, opLabel } from "./diagram.mjs?v=57";

const $ = (id) => document.getElementById(id);
const state = {
  rows: [],
  selected: null,
  op: null,
  outer: null,
  tensor: null,
  layoutView: "diagram",
  diagram: { sizes: true, internals: true, zoom: 1, fullscreen: false },
  sort: { key: "downloads", dir: -1 },
  filters: { q: "", arch: "", ops: new Set() }
};

const els = {
  count: $("count"),
  models: $("models"),
  splitter: $("splitter"),
  search: $("search"),
  arch: $("arch"),
  ops: $("ops"),
  reset: $("reset"),
  title: $("detail-title"),
  summary: $("summary"),
  dimensions: $("dimensions"),
  layoutTools: $("layout-tools"),
  layout: $("layout"),
  inspector: $("inspector")
};

const fmt = (n) => {
  if (n === undefined || n === null || Number.isNaN(Number(n))) return "";
  n = Number(n);
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return n.toLocaleString();
};
const full = (n) => (n === undefined || n === null || Number.isNaN(Number(n))) ? "" : Number(n).toLocaleString();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
const uniq = (values) => [...new Set(values.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
const layouts = (ir) => ir.blocks?.ranges ?? [{ range: [0, Math.max((ir.blocks?.count ?? 1) - 1, 0)], layout: ir.blocks?.shared_layout ?? [] }];
const params = (op) => op?.parameters ?? [];
const paramTotal = (op) => params(op).reduce((sum, p) => sum + Number(p.parameter_count ?? 0), 0);

const svgIcon = (p) => `<svg class="i" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p}</svg>`;
const ICON = {
  minus: svgIcon('<line x1="5" y1="12" x2="19" y2="12"/>'),
  plus: svgIcon('<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>'),
  expand: svgIcon('<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3"/>'),
  compress: svgIcon('<path d="M8 3v3a2 2 0 0 1-2 2H3M16 3v3a2 2 0 0 0 2 2h3M21 16h-3a2 2 0 0 0-2 2v3M3 16h3a2 2 0 0 1 2 2v3"/>'),
  download: svgIcon('<path d="M12 4v10"/><path d="M8 10.5l4 4 4-4"/><path d="M5 20h14"/>')
};

async function json(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${path}`);
  return res.json();
}

function optionList(select, values, label) {
  select.innerHTML = [`<option value="">All ${label}</option>`, ...values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`)].join("");
}

async function load() {
  const index = await json("../db/model-ir/index.json");
  state.rows = index.models.map((item) => ({
    id: item.id,
    file: item.file,
    ir: null,
    name: item.name ?? item.base_model ?? item.repo ?? item.id,
    baseModel: item.base_model,
    repo: item.repo,
    arch: item.architecture,
    params: item.parameter_count,
    blocks: item.block_count,
    context: item.context_length,
    downloads: item.downloads,
    likes: item.likes,
    hfRank: item.hf_rank,
    ops: item.ops ?? []
  }));
  const url = new URLSearchParams(location.search);
  const wanted = url.get("model");
  state.selected = (wanted && state.rows.find((r) => r.id === wanted || r.file === wanted || r.file === `${wanted}.json`)) || state.rows[0] || null;
  if (url.get("view")) state.layoutView = url.get("view") === "pattern" ? "pattern" : "diagram";
  if (url.get("sizes") === "0") state.diagram.sizes = false;
  if (url.get("internals") === "0") state.diagram.internals = false;
  if (url.get("zoom")) state.diagram.zoom = Math.max(0.6, Math.min(2.4, Number(url.get("zoom")) || 1));
  if (url.get("fullscreen") === "1") state.diagram.fullscreen = true;
  optionList(els.arch, uniq(state.rows.map((r) => r.arch)), "architectures");
  renderOpFacets();
  render();
  if (state.selected) {
    await loadIr(state.selected);
    applyUrlOpState(url);
    render();
  }
}

function applyUrlOpState(url) {
  const ir = state.selected?.ir;
  if (!ir) return;
  const groups = layoutGroups(layouts(ir));
  const range = (groups[Number(url.get("group") ?? 0)] ?? groups[0])?.ranges[0];
  if (range) state.op = { rangeIndex: range.index, opIndex: url.get("op") != null ? Number(url.get("op")) : pickInterestingOp(range.layout) };
  if (url.get("outer")) state.outer = url.get("outer");
}

function filteredRows() {
  const { q, arch, ops } = state.filters;
  return state.rows.filter((row) => {
    const haystack = [row.name, row.baseModel, row.repo, row.arch].join(" ").toLowerCase();
    return (!q || haystack.includes(q))
      && (!arch || row.arch === arch)
      && [...ops].every((op) => row.ops.includes(op));
  }).sort(compareRows);
}

function renderOpFacets() {
  const counts = new Map();
  for (const row of state.rows) for (const op of row.ops) counts.set(op, (counts.get(op) ?? 0) + 1);
  els.ops.innerHTML = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([op, count]) => `
    <label class="facet-option">
      <input type="checkbox" value="${esc(op)}" ${state.filters.ops.has(op) ? "checked" : ""}>
      <span>${esc(op)}</span>
      <small>${fmt(count)}</small>
    </label>
  `).join("");
}

function compareRows(a, b) {
  const { key, dir } = state.sort;
  const av = key === "ops" ? a.ops.length : a[key];
  const bv = key === "ops" ? b.ops.length : b[key];
  if (typeof av === "number" || typeof bv === "number") return dir * ((av ?? -Infinity) - (bv ?? -Infinity));
  return dir * String(av ?? "").localeCompare(String(bv ?? ""));
}

function render() {
  const rows = filteredRows();
  if (!rows.includes(state.selected)) state.selected = rows[0] ?? null;
  els.count.textContent = `${rows.length} / ${state.rows.length}`;
  els.reset.disabled = !(state.filters.q || state.filters.arch || state.filters.ops.size);
  els.models.innerHTML = rows.map((row) => `
    <tr data-id="${esc(row.id)}" class="${row === state.selected ? "active" : ""}">
      <td>${esc(row.name)}</td>
      <td>${fmt(row.downloads)}</td>
      <td>${fmt(row.likes)}</td>
      <td>${esc(row.arch)}</td>
      <td>${fmt(row.params)}</td>
      <td>${fmt(row.blocks)}</td>
      <td>${fmt(row.context)}</td>
      <td>${fmt(row.ops.length)}</td>
    </tr>
  `).join("");
  renderSort();
  renderDetails();
}

function renderSort() {
  document.querySelectorAll("th[data-sort]").forEach((th) => {
    th.dataset.dir = th.dataset.sort === state.sort.key ? (state.sort.dir > 0 ? "asc" : "desc") : "";
  });
}

async function loadIr(row) {
  if (!row || row.ir) return;
  row.ir = await json(`../db/model-ir/${row.file}`);
}

async function selectRow(id) {
  state.selected = state.rows.find((row) => row.id === id);
  state.op = null;
  render();
  await loadIr(state.selected);
  render();
}

function renderKv(el, entries) {
  el.innerHTML = entries.map(([key, value]) => `<div><b>${esc(key)}</b><span>${esc(value)}</span></div>`).join("");
}

function renderDetails() {
  const row = state.selected;
  if (!row) {
    els.title.textContent = "Architecture";
    els.summary.innerHTML = els.dimensions.innerHTML = els.layout.innerHTML = els.inspector.innerHTML = "";
    return;
  }
  if (!row.ir) {
    els.title.textContent = row.name;
    els.summary.innerHTML = `<div><b>loading</b><span>${esc(row.name)}</span></div>`;
    els.dimensions.innerHTML = els.layout.innerHTML = els.inspector.innerHTML = "";
    return;
  }
  const { ir } = row;
  els.title.textContent = row.name;
  const model = ir.model ?? {};
  renderKv(els.summary, [
    ["HF model", model.hf_name ?? model.base_model ?? row.name],
    ["GGUF repo", model.repo ?? row.repo],
    ["GGUF file", model.file],
    ["artifact name", model.artifact_name],
    ["architecture", model.architecture],
    ["parameters", full(model.parameter_count)],
    ["downloads", fmt(row.downloads)],
    ["likes", fmt(row.likes)]
  ].filter(([, value]) => value !== undefined && value !== ""));
  renderKv(els.dimensions, Object.entries(ir.dimensions ?? {}).map(([k, v]) => [k, fmt(v) || v]));
  renderLayout(ir);
  renderInspector();
}

function renderLayout(ir) {
  const ranges = layouts(ir);
  if (!ranges.some((range) => range.layout?.length)) {
    state.op = null;
    els.layoutTools.innerHTML = "";
    els.layout.innerHTML = `<div class="range warn"><div class="range-head"><b>No recognized block layout</b><span class="muted">${fmt(ir.blocks?.count)} declared blocks</span></div></div>`;
    return;
  }
  ensureOp(ranges);
  renderLayoutTools();
  const groups = layoutGroups(ranges);
  if (state.layoutView === "diagram") {
    renderDiagramLayout(ir, ranges, groups);
    return;
  }
  renderPatternLayout(groups);
}

function ensureOp(ranges) {
  if (state.op && ranges[state.op.rangeIndex]?.layout?.[state.op.opIndex]) return;
  const rangeIndex = ranges.findIndex((range) => range.layout?.length);
  state.op = rangeIndex >= 0 ? { rangeIndex, opIndex: 0 } : null;
}

function renderLayoutTools() {
  els.layoutTools.innerHTML = `
    <div class="view-tools">
      <div class="segmented" role="tablist" aria-label="Architecture view">
        ${["pattern", "diagram"].map((view) => `
          <button class="${state.layoutView === view ? "active" : ""}" data-view="${view}" type="button">${view === "pattern" ? "Patterns" : "Diagram"}</button>
        `).join("")}
      </div>
    </div>
  `;
}

function renderPatternLayout(groups) {
  els.layout.innerHTML = groups.map((group, groupIndex) => {
    const open = group.ranges.some((range) => range.index === state.op?.rangeIndex);
    const label = `Pattern ${groupIndex + 1}`;
    return `
      <details class="range" ${open ? "open" : ""}>
        <summary class="range-head">
          <b>${esc(label)}</b>
          <span class="muted">${fmt(group.blocks)} blocks | per block ${fmt(group.ranges[0].per_block_parameter_count)}</span>
        </summary>
        <div class="pattern-body">
          <div class="block-ranges">
            <b>Layers</b>
            ${group.ranges.map(({ range }) => `<span>${esc(rangeLabel(range))}</span>`).join("")}
          </div>
          <div class="layer-stack">
          ${(group.ranges[0].layout ?? []).map((op, opIndex) => `
            <button class="layer ${state.op?.rangeIndex === group.ranges[0].index && state.op?.opIndex === opIndex ? "selected" : ""}" data-range="${group.ranges[0].index}" data-op="${opIndex}" type="button">
              <b>${esc(op.op)}</b><small>${esc(op.name ?? "")}</small>
            </button>
          `).join("")}
          </div>
        </div>
      </details>
    `;
  }).join("");
}

function renderDiagramLayout(ir, ranges, groups) {
  let groupIndex = Math.max(0, groups.findIndex((group) => group.ranges.some((range) => range.index === state.op?.rangeIndex)));
  const group = groups[groupIndex] ?? groups[0];
  if (!group.ranges.some((range) => range.index === state.op?.rangeIndex)) {
    state.op = { rangeIndex: group.ranges[0].index, opIndex: 0 };
    groupIndex = 0;
  }
  const range = ranges[state.op.rangeIndex];
  els.layout.innerHTML = `
    <div class="diagram-shell ${state.diagram.fullscreen ? "fullscreen" : ""}">
      ${state.diagram.fullscreen ? `<button class="diagram-exit icon-btn" data-diagram-exit type="button" aria-label="Exit fullscreen" title="Exit fullscreen">${ICON.compress}</button>` : ""}
      <div class="diagram-patterns">
        ${groups.map((item, index) => `
          <button class="pattern-select ${index === groupIndex ? "active" : ""}" data-range="${item.ranges[0].index}" data-op="${pickInterestingOp(item.ranges[0].layout)}" type="button">
            <b>Pattern ${index + 1}</b>
            <span>${fmt(item.blocks)} layers · ${fmt(item.ranges[0].per_block_parameter_count)} / layer</span>
          </button>
        `).join("")}
      </div>
      <div class="diagram-range-strip">
        <b>Layers</b>
        ${group.ranges.map(({ range }) => `<span>${esc(rangeLabel(range))}</span>`).join("")}
      </div>
      <div class="diagram-tools" role="toolbar" aria-label="Diagram controls">
        <label class="chip"><input type="checkbox" data-diagram="sizes" ${state.diagram.sizes ? "checked" : ""}><span>Sizes</span></label>
        <label class="chip"><input type="checkbox" data-diagram="internals" ${state.diagram.internals ? "checked" : ""}><span>Internals</span></label>
        
        <div class="zoom-group" role="group" aria-label="Zoom">
          <button class="icon-btn" data-zoom="-1" type="button" aria-label="Zoom out" title="Zoom out">${ICON.minus}</button>
          <button class="zoom-val" data-zoom="0" type="button" aria-label="Reset zoom to 100%" title="Reset zoom">${Math.round(state.diagram.zoom * 100)}%</button>
          <button class="icon-btn" data-zoom="1" type="button" aria-label="Zoom in" title="Zoom in">${ICON.plus}</button>
        </div>
        <button class="icon-btn" data-fullscreen type="button" aria-label="Fullscreen" title="Fullscreen">${ICON.expand}</button>
        <div class="zoom-group export-group" role="group" aria-label="Download diagram">
          <button class="icon-btn labeled" data-export="svg" type="button" aria-label="Download SVG" title="Download SVG">${ICON.download}<span>SVG</span></button>
          <button class="icon-btn labeled" data-export="png" type="button" aria-label="Download PNG" title="Download PNG">${ICON.download}<span>PNG</span></button>
        </div>
      </div>
      <div class="diagram-canvas" style="--diagram-width:${Math.round(state.diagram.zoom * 100)}%">${renderDiagramSvg(ir, range, state.op.rangeIndex, groupIndex)}</div>
    </div>
  `;
}

function renderDiagramSvg(ir, range, rangeIndex, groupIndex) {
  return renderArchitectureSvg(ir, {
    range,
    rangeIndex,
    groupIndex,
    selectedOpIndex: state.op?.opIndex,
    selectedOuter: state.outer,
    showSizes: state.diagram.sizes,
    showInternals: state.diagram.internals,
    interactive: true
  });
}

function exportName(ext) {
  const name = state.selected?.baseModel ?? state.selected?.name ?? state.selected?.id ?? "model";
  return `${String(name).replace(/[^\w.-]+/g, "_")}-pattern-${(state.op?.rangeIndex ?? 0) + 1}.${ext}`;
}

function download(blob, filename) {
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: filename });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function exportDiagram(type) {
  const svg = els.layout.querySelector(".diagram-canvas svg");
  if (!svg) return;
  const text = new XMLSerializer().serializeToString(svg);
  if (type === "svg") return download(new Blob([text], { type: "image/svg+xml" }), exportName("svg"));
  const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
  const img = await new Promise((resolve, reject) => Object.assign(new Image(), { onload() { resolve(this); }, onerror: reject, src: url }));
  const w = Math.ceil(svg.viewBox.baseVal.width || Number(svg.getAttribute("width"))), h = Math.ceil(svg.viewBox.baseVal.height || Number(svg.getAttribute("height")));
  const canvas = Object.assign(document.createElement("canvas"), { width: w * 2, height: h * 2 });
  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  ctx.fillStyle = "#fbfbfc";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  URL.revokeObjectURL(url);
  canvas.toBlob((blob) => blob && download(blob, exportName("png")), "image/png");
}

function layoutGroups(ranges) {
  const groups = new Map();
  ranges.forEach((range, index) => {
    const key = JSON.stringify((range.layout ?? []).map((op) => [op.op, op.name]));
    if (!groups.has(key)) groups.set(key, { ranges: [], blocks: 0 });
    const group = groups.get(key);
    group.ranges.push({ ...range, index });
    group.blocks += range.range[1] - range.range[0] + 1;
  });
  return [...groups.values()];
}

function rangeLabel(range) {
  return range[0] === range[1] ? `${range[0]}` : `${range[0]}-${range[1]}`;
}

// the full weight list for the current model: embedding · every op in the block · output
function inspectorOps(ir) {
  const range = state.op ? layouts(ir)[state.op.rangeIndex] : layouts(ir)[0];
  const out = [];
  (ir.embeddings ?? []).forEach((op) => out.push({ key: "embedding", op, label: "Token embedding" }));
  (range?.layout ?? []).forEach((op, i) => out.push({ key: `op${i}`, op, label: opLabel(op), opIndex: i }));
  (ir.output ?? []).forEach((op) => out.push({ key: op.op === "rms_norm" ? "final_norm" : "head", op, label: op.op === "rms_norm" ? "Final norm" : "Output head" }));
  return out;
}

function renderInspector() {
  const ir = state.selected?.ir;
  if (!ir) { els.inspector.innerHTML = `<p class="muted">No model selected.</p>`; return; }
  const entries = inspectorOps(ir);
  const selKey = state.outer && state.outer !== "input" ? state.outer : (state.op ? `op${state.op.opIndex}` : null);
  const sel = entries.find((e) => e.key === selKey);
  const ordered = sel ? [sel, ...entries.filter((e) => e !== sel)] : entries;
  const tn = (p) => p.name.replace(/^blk\.(N|\d+)\./, "").replace(/\.(weight|bias)$/, "");
  const card = (e) => {
    const isSel = e === sel;
    const ps = params(e.op), cfg = Object.entries(e.op.config ?? {});
    const rows = ps.map((p) => {
      const focus = isSel && state.tensor && tn(p) === state.tensor ? " tensor-focus" : "";
      const note = [p.instances ? `${p.instances}×` : "", p.tied_to ? `tied → ${p.tied_to}` : ""].filter(Boolean).join(" ");
      return `<tr class="weight-row${focus}" data-tn="${esc(tn(p))}"><td>${esc(p.name)}${note ? ` <span class="wr-note">${esc(note)}</span>` : ""}</td><td>${esc((p.shape ?? []).join(" × "))}</td><td class="marker" title="${full(p.parameter_count)} parameters">${full(p.parameter_count)}</td></tr>`;
    }).join("");
    const cfgRows = cfg.map(([k, v]) => `<tr class="cfg-row"><td>config.${esc(k)}</td><td colspan="2">${esc(v)}</td></tr>`).join("");
    const body = ps.length || cfg.length
      ? `<table class="oc-table"><tbody>${rows}${cfgRows}</tbody></table>`
      : `<p class="muted oc-empty">no learned parameters</p>`;
    return `<details class="op-card${isSel ? " op-sel" : ""}"${isSel ? " open" : ""}>
      <summary><span class="oc-label">${esc(e.label)}</span><span class="oc-meta">${full(paramTotal(e.op))} params · ${ps.length} ${ps.length === 1 ? "tensor" : "tensors"}</span></summary>
      ${body}
    </details>`;
  };
  els.inspector.innerHTML = `
    <div class="inspector-head">
      <div><b>${esc(sel ? sel.label : (ir.model?.architecture ?? "Model"))}</b> ${sel ? `<span class="muted">${esc(sel.op.op)}</span>` : `<span class="muted">${entries.length} groups</span>`}</div>
      <div class="param-badge" title="${full(sel ? paramTotal(sel.op) : ir.model?.parameter_count)} parameters">${full(sel ? paramTotal(sel.op) : ir.model?.parameter_count)} params</div>
    </div>
    <div class="op-list">${ordered.map(card).join("")}</div>
  `;
}

function focusInspector() {
  // Flash the inspector to signal it updated — but never scroll the page (jarring).
  // If a specific weight is focused, reveal it WITHIN the inspector list only.
  requestAnimationFrame(() => {
    els.inspector.classList.remove("flash");
    void els.inspector.offsetWidth;
    els.inspector.classList.add("flash");
    const row = els.inspector.querySelector(".tensor-focus");
    const list = els.inspector.querySelector(".op-list");
    if (row && list) list.scrollTop = Math.max(0, row.offsetTop - list.offsetTop - 56);
  });
}

function renderOpSelection() {
  els.layout.querySelectorAll(".layer").forEach((button) => {
    button.classList.toggle(
      "selected",
      Number(button.dataset.range) === state.op?.rangeIndex && Number(button.dataset.op) === state.op?.opIndex
    );
  });
  els.layout.querySelectorAll(".diagram-node").forEach((node) => {
    node.classList.toggle(
      "selected",
      Number(node.dataset.range) === state.op?.rangeIndex && Number(node.dataset.op) === state.op?.opIndex
    );
  });
}

els.models.addEventListener("click", (event) => {
  const tr = event.target.closest("tr[data-id]");
  if (tr) selectRow(tr.dataset.id);
});

document.querySelector("thead").addEventListener("click", (event) => {
  const th = event.target.closest("th[data-sort]");
  if (!th) return;
  const key = th.dataset.sort;
  state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : key === "name" || key === "arch" ? 1 : -1 };
  render();
});

els.layoutTools.addEventListener("click", (event) => {
  const view = event.target.closest("button[data-view]")?.dataset.view;
  if (!view) return;
  state.layoutView = view;
  if (view === "diagram" && state.selected?.ir) {
    const range = layouts(state.selected.ir)[state.op?.rangeIndex ?? 0];
    state.op = { rangeIndex: state.op?.rangeIndex ?? 0, opIndex: pickInterestingOp(range?.layout) };
  }
  renderDetails();
});

function renderDetailsKeepScroll() {
  const before = els.layout.querySelector(".diagram-canvas");
  const sx = before ? before.scrollLeft : 0, sy = before ? before.scrollTop : 0;
  renderDetails();
  const after = els.layout.querySelector(".diagram-canvas");
  if (after) { after.scrollLeft = sx; after.scrollTop = sy; }
}

els.layout.addEventListener("click", (event) => {
  const zoom = event.target.closest("button[data-zoom]")?.dataset.zoom;
  const fullscreen = event.target.closest("button[data-fullscreen]");
  const exportType = event.target.closest("button[data-export]")?.dataset.export;
  if (event.target.closest("[data-diagram-exit]")) {
    state.diagram.fullscreen = false;
    renderDetails();
    return;
  }
  if (zoom) {
    state.diagram.zoom = zoom === "0" ? 1 : Math.max(.6, Math.min(2.4, state.diagram.zoom + Number(zoom) * .2));
    renderDetailsKeepScroll();
    return;
  }
  if (fullscreen) {
    state.diagram.fullscreen = !state.diagram.fullscreen;
    renderDetails();
    requestAnimationFrame(() => els.layout.querySelector(".diagram-canvas")?.scrollTo(0, 0));
    return;
  }
  if (exportType) {
    exportDiagram(exportType);
    return;
  }
  const tensor = event.target.closest("[data-tensor]")?.dataset.tensor;
  if (tensor) {
    state.tensor = tensor;
    renderInspector();
    focusInspector();
    return;
  }
  const outer = event.target.closest("[data-outer]")?.dataset.outer;
  if (outer) {
    state.outer = outer;
    state.tensor = null;
    renderDetailsKeepScroll();
    focusInspector();
    return;
  }
  const target = event.target.closest("[data-range][data-op]");
  if (!target) return;
  state.op = { rangeIndex: Number(target.dataset.range), opIndex: Number(target.dataset.op) };
  state.outer = null;
  state.tensor = null;
  if (state.layoutView === "diagram") renderDetailsKeepScroll();
  else renderOpSelection();
  renderInspector();
  focusInspector();
});

els.layout.addEventListener("change", (event) => {
  const key = event.target.closest("input[data-diagram]")?.dataset.diagram;
  if (!key) return;
  state.diagram[key] = event.target.checked;
  renderDetailsKeepScroll();
});

els.arch.addEventListener("change", () => {
  state.filters.arch = els.arch.value;
  state.op = null;
  render();
});

els.ops.addEventListener("change", (event) => {
  const input = event.target.closest("input[type=checkbox]");
  if (!input) return;
  if (input.checked) state.filters.ops.add(input.value);
  else state.filters.ops.delete(input.value);
  state.op = null;
  render();
});

els.search.addEventListener("input", () => {
  state.filters.q = els.search.value.trim().toLowerCase();
  render();
});

els.reset.addEventListener("click", () => {
  state.filters = { q: "", arch: "", ops: new Set() };
  els.search.value = els.arch.value = "";
  renderOpFacets();
  render();
});

els.splitter.addEventListener("pointerdown", (event) => {
  const startX = event.clientX;
  const startWidth = document.querySelector(".registry").getBoundingClientRect().width;
  els.splitter.classList.add("dragging");
  els.splitter.setPointerCapture(event.pointerId);

  const move = (moveEvent) => {
    const next = Math.max(320, Math.min(900, startWidth + moveEvent.clientX - startX));
    document.querySelector("main").style.setProperty("--registry-width", `${next}px`);
  };
  const up = () => {
    els.splitter.classList.remove("dragging");
    els.splitter.removeEventListener("pointermove", move);
    els.splitter.removeEventListener("pointerup", up);
    els.splitter.removeEventListener("pointercancel", up);
  };

  els.splitter.addEventListener("pointermove", move);
  els.splitter.addEventListener("pointerup", up);
  els.splitter.addEventListener("pointercancel", up);
});

load().catch((error) => {
  els.models.innerHTML = `<tr><td colspan="8">${esc(error.message)}</td></tr>`;
});
