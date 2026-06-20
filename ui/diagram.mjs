export const fmt = (n) => {
  if (n === undefined || n === null || Number.isNaN(Number(n))) return "";
  n = Number(n);
  if (Math.abs(n) >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (Math.abs(n) >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(n % 1e3 ? 1 : 0)}k`;
  return n.toLocaleString();
};

export const full = (n) => (n === undefined || n === null || Number.isNaN(Number(n))) ? "" : Number(n).toLocaleString();
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[c]);
export const layouts = (ir) => ir.blocks?.ranges ?? [{ range: [0, Math.max((ir.blocks?.count ?? 1) - 1, 0)], layout: ir.blocks?.shared_layout ?? [] }];
export const params = (op) => op?.parameters ?? [];
export const paramTotal = (op) => params(op).reduce((sum, p) => sum + Number(p.parameter_count ?? 0), 0);
export const rangeLabel = (range) => range[0] === range[1] ? `${range[0]}` : `${range[0]}-${range[1]}`;

// parameters arrive as [{name, shape, parameter_count}]; sketches want a clean-name map.
function paramMap(op) {
  const m = {};
  for (const p of params(op)) {
    const clean = p.name.replace(/^blk\.(N|\d+)\./, "").replace(/\.(weight|bias)$/, "");
    if (!(clean in m) || /\.weight$/.test(p.name)) m[clean] = p;
  }
  return m;
}
const shp = (p, name) => (p && p[name] && p[name].shape) ? p[name].shape : null;
// "in → out · params" for a weight (gguf stores [in, out, ...]); appends the tensor's parameter count.
const io = (p, name) => {
  const e = p && p[name];
  if (!e || !e.shape) return "";
  const s = e.shape;
  const sh = s.length >= 2 ? `${fmt(s[0])} → ${fmt(s[1])}` : `${fmt(s[0])}`;
  return e.parameter_count ? `${sh} · ${fmt(e.parameter_count)}` : sh;
};

export function layoutGroups(ranges) {
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

const ATTN = /attention|deltanet/;
const FFN = /moe|mlp/;

export function pickInterestingOp(layout = []) {
  const order = ["dsa_mla_attention", "mla_attention", "kimi_delta_attention", "gated_deltanet", "self_attention", "fused_qkv_attention", "state_space", "moe_mlp", "mlp", "fused_mlp"];
  for (const name of order) {
    const i = layout.findIndex((item) => item.op === name);
    if (i >= 0) return i;
  }
  return Math.max(0, layout.findIndex((op) => op.op !== "rms_norm" && op.op !== "residual_add"));
}

export function opLabel(op, ctx = {}) {
  const o = op?.op;
  if (o === "self_attention") return ctx.gqa ? "Grouped-Query Attention" : "Multi-Head Attention";
  return ({
    rms_norm: "RMSNorm",
    fused_qkv_attention: "Fused-QKV Attention",
    mla_attention: "Multi-Head Latent Attention",
    dsa_mla_attention: "MLA + Sparse Attention",
    gated_deltanet: "Gated DeltaNet",
    kimi_delta_attention: "Kimi Delta Attention",
    moe_mlp: "Mixture-of-Experts",
    mlp: "FeedForward (SwiGLU)",
    fused_mlp: "Fused FeedForward",
    state_space: "State-Space (Mamba)",
    nextn_prediction: "Multi-Token Prediction",
    residual_add: "Residual add"
  })[o] ?? o ?? "";
}

function opClass(op) {
  if (ATTN.test(op.op)) return "op-attn";
  if (FFN.test(op.op)) return "op-ffn";
  if (op.op === "state_space") return "op-state";
  if (op.op === "nextn_prediction") return "op-mtp";
  if (op.op === "rms_norm") return "op-norm";
  if (op.op === "residual_add") return "op-res";
  return "op-other";
}

// ===================== mini primitives (op-sketch contract) =================
function mbox(cx, cy, w, label, opts = {}) {
  const sub = opts.sub;
  const h = sub ? 42 : 30, x = cx - w / 2, y = cy - h / 2;
  const cls = `mbox${opts.hot ? " mhot" : ""}${opts.dark ? " mdark" : ""}${opts.tensor ? " wbox" : ""}`;
  const data = opts.tensor ? `data-tensor="${esc(opts.tensor)}" tabindex="0"` : "";
  const tip = opts.tip ? `<title>${esc(opts.tip)}</title>` : "";
  const dot = opts.tensor ? `<circle class="wdot" cx="${x + w - 9}" cy="${y + 9}" r="2.7"/>` : "";
  return `<g class="${cls}" ${data}>${tip}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="7"/>${dot}` +
    `<text x="${cx}" y="${sub ? cy - 6 : cy}">${esc(label)}</text>` +
    (sub ? `<text class="m-sub" x="${cx}" y="${cy + 10}">${esc(sub)}</text>` : "") + `</g>`;
}
// when the "Sizes" toggle is off, inset boxes suppress their numeric shape sub-lines
let SHOW_SIZES = true;
// option helpers for boxes backed by a real weight tensor (clickable → inspector, hover → exact count)
function wopt(p, name, extra = {}) {
  const e = p && p[name];
  return { ...extra, sub: SHOW_SIZES ? (extra.sub ?? io(p, name)) : (extra.sub ?? ""), tensor: e ? name : undefined, tip: e && e.parameter_count ? `${name}  ·  ${full(e.parameter_count)} parameters` : undefined };
}
function sopt(p, name, extra = {}) {
  const e = p && p[name];
  const sh = shp(p, name);
  const shape = sh ? (sh.length >= 2 ? `${fmt(sh[0])} → ${fmt(sh[1])}` : `${fmt(sh[0])}`) : "";
  return { ...extra, sub: SHOW_SIZES ? (extra.sub ?? shape) : (extra.sub ?? ""), tensor: e ? name : undefined, tip: e && e.parameter_count ? `${name}  ·  ${full(e.parameter_count)} parameters` : undefined };
}
// tag an existing box (keeping its custom sub) as a clickable weight
function tagw(p, name, opts = {}) {
  const e = p && p[name];
  return { ...opts, tensor: e ? name : opts.tensor, tip: e && e.parameter_count ? `${name}  ·  ${full(e.parameter_count)} parameters` : opts.tip };
}
// GQA / MHA / MQA head-grouping visual: query heads bracketed onto the K/V head they share
function headGrid(cx, y, nh, nkv) {
  if (!nh) return { h: 0, svg: "" };
  const groups = nkv || nh, g = Math.max(1, Math.round(nh / groups));
  const mode = groups === nh ? "MHA" : groups === 1 ? "MQA" : "GQA";
  const G = Math.min(groups, 4), qcap = Math.min(g, 4);
  const gw = 92, totalW = G * gw, x0 = cx - totalW / 2;
  const qy = y + 24, brY = qy + 16, kvY = brY + 18, sq = 13, gap = 6;
  let s = `<text class="head-cap" x="${cx}" y="${y + 8}">${mode}: ${nh} query · ${groups} key/value heads${mode === "GQA" ? `  (${g} queries share 1 K/V)` : ""}</text>`;
  for (let gi = 0; gi < G; gi++) {
    const gx = x0 + gi * gw + gw / 2;
    const tw = qcap * sq + (qcap - 1) * gap, sx = gx - tw / 2;
    for (let i = 0; i < qcap; i++) s += `<rect class="head-q" x="${sx + i * (sq + gap)}" y="${qy}" width="${sq}" height="${sq}" rx="2"/>`;
    if (g > qcap) s += `<text class="head-more" x="${sx + tw + 5}" y="${qy + sq - 1}">…</text>`;
    s += `<path class="head-brace" d="M${sx} ${brY} v4 H${sx + tw} v-4"/>`;
    s += ml(gx, brY, gx, kvY - 1, false);
    s += `<rect class="head-kv" x="${gx - 24}" y="${kvY}" width="48" height="16" rx="3"/><text class="head-kvt" x="${gx}" y="${kvY + 9}">K,V</text>`;
  }
  if (groups > G) s += `<text class="head-more" x="${x0 + totalW + 8}" y="${kvY + 9}">+${groups - G}</text>`;
  return { h: kvY + 16 - y + 8, svg: s };
}
function mnode(cx, cy, sym) {
  return `<g class="mbox mnode"><circle cx="${cx}" cy="${cy}" r="14"/><text x="${cx}" y="${cy + 1}">${esc(sym)}</text></g>`;
}
function mcap(cx, cy, text) {
  return `<text class="m-cap" x="${cx}" y="${cy}">${esc(text)}</text>`;
}
function ml(x1, y1, x2, y2, head = true) {
  return `<line class="mflow${head ? "" : " nh"}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
}
function mpath(d, head = true) {
  return `<path class="mflow${head ? "" : " nh"}" d="${d}"/>`;
}
function recur(cx, cy, w, label) {
  const xr = cx + w / 2;
  return `<path class="mflow recur" d="M${xr} ${cy + 7} h20 v-30 h-20"/>` +
    (label ? `<text class="recur-label" x="${xr + 24}" y="${cy - 8}">${esc(label)}</text>` : "");
}
// top half (cy-21) for sub boxes, (cy-15) otherwise
const tH = (sub) => sub ? 21 : 15;
// stack a column of boxes top-to-bottom, connected by arrows. each item = [label, opts, width].
// opts may carry mbox keys (hot/sub/tensor/tip) plus gap/recur/recurLabel.
function linOps(cx, y0, list) {
  let pb = null, svg = "";
  for (const [label, opts = {}, ww = 180] of list) {
    const half = opts.sub ? 21 : 15;
    const cy = (pb === null ? y0 : pb + (opts.gap ?? 18)) + half;
    svg += mbox(cx, cy, ww, label, opts);
    if (pb !== null) svg += ml(cx, pb, cx, cy - half);
    if (opts.recur) svg += recur(cx, cy, ww, opts.recurLabel);
    pb = cy + half;
  }
  return { svg, bottom: pb };
}

// ===================== op sketches (literal components + shapes) ============
function sketch_self_attention(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, c = op.config || {}, rows = [];
  const nh = c.num_heads ?? dims.num_attention_heads, nkv = c.num_kv_heads ?? dims.num_kv_heads ?? nh, hd = c.head_dim ?? dims.head_dim;
  const qk = !!(p.attn_q_norm || p.attn_k_norm);
  const off = 122;
  const r1 = y + 18; rows.push(mbox(cx, r1, 160, "hidden state", { sub: dims.hidden_size ? `d_model = ${fmt(dims.hidden_size)}` : "" }));
  const r2 = r1 + 70;
  rows.push(mbox(cx - off, r2, 112, "Q proj", sopt(p, "attn_q", { hot: true })));
  rows.push(mbox(cx, r2, 112, "K proj", sopt(p, "attn_k")));
  rows.push(mbox(cx + off, r2, 112, "V proj", sopt(p, "attn_v")));
  rows.push(ml(cx, r1 + 21, cx - off, r2 - 21, false) + ml(cx, r1 + 21, cx, r2 - 21, false) + ml(cx, r1 + 21, cx + off, r2 - 21, false));
  let cur = r2;
  if (qk) {
    const rn = cur + 54;
    rows.push(mbox(cx - off, rn, 112, "Q RMSNorm", tagw(p, "attn_q_norm")) + mbox(cx, rn, 112, "K RMSNorm", tagw(p, "attn_k_norm")));
    rows.push(ml(cx - off, cur + 21, cx - off, rn - 15) + ml(cx, cur + 21, cx, rn - 15));
    cur = rn;
  }
  const rope = cur + (qk ? 54 : 60);
  rows.push(mbox(cx - off / 2, rope, 170, "rotary position embedding", { sub: "RoPE on Q, K" }));
  rows.push(ml(cx - off, cur + (qk ? 15 : 21), cx - off / 2, rope - 15, false) + ml(cx, cur + (qk ? 15 : 21), cx - off / 2, rope - 15, false));
  const hg = headGrid(cx, rope + 36, nh, nkv);
  rows.push(hg.svg);
  rows.push(ml(cx - off / 2, rope + 15, cx, rope + 30, false));
  const sc = rope + 36 + hg.h + 24;
  rows.push(mbox(cx, sc, 200, "attention scores  QKᵀ", { sub: "causal mask · match query to keys" }));
  rows.push(ml(cx, rope + 36 + hg.h - 6, cx, sc - 21, false));
  const scl = sc + 56; rows.push(mbox(cx, scl, 200, "scale by 1/√dₕ", { sub: hd ? `normalize · dₕ = ${fmt(hd)}` : "normalize by head size" })); rows.push(ml(cx, sc + 21, cx, scl - 21));
  const sm = scl + 54; rows.push(mbox(cx, sm, 150, "softmax (row-wise)")); rows.push(ml(cx, scl + 21, cx, sm - 15));
  const r5 = sm + 54; rows.push(mbox(cx, r5, 180, "context = Σ softmax · V"));
  rows.push(ml(cx, sm + 15, cx, r5 - 15));
  // V flows down the right margin straight into the context step (not the scores)
  rows.push(mpath(`M${cx + off} ${r2 + 21} V${r5} H${cx + 92}`));
  const r6 = r5 + 56; rows.push(mbox(cx, r6, 180, "output projection", wopt(p, "attn_output")));
  rows.push(ml(cx, r5 + 15, cx, r6 - 21));
  return { h: (r6 + 21) - y + 8, svg: rows.join("") };
}

function sketch_fused_qkv_attention(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, c = op.config || {}, rows = [];
  const nh = c.num_heads ?? dims.num_attention_heads, nkv = c.num_kv_heads ?? dims.num_kv_heads ?? nh, hd = c.head_dim ?? dims.head_dim;
  const r1 = y + 18; rows.push(mbox(cx, r1, 160, "hidden state"));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 210, "fused QKV projection", wopt(p, p.attn_qkv ? "attn_qkv" : "attn_q", { hot: true })));
  rows.push(ml(cx, r1 + 15, cx, r2 - 21));
  const r3 = r2 + 56;
  rows.push(mbox(cx - 84, r3, 70, "Q") + mbox(cx, r3, 70, "K") + mbox(cx + 84, r3, 70, "V"));
  rows.push(ml(cx, r2 + 21, cx - 84, r3 - 15, false) + ml(cx, r2 + 21, cx, r3 - 15, false) + ml(cx, r2 + 21, cx + 84, r3 - 15, false));
  const r4 = r3 + 54; rows.push(mbox(cx, r4, 170, "rotary position embedding", { sub: "RoPE" }));
  rows.push(ml(cx - 84, r3 + 15, cx, r4 - 15, false) + ml(cx, r3 + 15, cx, r4 - 15, false) + ml(cx + 84, r3 + 15, cx, r4 - 15, false));
  const hg = headGrid(cx, r4 + 36, nh, nkv); rows.push(hg.svg); rows.push(ml(cx, r4 + 15, cx, r4 + 30, false));
  const r5 = r4 + 36 + hg.h + 24;
  rows.push(mbox(cx, r5, 200, "attention scores  QKᵀ", { sub: "causal mask · match query to keys" }));
  rows.push(ml(cx, r4 + 36 + hg.h - 6, cx, r5 - 21, false));
  const scl = r5 + 56; rows.push(mbox(cx, scl, 200, "scale by 1/√dₕ", { sub: hd ? `normalize · dₕ = ${fmt(hd)}` : "normalize by head size" })); rows.push(ml(cx, r5 + 21, cx, scl - 21));
  const sm = scl + 54; rows.push(mbox(cx, sm, 150, "softmax (row-wise)")); rows.push(ml(cx, scl + 21, cx, sm - 15));
  const r6 = sm + 54; rows.push(mbox(cx, r6, 180, "context = Σ softmax · V"));
  rows.push(ml(cx, sm + 15, cx, r6 - 15));
  const r7 = r6 + 56; rows.push(mbox(cx, r7, 180, "output projection", wopt(p, "attn_output")));
  rows.push(ml(cx, r6 + 15, cx, r7 - 21));
  return { h: (r7 + 21) - y + 8, svg: rows.join("") };
}

function sketch_mla_attention(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, c = op.config || {}, rows = [];
  const qRank = c.q_lora_rank ?? dims.q_lora_rank;
  const kvRank = dims.kv_lora_rank ?? (shp(p, "attn_kv") && shp(p, "attn_kv")[1]);
  const heads = c.num_heads ?? dims.num_attention_heads;
  const hd = dims.head_dim;
  const sinks = !!p.attn_sinks;
  void rows;
  const out = p.attn_output_a
    ? [["output up-proj", wopt(p, "attn_output_a"), 180], ["output projection", wopt(p, "attn_output_b"), 180]]
    : [["output projection", wopt(p, p.attn_output_b ? "attn_output_b" : "attn_output"), 180]];
  const r = linOps(cx, y + 18, [
    ["hidden state", {}, 150],
    ["Q down-proj", tagw(p, "attn_q_a", { sub: qRank ? `compress query · rank ${fmt(qRank)}` : io(p, "attn_q_a") }), 230],
    ...(p.attn_q_a_norm ? [["Q latent RMSNorm", tagw(p, "attn_q_a_norm"), 170]] : []),
    ["Q up-proj", wopt(p, "attn_q_b", { sub: "expand to heads" }), 190],
    ["KV down-proj → latent", tagw(p, "attn_kv", { hot: true, sub: kvRank ? `compress key+value · ${fmt(kvRank)}` : io(p, "attn_kv") }), 240],
    ...(p.attn_kv_a_norm ? [["KV latent RMSNorm", tagw(p, "attn_kv_a_norm"), 170]] : []),
    ["split Q,K → content + position", { sub: "plain dims · rotary dims" }, 230],
    ["rotary position embedding", { sub: hd ? `RoPE · per-head dₕ = ${fmt(hd)}` : "RoPE" }, 220],
    ["rejoin content + position", {}, 210],
    ["attention scores  QKᵀ", { hot: true, sub: heads ? ` heads` : "match query to keys" }, 220],
    ["scale by 1/√dₕ", { sub: "normalize by head size" }, 220],
    ...(sinks ? [["+ attention sinks", tagw(p, "attn_sinks", { sub: "always-attended tokens" }), 220]] : []),
    ["softmax (row-wise)", {}, 160],
    ["context = Σ softmax · V", {}, 200],
    ...out
  ]);
  return { h: r.bottom - y + 8, svg: r.svg };
}

function sketch_dsa_mla_attention(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, c = op.config || {}, rows = [];
  const qRank = c.q_lora_rank ?? dims.q_lora_rank;
  const kvRank = dims.kv_lora_rank ?? (shp(p, "attn_kv") && shp(p, "attn_kv")[1]);
  const heads = c.num_heads ?? dims.num_attention_heads;
  void rows;
  const hd = dims.head_dim, sinks = !!p.attn_sinks;
  const out = p.attn_output_a
    ? [["output up-proj", wopt(p, "attn_output_a"), 180], ["output projection", wopt(p, "attn_output_b"), 180]]
    : [["output projection", wopt(p, p.attn_output_b ? "attn_output_b" : "attn_output"), 180]];
  const r = linOps(cx, y + 18, [
    ["hidden state", {}, 150],
    ["Q down-proj", tagw(p, "attn_q_a", { sub: qRank ? `compress query · rank ${fmt(qRank)}` : io(p, "attn_q_a") }), 230],
    ...(p.attn_q_a_norm ? [["Q latent RMSNorm", tagw(p, "attn_q_a_norm"), 170]] : []),
    ["Q up-proj", wopt(p, "attn_q_b", { sub: "expand to heads" }), 190],
    ["KV down-proj → latent", tagw(p, "attn_kv", { hot: true, sub: kvRank ? `compress key+value · ${fmt(kvRank)}` : io(p, "attn_kv") }), 240],
    ...(p.attn_kv_a_norm ? [["KV latent RMSNorm", tagw(p, "attn_kv_a_norm"), 170]] : []),
    ["relevance indexer (query)", tagw(p, "indexer.attn_q_b", { hot: true, sub: "cheap relevance estimate" }), 210],
    ...(p.attn_compressor_ape ? [["attention key compressor", tagw(p, "attn_compressor_ape", { sub: "compress keys" }), 210]] : []),
    ...(p.indexer_compressor_ape ? [["indexer key compressor", tagw(p, "indexer_compressor_ape", { sub: "compress keys for index" }), 240]] : []),
    ["index scores", { sub: "estimate which keys matter" }, 230],
    ["keep top-k keys", { sub: "attend to most relevant only" }, 230],
    ["split Q,K → content + position", { sub: "plain dims · rotary dims" }, 230],
    ["rotary position embedding", { sub: hd ? `RoPE · per-head dₕ = ${fmt(hd)}` : "RoPE" }, 220],
    ["rejoin content + position", {}, 210],
    ["attention scores  QKᵀ", { hot: true, sub: heads ? `top-k keys ·  heads` : "selected keys only" }, 230],
    ["scale by 1/√dₕ", { sub: "normalize by head size" }, 220],
    ...(sinks ? [["+ attention sinks", tagw(p, "attn_sinks", { sub: "always-attended tokens" }), 220]] : []),
    ["softmax (row-wise)", {}, 160],
    ["context = Σ softmax · V", {}, 200],
    ...out
  ]);
  return { h: r.bottom - y + 8, svg: r.svg };
}

function sketch_gated_deltanet(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const k = (shp(p, "ssm_conv1d") && shp(p, "ssm_conv1d")[0]) || 4;
  const r = linOps(cx, y + 18, [
    ["hidden state", {}, 150],
    ["input projection", wopt(p, "ssm_in", { hot: true, sub: "→ query, key, value, β, a" }), 250],
    ["short conv1d", tagw(p, "ssm_conv1d", { sub: `mix ${k} neighboring tokens` }), 220],
    ["write-gate projection", tagw(p, "ssm_ba", { sub: "β = how much to write · a" }), 230],
    ["per-token step  Δ", tagw(p, "ssm_dt", { sub: "write rate per token" }), 200],
    ["decay rate  A", tagw(p, "ssm_a", { hot: true, sub: "how fast memory fades" }), 200],
    ["discretize", { sub: "αₜ = exp(−Δ·A)" }, 200],
    ["update memory (recurrent)", { hot: true, sub: "Sₜ = αₜ Sₜ₋₁ + βₜ kₜ vₜᵀ", recur: true, recurLabel: "memory Sₜ" }, 250],
    ["read memory with query", { sub: "oₜ = Sₜ qₜ" }, 200],
    ["output gate", { sub: "elementwise ⊙" }, 170],
    ["RMSNorm", tagw(p, "ssm_norm"), 150],
    ["output projection", wopt(p, "ssm_out"), 170]
  ]);
  return { h: r.bottom - y + 8, svg: r.svg };
}

function sketch_kimi_delta_attention(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const k = (shp(p, "ssm_conv1d_q") && shp(p, "ssm_conv1d_q")[0]) || (shp(p, "ssm_conv1d_k") && shp(p, "ssm_conv1d_k")[0]) || 4;
  const r = linOps(cx, y + 18, [
    ["hidden state", {}, 150],
    ["Q projection", sopt(p, "attn_q", { hot: true }), 170],
    ["K projection", sopt(p, "attn_k"), 170],
    ["V projection", sopt(p, "attn_v"), 170],
    ["short conv1d · query", tagw(p, "ssm_conv1d_q", { sub: `mix ${k} neighboring tokens` }), 230],
    ["short conv1d · key", tagw(p, "ssm_conv1d_k", { sub: `mix ${k} neighboring tokens` }), 230],
    ["short conv1d · value", tagw(p, "ssm_conv1d_v", { sub: `mix ${k} neighboring tokens` }), 230],
    ["write strength  β", tagw(p, "ssm_beta", { sub: "how much to write" }), 190],
    ["forget gate (low-rank ↓)", tagw(p, "ssm_f_a", { sub: "decides what to forget" }), 220],
    ["forget gate (low-rank ↑)", tagw(p, "ssm_f_b", { sub: "→ how much memory to keep" }), 230],
    ["output gate (low-rank ↓)", tagw(p, "ssm_g_a", { sub: "scales the read-out" }), 220],
    ["output gate (low-rank ↑)", tagw(p, "ssm_g_b", { sub: "→ final output scale" }), 220],
    ["per-token step  Δ", tagw(p, "ssm_dt", { sub: "write rate per token" }), 200],
    ["decay rate  A", tagw(p, "ssm_a", { hot: true, sub: "how fast memory fades" }), 200],
    ["discretize", { sub: "αₜ = exp(−Δ·A)" }, 200],
    ["update memory (recurrent)", { hot: true, sub: "Sₜ = Diag(αₜ) Sₜ₋₁ + βₜ kₜ vₜᵀ", recur: true, recurLabel: "memory Sₜ" }, 250],
    ["read memory with query", { sub: "oₜ = Sₜ qₜ" }, 200],
    ["RMSNorm", tagw(p, "ssm_norm"), 150],
    ["output projection", wopt(p, "attn_output"), 170]
  ]);
  return { h: r.bottom - y + 8, svg: r.svg };
}

function sketch_state_space(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const k = (shp(p, "ssm_conv1d") && shp(p, "ssm_conv1d")[0]) || 4;
  const r = linOps(cx, y + 18, [
    ["hidden state", {}, 150],
    ["input projection", wopt(p, "ssm_in", { hot: true, sub: "→ x, B, C, Δ" }), 200],
    ["short conv1d", tagw(p, "ssm_conv1d", { sub: `mix ${k} neighboring tokens` }), 220],
    ["per-token step  Δ", tagw(p, "ssm_dt", { sub: "write rate per token" }), 200],
    ["decay rate  A", tagw(p, "ssm_a", { hot: true, sub: "how fast memory fades" }), 200],
    ["discretize", { sub: "Ā=exp(Δ·A), B̄=Δ·B" }, 210],
    ["update memory (recurrent)", { hot: true, sub: "hₜ = Āₜ hₜ₋₁ + B̄ₜ xₜ", recur: true, recurLabel: "memory hₜ" }, 250],
    ["read memory", { sub: "yₜ = Cₜ hₜ" }, 180],
    ["skip path  D", tagw(p, "ssm_d", { sub: "+ D ⊙ xₜ" }), 190],
    ["output gate", { sub: "elementwise ⊙" }, 160],
    ["RMSNorm", tagw(p, "ssm_norm"), 150],
    ["output projection", wopt(p, "ssm_out"), 170]
  ]);
  return { h: r.bottom - y + 8, svg: r.svg };
}

function sketch_mlp(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const r1 = y + 18; rows.push(mbox(cx, r1, 150, "hidden state"));
  const r2 = r1 + 64;
  rows.push(mbox(cx - 84, r2, 132, "gate", wopt(p, "ffn_gate", { hot: true })));
  rows.push(mbox(cx + 84, r2, 132, "up", wopt(p, "ffn_up", { hot: true })));
  rows.push(ml(cx, r1 + 15, cx - 84, r2 - 21, false) + ml(cx, r1 + 15, cx + 84, r2 - 21, false));
  const r3 = r2 + 56; rows.push(mbox(cx - 84, r3, 132, "SiLU")); rows.push(ml(cx - 84, r2 + 21, cx - 84, r3 - 15));
  const r4 = r3 + 50; rows.push(mnode(cx, r4, "×"));
  rows.push(ml(cx - 84, r3 + 15, cx - 12, r4 - 6, false) + ml(cx + 84, r2 + 21, cx + 12, r4 - 6, false));
  const r5 = r4 + 52; rows.push(mbox(cx, r5, 160, "down", wopt(p, "ffn_down"))); rows.push(ml(cx, r4 + 14, cx, r5 - 21));
  return { h: (r5 + 21) - y + 8, svg: rows.join("") };
}

function sketch_fused_mlp(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const r1 = y + 18; rows.push(mbox(cx, r1, 150, "hidden state"));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 190, "fused gate-up", wopt(p, p.ffn_gate_up ? "ffn_gate_up" : "ffn_up", { hot: true }))); rows.push(ml(cx, r1 + 15, cx, r2 - 21));
  const r3 = r2 + 56; rows.push(mbox(cx - 72, r3, 96, "gate") + mbox(cx + 72, r3, 96, "up"));
  rows.push(ml(cx, r2 + 21, cx - 72, r3 - 15, false) + ml(cx, r2 + 21, cx + 72, r3 - 15, false));
  const r4 = r3 + 50; rows.push(mbox(cx - 72, r4, 96, "SiLU")); rows.push(ml(cx - 72, r3 + 15, cx - 72, r4 - 15));
  const r5 = r4 + 50; rows.push(mnode(cx, r5, "×")); rows.push(ml(cx - 72, r4 + 15, cx - 12, r5 - 6, false) + ml(cx + 72, r3 + 15, cx + 12, r5 - 6, false));
  const r6 = r5 + 52; rows.push(mbox(cx, r6, 160, "down", wopt(p, "ffn_down"))); rows.push(ml(cx, r5 + 14, cx, r6 - 21));
  return { h: (r6 + 21) - y + 8, svg: rows.join("") };
}

function sketch_moe_mlp(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, c = op.config || {}, rows = [];
  const N = c.expert_count ?? dims.expert_count, k = c.experts_per_token ?? dims.experts_per_token;
  const shared = !!p.ffn_gate_shexp;
  const r1 = y + 18; rows.push(mbox(cx, r1, 160, "hidden state"));
  const r2 = r1 + 60; rows.push(mbox(cx, r2, 210, "router / gate", { hot: true, sub: k && N ? `top-${k} of ${N}` : "", tensor: p.ffn_gate_inp ? "ffn_gate_inp" : undefined, tip: p.ffn_gate_inp && p.ffn_gate_inp.parameter_count ? `ffn_gate_inp  ·  ${full(p.ffn_gate_inp.parameter_count)} parameters` : undefined }));
  rows.push(ml(cx, r1 + 15, cx, r2 - 21));
  const ecx = shared ? cx + 64 : cx;  // routed-expert column (shifted right when a shared expert sits on the left)
  const g = r2 + 88, sR = g + 52, m = sR + 48, dn = m + 50;
  const fT = g - 34, fB = dn + 33, fL = ecx - 100, fW = 200, fH = fB - fT;
  // routed SwiGLU expert as a stacked deck of N (cards peek to the left, away from the shared column)
  rows.push(`<rect class="exp-card" x="${fL - 12}" y="${fT + 12}" width="${fW}" height="${fH}" rx="12"/>`);
  rows.push(`<rect class="exp-card" x="${fL - 6}" y="${fT + 6}" width="${fW}" height="${fH}" rx="12"/>`);
  rows.push(`<rect class="exp-frame" x="${fL}" y="${fT}" width="${fW}" height="${fH}" rx="12"/>`);
  rows.push(`<rect class="exp-badge" x="${fL + fW - 54}" y="${fT - 11}" width="64" height="22" rx="11"/><text class="exp-badge-t" x="${fL + fW - 22}" y="${fT}">× ${N ? fmt(N) : "N"}</text>`);
  rows.push(ml(cx, r2 + 21, ecx, fT - 2, false));
  rows.push(mbox(ecx - 50, g, 92, "gate", sopt(p, "ffn_gate_exps", { hot: true })));
  rows.push(mbox(ecx + 50, g, 92, "up", sopt(p, "ffn_up_exps", { hot: true })));
  rows.push(mbox(ecx - 50, sR, 92, "SiLU")); rows.push(ml(ecx - 50, g + 15, ecx - 50, sR - 15));
  rows.push(mnode(ecx, m, "×")); rows.push(ml(ecx - 50, sR + 15, ecx - 12, m - 6, false) + ml(ecx + 50, g + 15, ecx + 12, m - 6, false));
  rows.push(mbox(ecx, dn, 120, "down", sopt(p, "ffn_down_exps"))); rows.push(ml(ecx, m + 14, ecx, dn - 15));
  const cb = fB + 40; rows.push(mnode(cx, cb, "+"));
  rows.push(ml(ecx, dn + 15, cx + 11, cb - 7, false));
  if (shared) {
    const scx = cx - 122;
    rows.push(`<text class="m-cap" x="${scx}" y="${g - 28}">shared expert</text>`);
    rows.push(mbox(scx, g, 96, "gate", sopt(p, "ffn_gate_shexp", { hot: true })));
    rows.push(mbox(scx, sR, 96, "up", sopt(p, "ffn_up_shexp", { hot: true }))); rows.push(ml(scx, g + 15, scx, sR - 15));
    rows.push(mbox(scx, dn, 96, "down", sopt(p, "ffn_down_shexp"))); rows.push(ml(scx, sR + 15, scx, dn - 15));
    rows.push(ml(cx, r2 + 21, scx, g - 15, false));
    rows.push(ml(scx, dn + 15, cx - 11, cb - 7, false));
  }
  const o = cb + 46; rows.push(mbox(cx, o, 210, "weighted Σ of top-k experts")); rows.push(ml(cx, cb + 14, cx, o - 15));
  return { h: (o + 15) - y + 8, svg: rows.join("") };
}

function sketch_nextn_prediction(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const r1 = y + 18;
  rows.push(mbox(cx - 100, r1, 132, "hidden state hₜ") + mbox(cx + 100, r1, 132, "token t+1 embed"));
  const r2 = r1 + 58;
  rows.push(mbox(cx - 100, r2, 132, "RMSNorm (hnorm)", tagw(p, "nextn.hnorm")) + mbox(cx + 100, r2, 132, "RMSNorm (enorm)", tagw(p, "nextn.enorm")));
  rows.push(ml(cx - 100, r1 + 15, cx - 100, r2 - 15) + ml(cx + 100, r1 + 15, cx + 100, r2 - 15));
  const r3 = r2 + 56; rows.push(mbox(cx, r3, 180, "concat  [hₜ ⊕ eₜ]"));
  rows.push(ml(cx - 100, r2 + 15, cx, r3 - 15, false) + ml(cx + 100, r2 + 15, cx, r3 - 15, false));
  const r4 = r3 + 52; rows.push(mbox(cx, r4, 230, "combine projection", tagw(p, "nextn.eh_proj", { hot: true, sub: "[hidden ⊕ embed] → model dim" }))); rows.push(ml(cx, r3 + 15, cx, r4 - 21));
  const r5 = r4 + 56; rows.push(mbox(cx, r5, 190, "shared transformer block")); rows.push(ml(cx, r4 + 21, cx, r5 - 15));
  const r6 = r5 + 52; rows.push(mbox(cx, r6, 160, "final RMSNorm", tagw(p, "nextn.shared_head_norm"))); rows.push(ml(cx, r5 + 15, cx, r6 - 15));
  const r7 = r6 + 52; rows.push(mbox(cx, r7, 170, "predict token t+1")); rows.push(ml(cx, r6 + 15, cx, r7 - 15));
  return { h: (r7 + 15) - y + 8, svg: rows.join("") };
}

function sketch_rms_norm(op, dims, x, y, w) {
  const cx = x + w / 2, p = op.parameters || {}, rows = [];
  const wname = Object.keys(p).find((k) => shp(p, k));
  const d = wname ? shp(p, wname)[0] : dims.hidden_size;
  const r1 = y + 18; rows.push(mbox(cx, r1, 150, "input x", { sub: d ? `dim ${fmt(d)}` : "" }));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 210, "RMS = √(mean(xᵢ²) + ε)")); rows.push(ml(cx, r1 + 21, cx, r2 - 15));
  const r3 = r2 + 56; rows.push(mbox(cx, r3, 160, "x̂ = x / RMS")); rows.push(ml(cx, r2 + 15, cx, r3 - 15));
  const r4 = r3 + 54; rows.push(mnode(cx, r4, "×")); rows.push(ml(cx, r3 + 15, cx, r4 - 8));
  rows.push(mbox(cx + 116, r4, 110, "learned scale γ", { hot: true, tensor: wname, tip: wname && p[wname]?.parameter_count ? `${wname}  ·  ${full(p[wname].parameter_count)} parameters` : undefined })); rows.push(ml(cx + 61, r4, cx + 14, r4));
  const r5 = r4 + 50; rows.push(mbox(cx, r5, 170, "normalized output")); rows.push(ml(cx, r4 + 14, cx, r5 - 15));
  return { h: (r5 + 15) - y + 8, svg: rows.join("") };
}

function sketch_residual_add(op, dims, x, y, w) {
  const cx = x + w / 2, rows = [];
  const r1 = y + 18;
  rows.push(mbox(cx - 96, r1, 132, "block input x", { sub: "pre-norm shortcut" }));
  rows.push(mbox(cx + 96, r1, 132, "sub-layer output", { sub: "f(norm(x))" }));
  const r2 = r1 + 70; rows.push(mnode(cx, r2, "+"));
  rows.push(ml(cx - 96, r1 + 21, cx - 12, r2 - 6, false) + ml(cx + 96, r1 + 21, cx + 12, r2 - 6, false));
  const r3 = r2 + 50; rows.push(mbox(cx, r3, 170, "x + f(norm(x))")); rows.push(ml(cx, r2 + 14, cx, r3 - 15));
  return { h: (r3 + 15) - y + 8, svg: rows.join("") };
}

const SKETCHES = {
  self_attention: sketch_self_attention,
  fused_qkv_attention: sketch_fused_qkv_attention,
  mla_attention: sketch_mla_attention,
  dsa_mla_attention: sketch_dsa_mla_attention,
  gated_deltanet: sketch_gated_deltanet,
  kimi_delta_attention: sketch_kimi_delta_attention,
  moe_mlp: sketch_moe_mlp,
  mlp: sketch_mlp,
  fused_mlp: sketch_fused_mlp,
  state_space: sketch_state_space,
  nextn_prediction: sketch_nextn_prediction,
  rms_norm: sketch_rms_norm,
  residual_add: sketch_residual_add
};

function runSketch(op, dims, x, y, w) {
  const fn = SKETCHES[op.op];
  if (!fn) { const cx = x + w / 2; return { h: 90, svg: mbox(cx, y + 24, 170, opLabel(op), { hot: true }) }; }
  return fn({ ...op, parameters: paramMap(op) }, dims, x, y, w);
}

// ----- sketches for the nodes outside the transformer block ----------------
function sketchTokenize(op, dims, x, y, w) {
  const cx = x + w / 2, rows = [];
  const r1 = y + 18; rows.push(mbox(cx, r1, 170, "input text", { sub: "“the cat sat …”" }));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 200, "tokenizer (BPE merges)", { hot: true, sub: dims.vocab_size ? `vocabulary ${fmt(dims.vocab_size)}` : "" }));
  rows.push(ml(cx, r1 + 21, cx, r2 - 21));
  const r3 = r2 + 64; rows.push(mbox(cx, r3, 180, "token id sequence", { sub: dims.context_length ? `≤ ${fmt(dims.context_length)} tokens` : "" }));
  rows.push(ml(cx, r2 + 21, cx, r3 - 21));
  return { h: (r3 + 21) - y + 8, svg: rows.join("") };
}

function sketchEmbedding(op, dims, x, y, w) {
  const cx = x + w / 2, rows = [], p = op ? paramMap(op) : {};
  let s = null, pc = 0, kname = ""; for (const k in p) if (shp(p, k)) { s = shp(p, k); pc = p[k].parameter_count || 0; kname = k; break; }
  const tbl = s ? (s.length >= 2 ? `${fmt(Math.max(...s))} × ${fmt(Math.min(...s))}` : fmt(s[0])) : "";
  const r1 = y + 18; rows.push(mbox(cx, r1, 170, "token id", { sub: dims.vocab_size ? `0 … ${fmt(dims.vocab_size - 1)}` : "" }));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 210, "embedding lookup table", { hot: true, sub: tbl ? `vocab × d = ${tbl}` : "", tensor: kname || undefined, tip: pc ? `${kname}  ·  ${full(pc)} parameters` : undefined }));
  rows.push(ml(cx, r1 + 21, cx, r2 - 21));
  const r3 = r2 + 64; rows.push(mbox(cx, r3, 180, "hidden state h₀", { sub: dims.hidden_size ? `d_model = ${fmt(dims.hidden_size)}` : "" }));
  rows.push(ml(cx, r2 + 21, cx, r3 - 21));
  return { h: (r3 + 21) - y + 8, svg: rows.join("") };
}

function sketchHead(op, dims, x, y, w) {
  const cx = x + w / 2, rows = [], p = op ? paramMap(op) : {};
  let s = null, pc = 0, kname = ""; for (const k in p) if (shp(p, k)) { s = shp(p, k); pc = p[k].parameter_count || 0; kname = k; break; }
  const r1 = y + 18; rows.push(mbox(cx, r1, 180, "final hidden state", { sub: dims.hidden_size ? `d_model = ${fmt(dims.hidden_size)}` : "" }));
  const r2 = r1 + 64; rows.push(mbox(cx, r2, 210, "output projection", { hot: true, sub: s ? `${fmt(s[0])} → ${fmt(s[1])}` : "tied to embedding", tensor: kname || undefined, tip: pc ? `${kname}  ·  ${full(pc)} parameters` : undefined }));
  rows.push(ml(cx, r1 + 21, cx, r2 - 21));
  const r3 = r2 + 62; rows.push(mbox(cx, r3, 180, "logits", { sub: dims.vocab_size ? `${fmt(dims.vocab_size)} vocab` : "" }));
  rows.push(ml(cx, r2 + 21, cx, r3 - 21));
  const r4 = r3 + 58; rows.push(mbox(cx, r4, 180, "softmax → next token"));
  rows.push(ml(cx, r3 + 21, cx, r4 - 15));
  return { h: (r4 + 15) - y + 8, svg: rows.join("") };
}

// ===================== main diagram (Raschka-style) =========================
const CX = 506;
const NODE_W = 246, NODE_X = CX - NODE_W / 2;
const BX = 358, BW = 296;
const SKIP_X = 638;
const RX = 742, RW = 448;
const STEP = 74;

export function renderArchitectureSvg(ir, options = {}) {
  const ranges = layouts(ir);
  const rangeIndex = options.rangeIndex ?? 0;
  const range = options.range ?? ranges[rangeIndex] ?? ranges[0];
  const ops = range?.layout ?? [];
  const model = ir.model ?? {};
  const d = ir.dimensions ?? {};
  const accent = options.accent ?? palette(model.architecture);
  const showSizes = options.showSizes ?? true;
  const showInternals = options.showInternals ?? true;
  const interactive = options.interactive ?? true;
  const selectedOpIndex = Math.min(options.selectedOpIndex ?? pickInterestingOp(ops), Math.max(ops.length - 1, 0));
  const selectedOuter = options.selectedOuter;
  const title = options.title ?? model.hf_name ?? model.base_model ?? model.name ?? "Model";
  const gqa = d.num_kv_heads && d.num_attention_heads && d.num_kv_heads < d.num_attention_heads;

  const blockTop = 256;
  const firstCenter = blockTop + 84;
  let normCount = 0;
  const norms = ops.filter((o) => o.op === "rms_norm").length;
  const nodes = ops.map((op, i) => {
    const cy = firstCenter + i * STEP;
    const isRes = op.op === "residual_add";
    const h = isRes ? 34 : (op.op === "rms_norm" ? 42 : 54);
    let label = opLabel(op, { gqa });
    if (op.op === "rms_norm" && norms > 1) label = `RMSNorm ${++normCount}`;
    return { op, i, cy, x: NODE_X, y: cy - h / 2, w: NODE_W, h, isRes, label };
  });
  const lastCy = nodes.length ? nodes.at(-1).cy : firstCenter;
  const blockBottom = lastCy + 58;
  const blockH = blockBottom - blockTop;

  const tokenY = 116, embedY = 174;
  const normFinalY = blockBottom + 54, headY = normFinalY + 60;

  const entryY = nodes.length ? nodes[0].y : firstCenter;
  const spine = [];
  spine.push(flow(CX, tokenY + 18, embedY - 22));
  spine.push(flow(CX, embedY + 22, blockTop - 2));
  spine.push(flow(CX, blockTop + 2, entryY));
  for (let i = 0; i < nodes.length - 1; i++) spine.push(flow(CX, nodes[i].y + nodes[i].h, nodes[i + 1].y));
  if (nodes.length) spine.push(flow(CX, nodes.at(-1).y + nodes.at(-1).h, blockBottom + 2));
  spine.push(flow(CX, blockBottom + 2, normFinalY - 18));
  spine.push(flow(CX, normFinalY + 18, headY - 18));

  let mergeY = blockTop + 8;
  const skips = [];
  for (const node of nodes) {
    if (!node.isRes) continue;
    skips.push(`<path class="skip" d="M${CX} ${mergeY} H${SKIP_X} V${node.cy} H${CX + 16}"/>`);
    mergeY = node.cy;
  }

  const ropeNode = nodes.find((n) => /attention/.test(n.op.op) && (n.op.config?.rope || /mla/.test(n.op.op)));
  let rope = "";
  if (ropeNode) {
    const ry = ropeNode.cy;
    rope = `<g class="rope"><rect x="206" y="${ry - 16}" width="76" height="32" rx="8"/><text x="244" y="${ry}">RoPE</text><line class="flow" x1="282" y1="${ry}" x2="${NODE_X}" y2="${ry}"/></g>`;
  }

  // every node — inside or outside the block — is inspectable
  const outers = {
    input: { title: "Tokenization", cy: tokenY, w: 190, kind: "tokenize", op: null },
    embedding: { title: "Token embedding", cy: embedY, w: 232, kind: "embedding", op: ir.embeddings?.[0] ?? null },
    final_norm: { title: "Final RMSNorm", cy: normFinalY, w: 190, kind: "rms_norm", op: (ir.output ?? []).find((o) => o.op === "rms_norm") ?? null },
    head: { title: "Linear output head", cy: headY, w: 208, kind: "head", op: (ir.output ?? []).find((o) => o.op !== "rms_norm") ?? null }
  };
  const outerSel = selectedOuter && outers[selectedOuter] ? selectedOuter : null;
  const opSelIdx = outerSel ? -1 : selectedOpIndex;
  const ptotal = (key) => outers[key].op ? paramTotal(outers[key].op) : 0;
  const embedSub = showSizes && ptotal("embedding") ? `${fmt(ptotal("embedding"))} params` : "";
  const normSub = showSizes && ptotal("final_norm") ? `${fmt(ptotal("final_norm"))} params` : "";
  const headSub = showSizes ? (ptotal("head") ? `${fmt(ptotal("head"))} params` : "tied to embedding") : "";

  let insetSvg = "", insetBottom = 0, leaders = "", hint = "";
  if (showInternals) {
    let selCy, selRight, kind, sop, title2, total;
    if (outerSel) {
      const o = outers[outerSel];
      selCy = o.cy; selRight = CX + o.w / 2; kind = o.kind; sop = o.op; title2 = o.title; total = o.op ? paramTotal(o.op) : 0;
    } else if (ops.length) {
      const sel = nodes[opSelIdx] ?? nodes[0];
      selCy = sel.cy; selRight = sel.x + sel.w; kind = sel.op.op; sop = sel.op; title2 = opLabel(sel.op, { gqa }); total = paramTotal(sel.op);
    }
    if (kind) {
      const r = renderInset(kind, sop, title2, d, RX, blockTop, RW, total, showSizes);
      insetSvg = r.svg;
      leaders = leader(selRight, selCy, RX, blockTop + 28);
      insetBottom = blockTop + r.h;
      hint = `<text class="hint" x="${RX}" y="${blockTop - 14}">▸ internals of the selected layer — click any ● marked box to inspect its weight</text>`;
    }
  }

  const calloutSvg = showSizes ? buildCallouts(ir, nodes, { blockTop, embedY, headY }).map(callout).join("") : "";
  const perLayer = showSizes && range?.per_block_parameter_count ? `${fmt(range.per_block_parameter_count)} / layer` : "";
  const height = Math.max(headY + 64, insetBottom + 20, 720);
  const titleHalf = title.length * 6.8 + 24;        // ~25px/800 title centered at CX
  const width = Math.round(Math.max(showInternals ? RX + RW + 26 : 706, CX + titleHalf));

  return `<svg class="arch-svg" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)} architecture diagram">
    ${style(accent)}
    <defs>
      <marker id="ah" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"/></marker>
      <marker id="ahm" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="6.4" markerHeight="6.4" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#475569"/></marker>
      <marker id="ahs" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#94a3b8"/></marker>
    </defs>
    <rect class="page" x="0" y="0" width="${width}" height="${height}"/>
    <text class="h-title" x="${CX}" y="44">${esc(title)}</text>
    <text class="h-sub" x="${CX}" y="70">${esc(model.architecture)} · ${fmt(model.parameter_count)} parameters · ${fmt(ir.blocks?.count)} layers<title>${full(model.parameter_count)} parameters total</title></text>

    ${ioBox(CX, tokenY, 190, "Tokenized input", "token ids", { outer: "input", selected: outerSel === "input", interactive })}
    ${ioBox(CX, embedY, 232, "Token embedding", embedSub, { outer: "embedding", selected: outerSel === "embedding", interactive, params: ptotal("embedding") })}

    <rect class="blockcard" x="${BX}" y="${blockTop}" width="${BW}" height="${blockH}" rx="22"/>
    <text class="block-label" x="${BX + 18}" y="${blockTop + 24}">Transformer block</text>
    ${repeatBrace(blockTop, blockBottom, range, perLayer)}

    ${spine.join("")}
    ${skips.join("")}
    ${rope}
    ${nodes.map((n) => nodeSvg(n, rangeIndex, opSelIdx, interactive, showSizes)).join("")}

    ${ioBox(CX, normFinalY, 190, "Final RMSNorm", normSub, { outer: "final_norm", selected: outerSel === "final_norm", interactive, params: ptotal("final_norm") })}
    ${ioBox(CX, headY, 208, "Linear output head", headSub, { outer: "head", selected: outerSel === "head", interactive, params: ptotal("head") })}

    ${calloutSvg}
    ${hint}
    ${leaders}
    ${insetSvg}
  </svg>`;
}

function flow(x, y1, y2) {
  return `<line class="flow" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}"/>`;
}

function ioBox(cx, cy, w, label, sub, opts = {}) {
  const h = sub ? 44 : 36, x = cx - w / 2, y = cy - h / 2;
  const data = opts.outer && opts.interactive ? `data-outer="${opts.outer}" tabindex="0"` : "";
  const cls = `iobox${opts.outer && opts.interactive ? " clickable" : ""}${opts.selected ? " selected" : ""}`;
  const tip = opts.params ? `<title>${esc(label)} — ${full(opts.params)} parameters</title>` : "";
  return `<g class="${cls}" ${data}>${tip}<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/>` +
    `<text x="${cx}" y="${sub ? cy - 6 : cy}">${esc(label)}</text>` +
    (sub ? `<text class="io-sub" x="${cx}" y="${cy + 11}">${esc(sub)}</text>` : "") + `</g>`;
}

function nodeSvg(node, rangeIndex, selectedOpIndex, interactive, showSizes) {
  const { op, i, x, y, w, h, cy, isRes, label } = node;
  const data = interactive ? `data-range="${rangeIndex}" data-op="${i}" tabindex="0"` : "";
  const sel = i === selectedOpIndex ? " selected" : "";
  if (isRes) return `<g class="opnode residual${sel}" ${data}><circle cx="${CX}" cy="${cy}" r="15"/><text x="${CX}" y="${cy + 1}">+</text></g>`;
  const pt = paramTotal(op);
  const sub = showSizes && pt ? `${fmt(pt)} params` : "";
  const tip = pt ? `<title>${esc(label)} — ${full(pt)} parameters</title>` : "";
  return `<g class="opnode ${opClass(op)}${sel}" ${data}>${tip}
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="11"/>
    <text x="${CX}" y="${sub ? cy - 7 : cy}">${esc(label)}</text>
    ${sub ? `<text class="op-sub" x="${CX}" y="${cy + 11}">${esc(sub)}</text>` : ""}
  </g>`;
}

function repeatBrace(top, bottom, range, perLayer) {
  const x = BX - 10, my = (top + bottom) / 2;
  const n = range ? range.range[1] - range.range[0] + 1 : 1;
  return `<g class="brace">
    <path d="M${x} ${top + 12} q-9 0 -9 11 V${my - 14} q0 9 -9 11 q9 2 9 11 V${bottom - 23} q0 11 9 11"/>
    <text class="brace-x" x="${x - 18}" y="${my}">${n}×</text>
    ${perLayer ? `<text class="brace-sub" x="${x - 18}" y="${my + 16}">${esc(perLayer)}</text>` : ""}
  </g>`;
}

function renderInset(kind, op, title, dims, x, y, w, total, showSizes) {
  const cx = x + 22, cy = y + 46, cw = w - 44;
  SHOW_SIZES = showSizes;
  let r;
  if (kind === "tokenize") r = sketchTokenize(op, dims, cx, cy, cw);
  else if (kind === "embedding") r = sketchEmbedding(op, dims, cx, cy, cw);
  else if (kind === "head") r = sketchHead(op, dims, cx, cy, cw);
  else r = runSketch(op, dims, cx, cy, cw);
  const h = 46 + r.h + 14;
  const sub = showSizes && total ? `${fmt(total)} parameters` : "";
  return {
    h,
    svg: `<g class="inset">${total ? `<title>${esc(title)} — ${full(total)} parameters</title>` : ""}<rect class="inset-bd" x="${x}" y="${y}" width="${w}" height="${h}" rx="14"/>` +
      `<text class="inset-title" x="${x + 22}" y="${y + 28}">${esc(title)}</text>` +
      (sub ? `<text class="inset-sub" x="${w + x - 22}" y="${y + 28}">${esc(sub)}</text>` : "") +
      r.svg + `</g>`
  };
}

function leader(x1, y1, x2, y2) {
  const mx = (x1 + x2) / 2;
  return `<g class="callout"><path class="lead" d="M${x1} ${y1} C${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}"/><circle class="lead-dot" cx="${x1}" cy="${y1}" r="2.6"/></g>`;
}

function buildCallouts(ir, nodes, geo) {
  const d = ir.dimensions ?? {};
  const out = [];
  const push = (lines, value, tx, ty, bx, by) => value != null && value !== "" && out.push({ lines, value, tx, ty, bx, by });
  push(["Embedding dim"], fmt(d.hidden_size), CX - 115, geo.embedY, 120, geo.embedY);
  push(["Context length"], `${fmt(d.context_length)} tok`, BX, geo.blockTop + 56, 120, geo.blockTop + 52);
  const attn = nodes.find((n) => /attention/.test(n.op.op));
  if (attn && d.num_attention_heads) {
    const kv = d.num_kv_heads;
    push([kv && kv < d.num_attention_heads ? "Q / KV heads" : "Attention heads"], kv && kv < d.num_attention_heads ? `${d.num_attention_heads} / ${kv}` : `${d.num_attention_heads}`, NODE_X, attn.cy + 8, 120, attn.cy + 36);
  }
  push(["Vocabulary"], fmt(d.vocab_size), CX - 103, geo.headY, CX - 150, geo.headY + 6);
  return out;
}

function callout(c) {
  const { lines, value, tx, ty, bx, by } = c;
  const rw = Math.max(56, value.length * 7 + 20), rx = rw / 2;
  const top = by - 21 - (lines.length - 1) * 13;
  const txt = lines.map((l, i) => `<text class="call-label" x="${bx}" y="${top + i * 13}">${esc(l)}</text>`).join("");
  const dir = tx > bx ? rx : -rx;
  return `<g class="callout">${txt}<rect class="call-bubble" x="${bx - rx}" y="${by - 13}" width="${rw}" height="26" rx="13"/><text class="call-value" x="${bx}" y="${by}">${esc(value)}</text><path class="lead" d="M${bx + dir} ${by} L${tx} ${ty}"/><circle class="lead-dot" cx="${tx}" cy="${ty}" r="2.6"/></g>`;
}

// per-architecture accent theming
function palette(arch = "") {
  if (/qwen/.test(arch)) return { fill: "#eaf6fd", stroke: "#0ea5e9", dark: "#0369a1", hot: "#cdeafb" };
  if (/deepseek|deci|nemotron/.test(arch)) return { fill: "#eef2ff", stroke: "#4f46e5", dark: "#3730a3", hot: "#dfe4ff" };
  if (/glm/.test(arch)) return { fill: "#eef4ff", stroke: "#2563eb", dark: "#1d4ed8", hot: "#dde8ff" };
  if (/kimi/.test(arch)) return { fill: "#fdeef6", stroke: "#db2777", dark: "#9d174d", hot: "#fbdcec" };
  if (/falcon|mamba|state|jamba|granite/.test(arch)) return { fill: "#fff4e6", stroke: "#ea580c", dark: "#c2410c", hot: "#ffe6cc" };
  return { fill: "#ecfdf5", stroke: "#0d9488", dark: "#0f766e", hot: "#d3f5ec" };
}

function style(a) {
  return `<style>
    .page{fill:#fcfcfd}
    .h-title{font:800 25px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:middle}
    .h-sub{font:600 13px ui-sans-serif,system-ui;fill:#475569;text-anchor:middle}
    .hint{font:600 11px ui-sans-serif,system-ui;fill:#64748b;text-anchor:start}
    text{font-family:ui-sans-serif,system-ui;dominant-baseline:middle;text-anchor:middle}
    .iobox rect{fill:#fff;stroke:#1e293b;stroke-width:1.5}
    .iobox text{font:700 13px ui-sans-serif,system-ui;fill:#0f172a}
    .iobox .io-sub{font:600 11px ui-sans-serif,system-ui;fill:#475569}
    .iobox.clickable{cursor:pointer}
    .iobox.selected rect{stroke:${a.stroke};stroke-width:3.2}
    .blockcard{fill:${a.fill};stroke:${a.stroke};stroke-width:2}
    .block-label{font:700 11.5px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:start;letter-spacing:.05em;text-transform:uppercase}
    .brace path{fill:none;stroke:#94a3b8;stroke-width:1.6}
    .brace-x{font:800 14px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:end}
    .brace-sub{font:600 10px ui-sans-serif,system-ui;fill:#475569;text-anchor:end}
    .flow{stroke:#1e293b;stroke-width:1.7;marker-end:url(#ah)}
    .skip{fill:none;stroke:#94a3b8;stroke-width:1.5;marker-end:url(#ahs)}
    .rope rect{fill:#fff;stroke:#1e293b;stroke-width:1.4}
    .rope text{font:700 12px ui-sans-serif,system-ui;fill:#0f172a}
    .rope line{stroke:#1e293b;stroke-width:1.7;marker-end:url(#ah)}
    .opnode rect{fill:#fff;stroke:#1e293b;stroke-width:1.6}
    .opnode text{font:700 13.5px ui-sans-serif,system-ui;fill:#0f172a}
    .opnode .op-sub{font:600 10.5px ui-sans-serif,system-ui;fill:#334155}
    .opnode.op-attn rect{fill:#27313f;stroke:#0f172a}
    .opnode.op-attn text{fill:#fff}
    .opnode.op-attn .op-sub{fill:#c7d2e0}
    .opnode.op-ffn rect{fill:${a.hot};stroke:${a.stroke}}
    .opnode.op-state rect{fill:#fde9d3;stroke:#ea580c}
    .opnode.op-mtp rect{fill:#ede9fe;stroke:#7c3aed}
    .opnode.op-norm rect{fill:#e7ecf2;stroke:#64748b}
    .opnode.op-norm text{fill:#334155;font-weight:700}
    .opnode.op-norm .op-sub{fill:#475569}
    .opnode.residual circle{fill:#fff;stroke:#1e293b;stroke-width:1.7}
    .opnode.residual text{font:700 18px ui-sans-serif,system-ui;fill:#0f172a}
    .opnode.selected rect,.opnode.selected circle{stroke:${a.stroke};stroke-width:3.2}
    .opnode{cursor:pointer}
    .callout .call-label{font:600 11px ui-sans-serif,system-ui;fill:#475569}
    .call-bubble{fill:#fff;stroke:${a.dark};stroke-width:1.7}
    .call-value{font:800 12px ui-sans-serif,system-ui;fill:${a.dark}}
    .lead{fill:none;stroke:#9aa6b2;stroke-width:1.3;stroke-dasharray:2 3}
    .lead-dot{fill:${a.stroke}}
    .inset-bd{fill:#fff;stroke:${a.stroke};stroke-width:1.6;stroke-dasharray:5 4}
    .inset-title{font:800 14px ui-sans-serif,system-ui;fill:#0f172a;text-anchor:start}
    .inset-sub{font:600 11px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:end}
    .mbox rect,.mbox circle{fill:#fff;stroke:#475569;stroke-width:1.3}
    .mbox text{font:600 11.5px ui-sans-serif,system-ui;fill:#0f172a}
    .mbox .m-sub{font:600 10px ui-sans-serif,system-ui;fill:#1f2937}
    .m-cap{font:600 11px ui-sans-serif,system-ui;fill:#334155;text-anchor:middle}
    .recur-label{font:600 9.5px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:start}
    .mbox.mhot rect{fill:${a.hot};stroke:${a.stroke}}
    .mbox.mdark rect{fill:#27313f}.mbox.mdark text{fill:#fff}
    .mbox.mnode circle{fill:#fff;stroke:#1e293b;stroke-width:1.5}.mbox.mnode text{font-weight:800}
    .mbox.wbox{cursor:pointer}
    .mbox.wbox:hover rect{stroke:${a.stroke};stroke-width:2.6}
    .mbox .wdot{fill:${a.stroke};stroke:none}
    .mbox.mdark .wdot{fill:#fff;stroke:none}
    .head-q{fill:${a.hot};stroke:${a.stroke};stroke-width:1.3}
    .head-kv{fill:#27313f;stroke:#0f172a;stroke-width:1.3}
    .head-kvt{font:700 10px ui-sans-serif,system-ui;fill:#fff}
    .head-cap{font:600 11px ui-sans-serif,system-ui;fill:#334155}
    .head-more{font:700 12px ui-sans-serif,system-ui;fill:#94a3b8;text-anchor:start}
    .head-brace{fill:none;stroke:${a.stroke};stroke-width:1.4}
    .exp-card{fill:#f1f5f9;stroke:${a.stroke};stroke-width:1.2}
    .exp-frame{fill:#fff;stroke:${a.stroke};stroke-width:1.7}
    .exp-title{font:700 9.5px ui-sans-serif,system-ui;fill:${a.dark};text-anchor:start;letter-spacing:.04em;text-transform:uppercase}
    .exp-badge{fill:${a.dark}}
    .exp-badge-t{font:800 12px ui-sans-serif,system-ui;fill:#fff}
    .mflow{fill:none;stroke:#475569;stroke-width:1.2;marker-end:url(#ahm)}
    .mflow.nh{marker-end:none}
    .mflow.recur{stroke:${a.stroke};stroke-dasharray:3 2}
  </style>`;
}
