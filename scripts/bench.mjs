// Visual benchmark: renders a matrix of real screenshots into .cache/bench/.
// The matrix is intentionally easy to extend — the reviewing subagent may
// propose new scenarios/features and they get appended to SCENARIOS below.
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, rm, readFile, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { renderArchitectureSvg, layouts, layoutGroups, pickInterestingOp } from "../ui/diagram.mjs";

const exec = promisify(execFile);
const CHROME = process.env.CHROME_BIN ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const ROOT = resolve(import.meta.dirname, "..");
const OUT = resolve(ROOT, ".cache/bench");
const HOST = process.env.BENCH_HOST ?? "http://localhost:5174";

const M = {
  qwen: "qwen_qwen3_coder_next_gguf_a2010fef",
  deepseek: "deepseek_ai_deepseek_v4_flash_ee0ffc73",
  kimi: "moonshotai_kimi_linear_48b_a3b_instruct_24b0f4ba",
  glm: "zai_org_glm_5_ce5b5e07",
  falcon: "tiiuae_falcon_h1_7b_base_855f892b",
  nemo: "nvidia_llama_3_1_nemotron_ultra_253b_v1_d089e67e"
};

// kind: "fig" = full architecture figure · "inset" = right-column crop · "app" = live app state
const SCENARIOS = [
  { name: "fig-qwen", kind: "fig", file: M.qwen, op: 1, note: "Qwen3-Coder-Next full figure (Gated DeltaNet selected)" },
  { name: "inset-qwen-deltanet", kind: "inset", file: M.qwen, op: 1, note: "Gated DeltaNet internals" },
  { name: "inset-qwen-moe", kind: "inset", file: M.qwen, op: 4, note: "Mixture-of-Experts stacked-deck internals" },
  { name: "inset-qwen-norm", kind: "inset", file: M.qwen, op: 0, note: "RMSNorm internals" },
  { name: "inset-qwen-residual", kind: "inset", file: M.qwen, op: 3, note: "Residual add internals" },
  { name: "inset-qwen-embedding", kind: "inset", file: M.qwen, outer: "embedding", note: "Token embedding (outer node)" },
  { name: "inset-qwen-head", kind: "inset", file: M.qwen, outer: "head", note: "Output head (outer node)" },
  { name: "fig-deepseek", kind: "fig", file: M.deepseek, op: 1, note: "DeepSeek-V4-Flash full figure (MLA)" },
  { name: "inset-deepseek-mla", kind: "inset", file: M.deepseek, group: 0, op: 1, note: "Multi-head Latent Attention internals" },
  { name: "inset-deepseek-dsa", kind: "inset", file: M.deepseek, group: 1, op: 1, note: "MLA + Sparse Attention (DSA) internals" },
  { name: "inset-kimi-delta", kind: "inset", file: M.kimi, op: 1, note: "Kimi Delta Attention internals" },
  { name: "fig-glm", kind: "fig", file: M.glm, group: 1, op: 1, note: "GLM-5 full figure (DSA + MoE)" },
  { name: "inset-glm-mtp", kind: "inset", file: M.glm, group: 2, op: 5, note: "Multi-Token Prediction head" },
  { name: "fig-falcon", kind: "fig", file: M.falcon, op: 1, note: "Falcon-H1 full figure (GQA + SwiGLU + Mamba)" },
  { name: "inset-falcon-gqa", kind: "inset", file: M.falcon, op: 1, note: "Grouped-Query Attention + head-grouping visual" },
  { name: "inset-falcon-swiglu", kind: "inset", file: M.falcon, op: 3, note: "SwiGLU FeedForward internals" },
  { name: "inset-falcon-mamba", kind: "inset", file: M.falcon, op: 4, note: "State-Space (Mamba) internals" },
  { name: "inset-nemo-mha", kind: "inset", file: M.nemo, op: 1, note: "Multi-Head Attention (no GQA)" },
  { name: "inset-qwen-nosizes", kind: "inset", file: M.qwen, op: 4, sizes: 0, note: "Sizes toggle OFF" },
  { name: "inset-qwen-nointernals", kind: "fig", file: M.qwen, op: 1, internals: 0, note: "Internals toggle OFF (no inset)" },
  { name: "app-landing", kind: "app", w: 1512, h: 950, note: "Landing / first load" },
  { name: "app-diagram", kind: "app", w: 1512, h: 1100, query: { model: M.falcon, view: "diagram", op: 1 }, note: "Diagram view + toolbar icons" },
  { name: "app-patterns", kind: "app", w: 1512, h: 1000, query: { model: M.qwen, view: "pattern" }, note: "Patterns view" },
  { name: "app-inspector", kind: "app", w: 1380, h: 1700, query: { model: M.falcon, view: "diagram", op: 1 }, note: "Op Inspector (collapsible, selected expanded)" },
  { name: "app-fullscreen", kind: "app", w: 1512, h: 950, query: { model: M.deepseek, view: "diagram", op: 1, fullscreen: 1 }, note: "Fullscreen overlay" },
  // NOTE: headless Chrome clamps the layout viewport to a 500px minimum, so the
  // capture width must be >=500 or the right edge of a correctly-laid-out page is
  // cropped off (a capture artifact, not a real overflow). 500 = true mobile min.
  { name: "app-mobile", kind: "app", w: 500, h: 2800, query: { model: M.qwen, view: "diagram", op: 1 }, note: "Narrow / mobile viewport — FULL PAGE (scrolls: filters → registry → diagram → inspector)" }
];

async function shotFile(svgPath, png, w, h) {
  await exec(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--force-device-scale-factor=3", `--screenshot=${png}`, `--window-size=${w},${h}`, pathToFileURL(svgPath).href]);
}
async function shotUrl(url, png, w, h) {
  await exec(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--force-device-scale-factor=2", "--virtual-time-budget=3500", `--screenshot=${png}`, `--window-size=${w},${h}`, url]);
}

await rm(OUT, { recursive: true, force: true });
await mkdir(OUT, { recursive: true });
if (!existsSync(CHROME)) throw new Error(`Chrome not found: ${CHROME}`);

const manifest = [];
for (const s of SCENARIOS) {
  const png = resolve(OUT, `${s.name}.png`);
  if (s.kind === "app") {
    const q = new URLSearchParams(s.query ?? {}).toString();
    await shotUrl(`${HOST}/ui/${q ? "?" + q : ""}`, png, s.w, s.h);
  } else {
    const ir = JSON.parse(await readFile(resolve(ROOT, "db/model-ir", `${s.file}.json`), "utf8"));
    const groups = layoutGroups(layouts(ir));
    const group = groups[Math.min(s.group ?? 0, groups.length - 1)];
    const range = group.ranges[0];
    let svg = renderArchitectureSvg(ir, {
      range, rangeIndex: range.index, groupIndex: groups.indexOf(group),
      selectedOpIndex: s.op ?? pickInterestingOp(range.layout), selectedOuter: s.outer ?? null,
      showSizes: s.sizes !== 0, showInternals: s.internals !== 0, interactive: false
    });
    const m = svg.match(/width="(\d+)" height="(\d+)" viewBox="0 0 \d+ \d+"/);
    let W = +m[1], H = +m[2];
    if (s.kind === "inset") {
      // tight crop to the inset panel's actual bounds (avoids wasted vertical space)
      const bd = svg.match(/class="inset-bd" x="(\d+)" y="(\d+)" width="(\d+)" height="(\d+)"/);
      const x0 = bd ? +bd[1] - 14 : 686, y0 = bd ? +bd[2] - 6 : 246;
      const cw = W - x0, ch = bd ? +bd[4] + 18 : H - y0;
      svg = svg.replace(/width="\d+" height="\d+" viewBox="0 0 \d+ \d+"/, `width="${cw}" height="${ch}" viewBox="${x0} ${y0} ${cw} ${ch}"`);
      W = cw; H = ch;
    }
    const svgPath = resolve(OUT, `${s.name}.svg`);
    await writeFile(svgPath, svg);
    await shotFile(svgPath, png, W, H);
  }
  manifest.push({ name: s.name, png: `${s.name}.png`, kind: s.kind, note: s.note });
  console.log(s.name);
}
await writeFile(resolve(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`\n${manifest.length} screenshots → ${OUT}`);
