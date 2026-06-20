import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { promisify } from "node:util";
import { pathToFileURL } from "node:url";
import { layoutGroups, layouts, pickInterestingOp, renderArchitectureSvg } from "../ui/diagram.mjs";

const exec = promisify(execFile);
const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.findIndex((arg) => arg === name || arg.startsWith(`${name}=`));
  if (i < 0) return fallback;
  return args[i].startsWith(`${name}=`) ? args[i].slice(name.length + 1) : args[i + 1];
};
const optionValues = new Set(["--out-dir", "--ids", "--pattern"].map((name) => opt(name)));
const positional = args.filter((arg) => !arg.startsWith("--") && !optionValues.has(arg));
const outDir = opt("--out-dir", ".cache/diagrams");
const ids = (opt("--ids", "") || "").split(",").map((id) => id.trim()).filter(Boolean);
const png = args.includes("--png");
const noInternals = args.includes("--no-internals");
const noSizes = args.includes("--no-sizes");
const patternIndex = Number(opt("--pattern", 0));
const chrome = process.env.CHROME_BIN ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const defaults = [
  "qwen_qwen3_coder_next_gguf_a2010fef",
  "deepseek_ai_deepseek_v4_flash_ee0ffc73",
  "moonshotai_kimi_linear_48b_a3b_instruct_24b0f4ba",
  "zai_org_glm_5_ce5b5e07",
  "tiiuae_falcon_h1_7b_base_855f892b",
  "nvidia_llama_3_1_nemotron_ultra_253b_v1_d089e67e"
];
const files = positional.length ? positional : (ids.length ? ids : defaults).map((id) => `db/model-ir/${id}.json`);

await mkdir(outDir, { recursive: true });
for (const file of files) {
  const ir = JSON.parse(await readFile(file, "utf8"));
  const groups = layoutGroups(layouts(ir));
  const group = groups[Math.min(patternIndex, groups.length - 1)] ?? groups[0];
  const range = group.ranges[0];
  const svg = renderArchitectureSvg(ir, {
    range,
    rangeIndex: range.index,
    groupIndex: groups.indexOf(group),
    selectedOpIndex: pickInterestingOp(range.layout),
    showSizes: !noSizes,
    showInternals: !noInternals,
    interactive: false
  });
  const name = `${ir.model.id ?? basename(file, ".json")}.svg`;
  const svgPath = resolve(outDir, name);
  await writeFile(svgPath, `${svg}\n`);
  if (png) await renderPng(svgPath, svg.replace(/\n/g, ""), svgPath.replace(/\.svg$/, ".png"));
  console.log(png ? svgPath.replace(/\.svg$/, ".png") : svgPath);
}

async function renderPng(svgPath, svg, pngPath) {
  if (!existsSync(chrome)) throw new Error(`Chrome not found: ${chrome}`);
  const [, width, height] = svg.match(/<svg[^>]+width="(\d+)"[^>]+height="(\d+)"/) ?? [];
  await exec(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    `--screenshot=${pngPath}`,
    `--window-size=${width ?? 1260},${height ?? 900}`,
    pathToFileURL(svgPath).href
  ]);
}
