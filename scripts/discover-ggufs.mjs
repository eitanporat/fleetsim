import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname } from "node:path";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.findIndex((arg) => arg === name || arg.startsWith(`${name}=`));
  if (i < 0) return fallback;
  return args[i].startsWith(`${name}=`) ? args[i].slice(name.length + 1) : args[i + 1];
};

const out = opt("--out", "db/ggufs.json");
const pageSize = Number(opt("--page-size", 1000));
const limit = Number(opt("--limit", Infinity));
const sort = opt("--sort", "likes");
const task = opt("--task", "text-generation");
const allFiles = args.includes("--all-files");
const dedup = !args.includes("--no-dedup");
const ogOnly = !args.includes("--include-derivatives");

const ogOwners = new Set([
  "01-ai", "ai21labs", "allenai", "baai", "baichuan-inc", "bigscience", "black-forest-labs",
  "cohere", "coherelabs", "deepseek-ai", "eleutherai", "facebook", "google", "google-deepmind",
  "huggingfacetb", "ibm-granite", "internlm", "lightricks", "meta-llama", "microsoft",
  "minimaxai", "mistralai", "mixedbread-ai", "moonshotai", "nexaai", "nvidia", "openai",
  "qwen", "sentence-transformers", "snowflake", "stabilityai", "tiiuae", "thudm", "tencent",
  "wan-ai", "zai-org"
]);

const slug = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
const hash = (value) => createHash("sha1").update(value).digest("hex").slice(0, 8);
const nextPage = (link) => link?.match(/<([^>]+)>;\s*rel="next"/)?.[1];
const ggufs = (model) => (model.siblings ?? []).map((file) => file.rfilename).filter((file) => file.endsWith(".gguf"));
const primary = (file) => !/mmproj|projector|adapter|lora/i.test(file);
const owner = (modelId) => modelId?.split("/")[0].toLowerCase();
const baseModel = (model) => (model.tags ?? [])
  .map((tag) => {
    const value = tag.match(/^base_model:(.+)$/)?.[1];
    if (!value) return undefined;
    const parts = value.split(":");
    return parts.length > 1 && parts.slice(1).join(":").includes("/") ? parts.slice(1).join(":") : value;
  })
  .find(Boolean);

function score(file) {
  const f = file.toLowerCase();
  return [
    primary(f) ? 0 : 1000,
    /q4_k_m/.test(f) ? 0 : /q4_0/.test(f) ? 1 : /q5_k_m/.test(f) ? 2 : /q8_0/.test(f) ? 3 : /iq4/.test(f) ? 4 : /q3_k_m/.test(f) ? 5 : /00001-of-\d+/.test(f) ? 6 : 20,
    /0000[2-9]-of-\d+/.test(f) ? 100 : 0,
    f.length
  ];
}

function compareFile(a, b) {
  const left = score(a);
  const right = score(b);
  for (let i = 0; i < left.length; i++) if (left[i] !== right[i]) return left[i] - right[i];
  return a.localeCompare(b);
}

function repoScore(entry) {
  const repo = entry.repo.toLowerCase();
  const base = entry.base_model?.toLowerCase();
  const owner = repo.split("/")[0];
  if (base && repo.startsWith(`${base}-`)) return 0;
  if (base && owner === base.split("/")[0]) return 1;
  if (owner === "ggml-org") return 2;
  if (owner === "unsloth") return 3;
  if (owner === "bartowski") return 4;
  if (owner === "lmstudio-community") return 5;
  return 10;
}

function popularity(a, b) {
  return (b.likes ?? 0) - (a.likes ?? 0)
    || (b.downloads ?? 0) - (a.downloads ?? 0)
    || (b.trending_score ?? 0) - (a.trending_score ?? 0)
    || (a.hf_rank ?? Infinity) - (b.hf_rank ?? Infinity);
}

function compareEntry(a, b) {
  const repo = repoScore(a) - repoScore(b);
  if (repo) return repo;
  const file = compareFile(a.file, b.file);
  if (file) return file;
  const pop = popularity(a, b);
  if (pop) return pop;
  return a.repo.localeCompare(b.repo);
}

function entries(model, hfRank) {
  if (model.gated) return [];
  const base_model = baseModel(model) ?? model.modelId;
  if (ogOnly && !ogOwners.has(owner(base_model ?? model.modelId))) return [];
  const files = ggufs(model).filter(primary).sort(compareFile);
  const selected = allFiles ? files : files.slice(0, 1);
  const group = dedup ? `base:${base_model.toLowerCase()}` : `repo:${model.modelId}`;
  return selected.map((file) => ({
    id: `${slug(base_model ?? (allFiles ? `${model.modelId}_${file}` : model.modelId))}_${hash(group)}`,
    base_model,
    repo: model.modelId,
    file,
    pipeline_tag: model.pipeline_tag,
    group,
    downloads: model.downloads ?? 0,
    likes: model.likes ?? 0,
    trending_score: model.trendingScore ?? 0,
    hf_rank: hfRank
  }));
}

const taskFilter = task === "all" ? "" : `&pipeline_tag=${encodeURIComponent(task)}`;
let url = `https://huggingface.co/api/models?filter=gguf${taskFilter}&sort=${encodeURIComponent(sort)}&direction=-1&limit=${pageSize}&full=false`;
const models = new Map();
const seen = new Set();
let scanned = 0;

while (url && models.size < limit) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HF API failed: ${res.status} ${res.statusText}`);
  const page = await res.json();
  for (const model of page) {
    scanned++;
    for (const entry of entries(model, scanned)) {
      const key = `${entry.repo}/${entry.file}`;
      if (seen.has(key)) continue;
      seen.add(key);
      if (!models.has(entry.group) || compareEntry(entry, models.get(entry.group)) < 0) models.set(entry.group, entry);
      if (models.size >= limit) break;
    }
    if (models.size >= limit) break;
  }
  console.error(`scanned ${scanned}, selected ${models.size} gguf model${models.size === 1 ? "" : "s"}`);
  url = nextPage(res.headers.get("link"));
}

const output = [...models.values()]
  .map(({ group, ...entry }) => entry)
  .sort((a, b) => popularity(a, b) || a.id.localeCompare(b.id));
await mkdir(dirname(out), { recursive: true });
await writeFile(out, `${JSON.stringify(output, null, 2)}\n`);
console.error(`wrote ${out}`);
