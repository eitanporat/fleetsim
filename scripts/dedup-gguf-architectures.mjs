import { appendFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname } from "node:path";
import { gguf } from "@huggingface/gguf";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.findIndex((arg) => arg === name || arg.startsWith(`${name}=`));
  if (i < 0) return fallback;
  return args[i].startsWith(`${name}=`) ? args[i].slice(name.length + 1) : args[i + 1];
};

const modelsPath = opt("--models", "db/ggufs.json");
const out = opt("--out", "db/gguf-architectures.json");
const cachePath = opt("--cache", ".cache/gguf-architecture-cache.jsonl");
const limit = Number(opt("--limit", Infinity));
const concurrency = Number(opt("--concurrency", 4));

const hash = (value) => createHash("sha1").update(value).digest("hex").slice(0, 12);
const artifactUrl = (repo, file) => `https://huggingface.co/${repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}`;
const scalar = (value) => ["string", "number", "boolean"].includes(typeof value);
const normTensor = (name) => name.replace(/^blk\.\d+\./, "blk.N.").replace(/\.\d+\./g, ".E.");

function fileScore(file) {
  const f = file.toLowerCase();
  return /q4_k_m/.test(f) ? 0 : /q4_0/.test(f) ? 1 : /q5_k_m/.test(f) ? 2 : /q8_0/.test(f) ? 3 : /iq4/.test(f) ? 4 : 20;
}

function repoScore(entry) {
  const owner = entry.repo.toLowerCase().split("/")[0];
  if (entry.base_model && owner === entry.base_model.toLowerCase().split("/")[0]) return 0;
  return owner === "ggml-org" ? 1 : owner === "unsloth" ? 2 : owner === "bartowski" ? 3 : owner === "lmstudio-community" ? 4 : 10;
}

function better(a, b) {
  return repoScore(a) - repoScore(b)
    || fileScore(a.file) - fileScore(b.file)
    || (b.likes ?? 0) - (a.likes ?? 0)
    || (b.downloads ?? 0) - (a.downloads ?? 0)
    || a.repo.localeCompare(b.repo);
}

function popularity(a, b) {
  return (b.likes ?? 0) - (a.likes ?? 0)
    || (b.downloads ?? 0) - (a.downloads ?? 0)
    || (a.hf_rank ?? Infinity) - (b.hf_rank ?? Infinity)
    || a.id.localeCompare(b.id);
}

function signature(parsed) {
  const metadata = parsed.metadata;
  const arch = metadata["general.architecture"] ?? "unknown";
  const archMeta = Object.fromEntries(Object.entries(metadata)
    .filter(([key, value]) => key.startsWith(`${arch}.`) && (scalar(value) || (Array.isArray(value) && value.length <= 16 && value.every(scalar))))
    .sort(([a], [b]) => a.localeCompare(b)));
  const tensors = parsed.tensorInfos.map((tensor) => [normTensor(tensor.name), tensor.shape.map(Number)]).sort();
  const shape = { arch, archMeta, split_tensor_count: metadata["split.tensors.count"], tensors };
  return { architecture: arch, architecture_signature: hash(JSON.stringify(shape)) };
}

async function loadCache() {
  if (!cachePath) return new Map();
  const lines = await readFile(cachePath, "utf8").catch(() => "");
  return new Map(lines.split("\n").filter(Boolean).map((line) => {
    const record = JSON.parse(line);
    return [record.key, record];
  }));
}

const models = JSON.parse(await readFile(modelsPath, "utf8")).slice(0, limit);
const cache = await loadCache();
const groups = new Map();
const errors = [];
let next = 0;
let done = 0;

async function record(model) {
  const key = `${model.repo}/${model.file}`;
  if (cache.has(key)) return cache.get(key);
  const sig = signature(await gguf(artifactUrl(model.repo, model.file)));
  const entry = { key, ...sig, model };
  if (cachePath) await appendFile(cachePath, `${JSON.stringify(entry)}\n`);
  return entry;
}

async function worker() {
  while (next < models.length) {
    const model = models[next++];
    try {
      const parsed = await record(model);
      const entry = { ...parsed.model, architecture: parsed.architecture, architecture_signature: parsed.architecture_signature };
      const current = groups.get(parsed.architecture_signature);
      if (!current || better(entry, current) < 0) groups.set(parsed.architecture_signature, entry);
    } catch (error) {
      errors.push({ repo: model.repo, file: model.file, error: error.message });
    }
    done++;
    if (done % 100 === 0 || done === models.length) {
      console.error(`parsed ${done}/${models.length}, selected ${groups.size}, errors ${errors.length}`);
    }
  }
}

await mkdir(dirname(out), { recursive: true });
if (cachePath) await mkdir(dirname(cachePath), { recursive: true });
await Promise.all(Array.from({ length: concurrency }, worker));
await writeFile(out, `${JSON.stringify([...groups.values()].sort(popularity), null, 2)}\n`);
const errorsPath = out.replace(/\.json$/, ".errors.json");
if (errors.length) await writeFile(errorsPath, `${JSON.stringify(errors, null, 2)}\n`);
else await rm(errorsPath, { force: true });
console.error(`wrote ${out}`);
