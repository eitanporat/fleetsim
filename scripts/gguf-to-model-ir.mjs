import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { ggufAllShards } from "@huggingface/gguf";

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.findIndex((arg) => arg === name || arg.startsWith(`${name}=`));
  if (i < 0) return fallback;
  return args[i].startsWith(`${name}=`) ? args[i].slice(name.length + 1) : args[i + 1];
};
const optionValues = new Set(["--models", "--out-dir", "--out", "--limit", "--concurrency"].map((name) => opt(name)));
const positional = args.filter((arg) => !arg.startsWith("--") && !optionValues.has(arg));

const modelsPath = opt("--models");
const outDir = opt("--out-dir");
const outFile = opt("--out");
const maxModels = Number(opt("--limit", Infinity));
const concurrency = Number(opt("--concurrency", 4));
const continueOnError = args.includes("--continue-on-error");

const num = (value) => value === undefined ? undefined : Number(value);
const present = (value) => value !== undefined && !(typeof value === "number" && Number.isNaN(value));
const compactObject = (object) => Object.fromEntries(Object.entries(object).filter(([, value]) => present(value)));
const tensorCount = (shape) => shape.reduce((acc, value) => acc * Number(value), 1);
const artifactUrl = (repo, file) => `https://huggingface.co/${repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}`;

function usage() {
  console.error("usage:");
  console.error("  npm run gguf:ir -- <path-or-url.gguf> [--out=model-ir.json]");
  console.error("  npm run gguf:ir -- --models=db/ggufs.json --out-dir=db/model-ir [--limit=100] [--continue-on-error]");
  process.exit(1);
}

function tensorParam(tensor, options = {}) {
  if (!tensor) return null;
  const shape = tensor.shape.map(num);
  return {
    name: tensor.name.replace(/blk\.\d+/g, "blk.N").replace(/\.(\d+)\./g, ".E."),
    shape,
    parameter_count: options.tiedTo ? 0 : tensorCount(shape) * (options.instances ?? 1),
    ...(options.instances ? { instances: options.instances } : {}),
    ...(options.tiedTo ? { tied_to: options.tiedTo } : {})
  };
}

function sortedByName(tensors) {
  return tensors.sort((a, b) => a.name.localeCompare(b.name));
}

function tensorsFor(tensorsByBlock, block, pattern) {
  return sortedByName((tensorsByBlock.get(block) ?? []).filter((tensor) => pattern.test(tensor.name)));
}

function groupedExpertParams(tensorsByBlock, block, stem) {
  const tensors = tensorsFor(tensorsByBlock, block, new RegExp(`${stem}\\.\\d+\\.weight$`));
  if (!tensors.length) return [];
  return [tensorParam(tensors[0], { instances: tensors.length })];
}

function firstParam(tensorsByBlock, block, pattern) {
  return tensorParam(tensorsFor(tensorsByBlock, block, pattern)[0]);
}

function allParams(tensorsByBlock, block, pattern) {
  return tensorsFor(tensorsByBlock, block, pattern).map((tensor) => tensorParam(tensor));
}

function normPattern(name) {
  const stem = {
    attention_norm: "attn_norm",
    attention_norm_2: "attn_norm_2",
    ffn_norm: "ffn_norm",
    post_attention_norm: "post_attention_norm",
    post_ffw_norm: "post_ffw_norm"
  }[name];
  return new RegExp(`${stem}\\.(weight|bias)$`);
}

function params(tensorsByBlock, block, patterns) {
  return patterns.flatMap((pattern) => {
    if (typeof pattern === "string") return groupedExpertParams(tensorsByBlock, block, pattern);
    const param = firstParam(tensorsByBlock, block, pattern);
    return param ? [param] : [];
  });
}

function metadataDimensions(metadata, arch, tensors) {
  const get = (key) => metadata[`${arch}.${key}`];
  const tokenEmbedding = tensors.find((tensor) => tensor.name === "token_embd.weight");
  const hiddenSize = num(get("embedding_length")) ?? num(tokenEmbedding?.shape[0]);
  const heads = num(get("attention.head_count"));
  const keyDim = num(get("attention.key_length")) ?? (hiddenSize && heads ? hiddenSize / heads : undefined);

  return compactObject({
    vocab_size: num(get("vocab_size")) ?? num(tokenEmbedding?.shape[1]),
    context_length: num(get("context_length")),
    hidden_size: hiddenSize,
    num_blocks: num(get("block_count")),
    num_attention_heads: heads,
    num_kv_heads: num(get("attention.head_count_kv")),
    head_dim: keyDim,
    value_head_dim: num(get("attention.value_length")) ?? keyDim,
    intermediate_size: num(get("feed_forward_length")),
    expert_count: num(get("expert_count")),
    experts_per_token: num(get("expert_used_count")),
    leading_dense_blocks: num(get("leading_dense_block_count")),
    nextn_blocks: num(get("nextn_predict_layers")),
    split_count: num(metadata["split.count"]),
    split_tensor_count: num(metadata["split.tensors.count"])
  });
}

function blockSignature(tensorsByBlock, block, arch) {
  const names = (tensorsByBlock.get(block) ?? []).map((tensor) => tensor.name);
  const has = (pattern) => names.some((name) => pattern.test(name));
  const hasMla = has(/attn_q_a|attn_kv_a_mqa|attn_k_b|attn_v_b/);
  const hasSsm = has(/ssm_/);
  const hasIndexer = has(/indexer\./);
  const ops = [];

  if (has(/attn_norm\.weight/)) ops.push("rms_norm:attention_norm");
  if (hasMla && hasIndexer) ops.push("dsa_mla_attention");
  else if (hasMla) ops.push("mla_attention");
  else if (hasSsm && arch === "kimi-linear") ops.push("kimi_delta_attention");
  else if (hasSsm && (has(/attn_qkv\.weight/) || /qwen3next|qwen35/.test(arch))) ops.push("gated_deltanet");
  else if (has(/attn_qkv\.weight/)) ops.push("fused_qkv_attention");
  else if (has(/attn_q\.weight|attn_k\.weight|attn_v\.weight/)) ops.push("self_attention");
  if (hasIndexer && !hasMla) ops.push("dsa_indexer");
  if (has(/post_attention_norm\.weight/)) ops.push("rms_norm:post_attention_norm");
  if (ops.some((op) => ["self_attention", "fused_qkv_attention", "mla_attention", "dsa_mla_attention", "dsa_indexer", "gated_deltanet", "kimi_delta_attention"].includes(op))) {
    ops.push("residual_add:attention_residual");
  }

  if (has(/ffn_norm\.weight/)) ops.push("rms_norm:ffn_norm");
  if (has(/ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate\.\d+\.weight/)) ops.push("moe_mlp");
  else if (has(/ffn_up\.weight/) && !has(/ffn_gate\.weight/)) ops.push("fused_mlp");
  else if (has(/ffn_gate\.weight|ffn_up\.weight|ffn_down\.weight/)) ops.push("mlp");
  if (has(/time_mix_/)) ops.push("rwkv_time_mix");
  if (has(/attn_norm_2\.weight/)) ops.push("rms_norm:attention_norm_2");
  if (has(/channel_mix_/)) ops.push("rwkv_channel_mix");
  if (hasSsm && !ops.some((op) => ["gated_deltanet", "kimi_delta_attention"].includes(op))) ops.push("state_space");
  if (has(/nextn\./)) ops.push("nextn_prediction");
  if (has(/post_ffw_norm\.weight/)) ops.push("rms_norm:post_ffw_norm");
  if (ops.some((op) => ["mlp", "fused_mlp", "moe_mlp", "rwkv_time_mix", "rwkv_channel_mix", "state_space"].includes(op))) {
    ops.push("residual_add:ffn_residual");
  }
  return ops.join("|");
}

function rangesFor(tensorsByBlock, count, arch) {
  if (!count || !tensorsByBlock.size) return [];
  count = Math.min(count, Math.max(...tensorsByBlock.keys()) + 1);
  const ranges = [];
  let start = 0;
  let last = blockSignature(tensorsByBlock, 0, arch);

  for (let block = 1; block < count; block++) {
    const current = blockSignature(tensorsByBlock, block, arch);
    if (current !== last) {
      if (last) ranges.push({ start, end: block - 1, signature: last });
      start = block;
      last = current;
    }
  }
  if (last) ranges.push({ start, end: count - 1, signature: last });
  return ranges;
}

function opLayout(signature, block, tensorsByBlock, dimensions) {
  return signature.split("|").filter(Boolean).map((entry) => {
    const [op, name = op] = entry.split(":");
    if (op === "rms_norm") return {
      op,
      name,
      parameters: allParams(tensorsByBlock, block, normPattern(name))
    };
    if (op === "self_attention") return {
      op,
      name: "attention",
      config: {
        num_heads: dimensions.num_attention_heads,
        num_kv_heads: dimensions.num_kv_heads,
        head_dim: dimensions.head_dim,
        rope: true
      },
      parameters: params(tensorsByBlock, block, [
        /attn_q\.weight$/, /attn_q\.bias$/, /attn_q_norm\.weight$/,
        /attn_k\.weight$/, /attn_k\.bias$/, /attn_k_norm\.weight$/,
        /attn_v\.weight$/, /attn_v\.bias$/, /attn_sinks\.weight$/,
        /attn_output\.weight$/, /attn_output\.bias$/
      ])
    };
    if (op === "fused_qkv_attention") return {
      op,
      name: "attention",
      config: { num_heads: dimensions.num_attention_heads, num_kv_heads: dimensions.num_kv_heads, head_dim: dimensions.head_dim, rope: true },
      parameters: params(tensorsByBlock, block, [/attn_qkv\.weight$/, /attn_qkv\.bias$/, /attn_output\.weight$/, /attn_output\.bias$/])
    };
    if (op === "mla_attention") return {
      op,
      name: "attention",
      config: { num_heads: dimensions.num_attention_heads, q_lora_rank: dimensions.q_lora_rank, kv_lora_rank: dimensions.kv_lora_rank },
      parameters: params(tensorsByBlock, block, [
        /attn_q\.weight$/, /attn_q_a\.weight$/, /attn_q_a_norm\.weight$/, /attn_q_b\.weight$/,
        /attn_kv\.weight$/, /attn_kv_a_mqa\.weight$/, /attn_kv_a_norm\.weight$/, /attn_k_b\.weight$/,
        /attn_v_b\.weight$/, /attn_kv_b\.weight$/, /attn_output\.weight$/, /attn_output_a\.weight$/, /attn_output_b\.weight$/,
        /attn_sinks\.weight$/, /hc_attn_.*\.weight$/, /attn_compressor.*\.weight$/
      ])
    };
    if (op === "dsa_mla_attention") return {
      op,
      name: "attention",
      config: { num_heads: dimensions.num_attention_heads, q_lora_rank: dimensions.q_lora_rank, kv_lora_rank: dimensions.kv_lora_rank, sparse_indexer: true },
      parameters: params(tensorsByBlock, block, [
        /attn_q\.weight$/, /attn_q_a\.weight$/, /attn_q_a_norm\.weight$/, /attn_q_b\.weight$/,
        /attn_kv\.weight$/, /attn_kv_a_mqa\.weight$/, /attn_kv_a_norm\.weight$/, /attn_k_b\.weight$/,
        /attn_v_b\.weight$/, /attn_kv_b\.weight$/, /attn_output\.weight$/, /attn_output_a\.weight$/, /attn_output_b\.weight$/,
        /attn_sinks\.weight$/, /hc_attn_.*\.weight$/, /attn_compressor.*\.weight$/,
        /indexer\..*\.(weight|bias)$/, /indexer_compressor.*\.weight$/
      ])
    };
    if (op === "gated_deltanet") return {
      op,
      name: "linear_attention",
      config: {
        variant: "gated_delta",
        input_projection: tensorsFor(tensorsByBlock, block, /attn_qkv\.weight$/).length ? "attn_qkv" : "ssm_in",
        recurrent_state: true
      },
      parameters: [
        ...params(tensorsByBlock, block, [/attn_qkv\.weight$/, /attn_qkv\.bias$/, /attn_gate\.weight$/, /attn_output\.weight$/, /attn_output\.bias$/]),
        ...allParams(tensorsByBlock, block, /ssm_/)
      ]
    };
    if (op === "kimi_delta_attention") return {
      op,
      name: "linear_attention",
      config: { variant: "kimi_delta", qkv_projection: "split", recurrent_state: true },
      parameters: [
        ...params(tensorsByBlock, block, [
          /attn_q\.weight$/, /attn_q\.bias$/, /attn_q_norm\.weight$/,
          /attn_k\.weight$/, /attn_k\.bias$/, /attn_k_norm\.weight$/,
          /attn_v\.weight$/, /attn_v\.bias$/, /attn_sinks\.weight$/,
          /attn_output\.weight$/, /attn_output\.bias$/
        ]),
        ...allParams(tensorsByBlock, block, /ssm_/)
      ]
    };
    if (op === "dsa_indexer") return {
      op,
      name: "indexer",
      parameters: params(tensorsByBlock, block, [
        /indexer\.proj\.weight$/, /indexer\.attn_k\.weight$/, /indexer\.attn_q_b\.weight$/,
        /indexer\.k_norm\.weight$/, /indexer\.k_norm\.bias$/
      ])
    };
    if (op === "mlp") return {
      op,
      name: "feed_forward",
      config: { activation: "silu", gating: "swiglu" },
      parameters: params(tensorsByBlock, block, [
        /ffn_gate\.weight$/, /ffn_gate\.bias$/, /ffn_up\.weight$/, /ffn_up\.bias$/, /ffn_down\.weight$/, /ffn_down\.bias$/
      ])
    };
    if (op === "fused_mlp") return {
      op,
      name: "feed_forward",
      config: { activation: "silu", gating: "fused" },
      parameters: params(tensorsByBlock, block, [/ffn_up\.weight$/, /ffn_up\.bias$/, /ffn_down\.weight$/, /ffn_down\.bias$/])
    };
    if (op === "moe_mlp") return {
      op,
      name: "expert_feed_forward",
      config: { expert_count: dimensions.expert_count, experts_per_token: dimensions.experts_per_token },
      parameters: params(tensorsByBlock, block, [
        /ffn_gate_inp\.weight$/, /ffn_gate_inp\.bias$/, /exp_probs_b\.bias$/,
        /ffn_gate_exps\.weight$/, /ffn_gate_exps\.bias$/, /ffn_up_exps\.weight$/, /ffn_up_exps\.bias$/,
        /ffn_down_exps\.weight$/, /ffn_down_exps\.bias$/,
        "ffn_gate", "ffn_up", "ffn_down",
        /ffn_gate_shexp\.weight$/, /ffn_up_shexp\.weight$/, /ffn_down_shexp\.weight$/
      ])
    };
    if (op === "rwkv_time_mix") return {
      op,
      name: "time_mix",
      parameters: allParams(tensorsByBlock, block, /time_mix_/)
    };
    if (op === "rwkv_channel_mix") return {
      op,
      name: "channel_mix",
      parameters: allParams(tensorsByBlock, block, /channel_mix_/)
    };
    if (op === "state_space") return {
      op,
      name: "state_space",
      parameters: allParams(tensorsByBlock, block, /ssm_/)
    };
    if (op === "nextn_prediction") return {
      op,
      name: "nextn",
      parameters: allParams(tensorsByBlock, block, /nextn\./)
    };
    return { op, name };
  });
}

function sumLayoutParams(layout) {
  return layout.flatMap((op) => op.parameters ?? []).reduce((sum, param) => sum + param.parameter_count, 0);
}

async function parseModel(model) {
  const input = model.url ?? (model.repo ? artifactUrl(model.repo, model.file) : model.file);
  const parsed = await ggufAllShards(input, { allowLocalFile: true, parallelDownloads: 4 });
  const metadata = parsed.shards[0].metadata;
  const arch = metadata["general.architecture"];
  const tensors = parsed.shards.flatMap((shard) => shard.tensorInfos);
  const tensorsByBlock = new Map();
  for (const tensor of tensors) {
    const match = tensor.name.match(/^blk\.(\d+)\./);
    if (!match) continue;
    const block = Number(match[1]);
    tensorsByBlock.set(block, [...(tensorsByBlock.get(block) ?? []), tensor]);
  }

  const dimensions = metadataDimensions(metadata, arch, tensors);
  dimensions.q_lora_rank = num(metadata[`${arch}.attention.q_lora_rank`]);
  dimensions.kv_lora_rank = num(metadata[`${arch}.attention.kv_lora_rank`]);
  for (const key of ["q_lora_rank", "kv_lora_rank"]) if (!present(dimensions[key])) delete dimensions[key];

  const tokenEmbedding = tensors.find((tensor) => tensor.name === "token_embd.weight");
  const tokenEmbeddingNorm = tensors.filter((tensor) => /^token_embd_norm\.(weight|bias)$/.test(tensor.name));
  const ropeFreqs = tensors.find((tensor) => tensor.name === "rope_freqs.weight");
  const outputNorm = tensors.find((tensor) => tensor.name === "output_norm.weight");
  const outputNormBias = tensors.find((tensor) => tensor.name === "output_norm.bias");
  const outputWeight = tensors.find((tensor) => tensor.name === "output.weight");
  const outputBias = tensors.find((tensor) => tensor.name === "output.bias");
  const ranges = rangesFor(tensorsByBlock, dimensions.num_blocks, arch);

  const blockRanges = ranges.map((range) => {
    const layout = opLayout(range.signature, range.start, tensorsByBlock, dimensions);
    const perBlockParameterCount = sumLayoutParams(layout);
    return {
      range: [range.start, range.end],
      layout,
      per_block_parameter_count: perBlockParameterCount,
      total_parameter_count: perBlockParameterCount * (range.end - range.start + 1)
    };
  });
  if (dimensions.num_blocks && blockRanges.every((range) => !range.layout.length)) {
    throw new Error("no recognized block layout tensors");
  }

  const metadataName = metadata["general.name"];
  const hfName = model.base_model ?? model.repo;
  const name = hfName ?? metadataName ?? basename(input, ".gguf");

  return {
    schema: "fleetsim.model_architecture_ir.v0",
    model: {
      id: model.id ?? basename(input, ".gguf"),
      name,
      ...(hfName ? { hf_name: hfName } : {}),
      ...(model.base_model ? { base_model: model.base_model } : {}),
      ...(metadataName && metadataName !== name ? { artifact_name: metadataName } : {}),
      ...(model.repo ? { repo: model.repo, file: model.file } : {}),
      architecture: arch,
      source_format: "gguf",
      source: input,
      parameter_count: parsed.parameterCount
    },
    dimensions,
    embeddings: [
      tokenEmbedding && { op: "embedding", name: "token_embedding", parameters: [tensorParam(tokenEmbedding)] },
      tokenEmbeddingNorm.length && { op: "rms_norm", name: "token_embedding_norm", parameters: tokenEmbeddingNorm.map((tensor) => tensorParam(tensor)) },
      ropeFreqs && { op: "rope_frequencies", name: "rope_freqs", parameters: [tensorParam(ropeFreqs)] }
    ].filter(Boolean),
    blocks: {
      count: dimensions.num_blocks ?? 0,
      ranges: blockRanges,
      total_parameter_count: blockRanges.reduce((sum, range) => sum + range.total_parameter_count, 0)
    },
    output: [
      outputNorm && { op: "rms_norm", name: "output_norm", parameters: [outputNorm, outputNormBias].filter(Boolean).map((tensor) => tensorParam(tensor)) },
      {
        op: "linear",
        name: "lm_head",
        parameters: [
          outputWeight ? tensorParam(outputWeight) : tensorParam(tokenEmbedding, { tiedTo: "token_embedding" }),
          outputBias && tensorParam(outputBias)
        ].filter(Boolean)
      }
    ].filter(Boolean)
  };
}

async function writeIr(ir, path) {
  await writeFile(path, `${JSON.stringify(ir, null, 2)}\n`);
}

function irOps(ir) {
  const ranges = ir.blocks?.ranges ?? [];
  return [...new Set([
    ...(ir.embeddings ?? []),
    ...ranges.flatMap((range) => range.layout ?? []),
    ...(ir.output ?? [])
  ].map((op) => op.op).filter(Boolean))].sort();
}

if (!modelsPath && !positional[0]) usage();

if (modelsPath) {
  if (!outDir) usage();
  await mkdir(outDir, { recursive: true });
  const models = JSON.parse(await readFile(modelsPath, "utf8")).slice(0, maxModels);
  const results = [];
  const errors = [];
  let next = 0;

  async function worker() {
    while (next < models.length) {
      const model = models[next++];
      try {
        console.error(`parsing ${model.id}`);
        const ir = await parseModel(model);
        const file = `${model.id}.json`;
        await writeIr(ir, join(outDir, file));
        results.push({
          id: model.id,
          name: ir.model.name,
          ...(model.base_model ? { base_model: model.base_model } : {}),
          ...(model.repo ? { repo: model.repo } : {}),
          file,
          architecture: ir.model.architecture,
          source_format: ir.model.source_format,
          parameter_count: ir.model.parameter_count,
          block_count: ir.blocks?.count,
          context_length: ir.dimensions?.context_length,
          downloads: model.downloads,
          likes: model.likes,
          hf_rank: model.hf_rank,
          ops: irOps(ir)
        });
      } catch (error) {
        if (!continueOnError) throw error;
        console.error(`failed ${model.id}: ${error.message}`);
        await rm(join(outDir, `${model.id}.json`), { force: true });
        errors.push({ id: model.id, repo: model.repo, file: model.file, error: error.message });
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, models.length) }, worker));
  const order = new Map(models.map((model, index) => [model.id, index]));
  const index = results.sort((a, b) => order.get(a.id) - order.get(b.id));
  await writeIr({ schema: "fleetsim.model_architecture_ir.index.v0", models: index, ...(errors.length ? { errors } : {}) }, join(outDir, "index.json"));
} else {
  const ir = await parseModel({ file: positional[0] });
  if (outFile) await writeIr(ir, outFile);
  else process.stdout.write(`${JSON.stringify(ir, null, 2)}\n`);
}
