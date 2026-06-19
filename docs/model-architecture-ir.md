# ModelArchitectureIR

`ModelArchitectureIR` is FleetSim's normalized description of one concrete model.

It answers:

- how many blocks the model has
- what each block does, in order
- which tensors each layer uses
- the numeric shape and parameter count for every tensor

It does not describe serving, batching, kernels, hardware, or pricing.

## Rules

- The IR is concrete: dimensions must be numbers, not symbols like `n_embd`.
- `parameter_count` is the math parameter count: product of tensor shape.
- Quantization affects `storage`, not `parameter_count`.
- Blocks may use one `shared_layout` or multiple `ranges` when the model changes layout by depth.
- Operation order matters.
- Tensor names should be artifact-style names such as `blk.N.attn_q.weight`, not implementation enum names.

## Shape

```json
{
  "schema": "fleetsim.model_architecture_ir.v0",
  "model": {
    "name": "string",
    "architecture": "string",
    "source_format": "gguf | hf_config | vllm | manual",
    "parameter_count": 0
  },
  "dimensions": {
    "vocab_size": 0,
    "context_length": 0,
    "hidden_size": 0,
    "num_blocks": 0
  },
  "embeddings": [],
  "blocks": {
    "count": 0,
    "shared_layout": []
  },
  "output": []
}
```

## Example

```json
{
  "schema": "fleetsim.model_architecture_ir.v0",
  "model": {
    "name": "Llama-3.2-1B-Instruct",
    "architecture": "llama",
    "source_format": "gguf",
    "parameter_count": 1235814432
  },
  "dimensions": {
    "vocab_size": 128256,
    "context_length": 131072,
    "hidden_size": 2048,
    "num_blocks": 16,
    "num_attention_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 64,
    "intermediate_size": 8192
  },
  "embeddings": [
    {
      "op": "embedding",
      "name": "token_embedding",
      "parameters": [
        {
          "name": "token_embd.weight",
          "shape": [128256, 2048],
          "parameter_count": 262668288
        }
      ]
    },
    {
      "op": "rope_frequencies",
      "name": "rope_freqs",
      "parameters": [
        {
          "name": "rope_freqs.weight",
          "shape": [32],
          "parameter_count": 32
        }
      ]
    }
  ],
  "blocks": {
    "count": 16,
    "shared_layout": [
      {
        "op": "rms_norm",
        "name": "attention_norm",
        "parameters": [
          {
            "name": "blk.N.attn_norm.weight",
            "shape": [2048],
            "parameter_count": 2048
          }
        ]
      },
      {
        "op": "self_attention",
        "name": "attention",
        "config": {
          "num_heads": 32,
          "num_kv_heads": 8,
          "head_dim": 64,
          "rope": true
        },
        "parameters": [
          {
            "name": "blk.N.attn_q.weight",
            "shape": [2048, 2048],
            "parameter_count": 4194304
          },
          {
            "name": "blk.N.attn_k.weight",
            "shape": [512, 2048],
            "parameter_count": 1048576
          },
          {
            "name": "blk.N.attn_v.weight",
            "shape": [512, 2048],
            "parameter_count": 1048576
          },
          {
            "name": "blk.N.attn_output.weight",
            "shape": [2048, 2048],
            "parameter_count": 4194304
          }
        ]
      },
      {
        "op": "residual_add",
        "name": "attention_residual"
      },
      {
        "op": "rms_norm",
        "name": "ffn_norm",
        "parameters": [
          {
            "name": "blk.N.ffn_norm.weight",
            "shape": [2048],
            "parameter_count": 2048
          }
        ]
      },
      {
        "op": "mlp",
        "name": "feed_forward",
        "config": {
          "activation": "silu",
          "gating": "swiglu",
          "hidden_size": 2048,
          "intermediate_size": 8192
        },
        "parameters": [
          {
            "name": "blk.N.ffn_gate.weight",
            "shape": [8192, 2048],
            "parameter_count": 16777216
          },
          {
            "name": "blk.N.ffn_up.weight",
            "shape": [8192, 2048],
            "parameter_count": 16777216
          },
          {
            "name": "blk.N.ffn_down.weight",
            "shape": [2048, 8192],
            "parameter_count": 16777216
          }
        ]
      },
      {
        "op": "residual_add",
        "name": "ffn_residual"
      }
    ],
    "per_block_parameter_count": 60821504,
    "total_block_parameter_count": 973144064
  },
  "output": [
    {
      "op": "rms_norm",
      "name": "output_norm",
      "parameters": [
        {
          "name": "output_norm.weight",
          "shape": [2048],
          "parameter_count": 2048
        }
      ]
    },
    {
      "op": "linear",
      "name": "lm_head",
      "parameters": [
        {
          "name": "token_embd.weight",
          "shape": [128256, 2048],
          "parameter_count": 0,
          "tied_to": "token_embedding"
        }
      ]
    }
  ]
}
```

## Non-Uniform Blocks

Models with changing block structure use ranges:

```json
{
  "blocks": {
    "count": 79,
    "ranges": [
      {
        "range": [0, 2],
        "layout": ["rms_norm", "mla_attention", "residual_add", "rms_norm", "dense_mlp", "residual_add"]
      },
      {
        "range": [3, 77],
        "layout": ["rms_norm", "mla_attention", "dsa_indexer", "residual_add", "rms_norm", "moe_mlp", "shared_expert", "residual_add"]
      },
      {
        "range": [78, 78],
        "layout": ["next_token_prediction_head"]
      }
    ]
  }
}
```
