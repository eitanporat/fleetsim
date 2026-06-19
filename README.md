# FleetSim

FleetSim is an open-source simulator and optimizer for AI inference and training workloads across heterogeneous hardware fleets.

Large AI labs have internal tools for asking which hardware, kernels, model formats, parallelism modes, providers, and prices best satisfy a workload SLO. The open-source ecosystem does not have a shared version of that layer. FleetSim aims to build one.

The long-term goal is a system where, instead of benchmarking every deployment by hand, users can describe a model, hardware fleet, pricing model, and SLO, then simulate or optimize across possible deployments. FleetSim should also aggregate compute supply, pricing, and availability across providers, then optimize directly over the real market for the best deployment.

The vision is plug-and-play: provide a workload and model, and FleetSim returns the optimal compute fleet across NVIDIA GPUs, AMD GPUs, Google TPUs, AWS Trainium, CPUs, storage, and networking, with exact costs from actual provider data.

FleetSim starts small: model extraction and kernel measurements.

## Initial Scope

FleetSim V0 focuses on two layers:

1. Models
   Architectures, tensor shapes, model formats, and weight quantization.

2. Kernel Layer
   FlashInfer, FlashAttention, Triton, CUTLASS, attention kernels, GEMM, MoE, norms, and collectives.

The first milestone is to extract model structure from real artifacts and map that structure to measurable kernel shapes.

## Measurements

FleetSim is measurement-first. Runtime estimates should come from measured data whenever possible.

A measurement records:

- model architecture
- kernel name
- operation type
- tensor shape
- dtype or quantization
- hardware
- software stack
- runtime distribution

These measurements become the runtime database used by later simulation and optimization layers.

## TODO

- Backend scheduling: waiting queues, running queues, batching, prefill/decode scheduling.
- Deployment routing: load balancing, replica routing, cache-aware routing, request placement.
- Decoding strategies: standard decoding, speculative decoding, draft models, MTP, EAGLE, n-gram speculation.
- Cache strategies: KV cache layout, KV allocation, KV reuse, prefix caching, KV eviction, KV quantization.
- Disaggregation: prefill/decode separation, KV transfer, KV connectors, KV offload tiers.
- Parallelism: tensor, pipeline, data, expert, context parallelism, replicas.
- Infrastructure: GPUs, CPUs, memory, storage, interconnects, networking, cluster topology, placement.
- Providers: cloud and bare-metal providers, pricing, regions, availability.
- Compute aggregation: available hardware, provider inventory, capacity, and price feeds.
- Workloads: synthetic workloads, measured traces, traffic models, agent steps, tool calls.
- Optimization: SLOs, latency, throughput, cost, utilization, memory limits, availability targets.
