# Final report — torchtune LoRA SFT for Django-RCX (Qwen3-4B-Instruct-2507)

Goal: a LoRA recipe on torchtune (2× A100-40, FSDP) that matches full-SFT downstream
performance as closely as possible, picked by a principled, evidence-driven sweep following
[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). Two phases: **(1)** lock the
throughput/memory config; **(2)** sweep LR×rank by held-out val loss and emit the finalists for
full 2-epoch training. Companion notes: `THROUGHPUT_TIER1_NOTE.md`, `LORA_SWEEP_GOAL.md`.

## TL;DR
- **Throughput Tier-1:** keep `enable_activation_offloading=True` + `num_output_chunks=16`
  (the validated baseline). Turning offloading off gained only **+1.6% tok/s** but pushed peak
  reserved **29.7 → 37.6 GiB** (OOM-risky at 32k). The offload copies overlap the checkpointed
  backward, so they're ≈free — the knobs are a wash; real speedups need packing (out of scope).
- **LoRA sweep (7 runs, α=32, constant LR, 150 steps, val_loss on 128 held-out rows):**
  - **LR\* = 7e-4** with a flat optimum basin 5e-4–1e-3; `1e-4` clearly worst.
  - **Rank is not the lever:** val_loss for r∈{64,128,256} is flat within 0.002 → not
    capacity-bound (matches the capacity estimate and the blog's rank-independence).
- **Finalists for 2-epoch training (both r128, α=32):** `lr=7e-4` (sweep best) and `lr=5e-4`
  (lower-LR hedge for the longer run; also the torchtune analogue of the 19.7% reference).
  Both launched (~14 h each) → adapters → `eval/sbv.sh` (N=5) vs the full-SFT target.

## Reference (ms-swift Megatron, unswept) — the bar to match
| Run | N | pass@1 | pass@5 |
|---|---:|---:|---:|
| Base (zero-shot) | 1 | 4.2% | — |
| **Full SFT** (2ep) | 5 | **22.2 ± 4.0%** | 40.7% |
| LoRA r128 5e-4 bs8 (2ep) | 5 | 19.7 ± 3.0% | 38.1% |
| LoRA r128 1e-4 (2ep) | 5 | 13.7 ± 3.9% | 28.8% |

The reference Full-SFT LR (`1e-5`) was never swept; `5e-4` LoRA ≫ `1e-4` LoRA already hinted the
LoRA optimum sits high — the sweep confirms it (~7e-4).

## Phase 1 — Throughput Tier-1 (2× A100-40, global batch 8)
40-step probes, seed 0 (identical workload), tok/s median over steady-state:
| id | offload | chunks | tok/s/gpu | peak_active GiB | reserved GiB | verdict |
|---|---|---|---|---|---|---|
| A | on  | 16 | 1034 | 27.1 | 29.7 | baseline (== 7 h historical run) |
| B | off | 8  | 1051 (+1.6%) | 34.8 | **37.6** | OOM-risky, not worth it |

**Decision:** offload ON + chunks 16. (C: off/16 and D: on/8 ablation + a profiler
mechanism-confirmation were cut as low-value once the spread proved ≤1.6%.)

## Phase 2 — LoRA LR×rank sweep
Methodology from the blog, applied: LoRA on all linear layers (attn q,k,v,o + MLP; unembed
excluded — tied embeddings); **α=32** fixed (the α/r scaling makes optimal LR ~rank-independent,
verified in torchtune `peft/lora.py`); **constant LR** for clean comparison (stock
`get_cosine_schedule_with_warmup` with `num_cycles=0`); rank by **held-out log-loss**, not
sampling evals. Selection metric: val_loss on a 128-row strided subset of the disjoint held-out
set (caveat: that set covers 3 of 4 units — ctx_impl is fully in train).

### Stage 1 — LR @ r128
| LR | 1e-4 | 2e-4 | 5e-4 | **7e-4** | 1e-3 |
|---|---|---|---|---|---|
| val_loss@150 | 0.6455 | 0.6272 | 0.6101 | **0.6070** | 0.6101 |

U-curve bracketed (turns up at 1e-3); flat basin 5e-4–1e-3; **LR\* = 7e-4**. val_loss ordering
matches the known downstream gap (1e-4 → weak 13.7%; 5e-4 → strong 19.7%).

### Stage 2 — rank @ lr 5e-4
| rank | 64 | 128 | 256 |
|---|---|---|---|
| val_loss@150 | 0.6088 | 0.6101 | 0.6083 |
| params (M) | 132 | 264 | 529 |

Flat within 0.002 → **not capacity-bound**; r128 chosen (reference-comparable, cheaper than 256).
(r32 dropped to save GPU time; capacity estimate says it's non-binding.)

## Finalists → 2-epoch runs (the deliverable)
Both r128, α=32, cosine-to-0 with 3% warmup, full 8039-example data, global batch 8, Tier-1
memory knobs. Each ≈ 14 h on 2× A100-40.
| finalist | config file | LR | why |
|---|---|---|---|
| **F1** | `final_lora_r128_lr7e-4_2ep.yaml` | 7e-4 | sweep-best |
| **F2** | `final_lora_r128_lr5e-4_2ep.yaml` | 5e-4 | lower-LR hedge (short-run bias) + reproduces 19.7% ref on torchtune |

**Next:** both train (launched via `run_finalists.sh`), then `eval/sbv.sh` (N=5) → compare
pass@1/pass@5 to 22.2%/40.7%. Expectation (H-match): the LR-optimized LoRA closes most of the
22.2→19.7 gap; any residual likely from the unembed-LoRA omission (tied embeddings) + LoRA's
mild batch-size sensitivity.

## Eval results — pass@1 / pass@5 (sbv, N=5, 118 Django instances)

| model | pass@1 | pass@5 | per-run resolved/118 |
|---|---|---|---|
| **F1 torchtune r128 lr7e-4 2ep** | **11.2 ± 4.0%** | **28.8%** | 16, 18, 15, 11, 6 |
| **F2 torchtune r128 lr5e-4 2ep** | **7.8 ± 1.5%** | **11.9%** | 9, 7, 9, 9, 12 |
| *Reference: Full SFT (ms-megatron)* | *22.2 ± 4.0%* | *40.7%* | — |
| *Reference: LoRA r128 5e-4 (ms-megatron)* | *19.7 ± 3.0%* | *38.1%* | — |

**F1** is the better finali but falls 11 pp below full-SFT pass@1 (22.2%) and **12.5 pp below the
ms-megatron LoRA run at identical hyperparams** (F2 vs 19.7% — same r128, 5e-4, 2ep). The
torchtune-vs-megatron LoRA gap (7.8% vs 19.7%) is far larger than the throughput/kernel-fusion
story and points to a training-recipe difference, not serving.

F1's V=4 dropping to 6/118 (5.1%) is also notable and inflates the variance estimate.

### Root-cause investigation — update (code + live-log audit of both pipelines)

The four causes originally listed here were checked directly against ms-swift's actual training
logs (`train/logs/lora/django_r128_lr5e-4_2ep_qwen34i_bs8_v0.log`, the logged `MegatronSftArguments`
/ `LoraConfig`) and torchtune's masking/tokenizer/LoRA source. Two are **ruled out**, one is
**not applicable as stated**, and the investigation surfaced **four different, real, uncontrolled
hyperparameter deltas** that were never part of the "identical hyperparameters" comparison.

**Ruled out:**
- **Masking / prompt template — no mismatch.** ms-swift's `--loss_scale default` masks
  system/user and trains on the assistant span + trailing `<|im_end|>\n` on every round
  (`swift/loss_scale/base.py`); torchtune's `train_on_input=False` → `masking_strategy=
  "train_on_assistant"` does the identical thing across all rounds (`data/_messages.py:920-935`).
  The served checkpoint (`Qwen3-4B-Instruct-2507`) has **no thinking-mode branch** in its chat
  template at all (verified in the actual `tokenizer_config.json`), so ms-swift's
  `qwen3_nothinking` template is a no-op and neither pipeline ever emits `<think>` tokens — the
  rendered training strings are effectively byte-identical. mini-swe-agent never sends an
  OpenAI `tools=[...]` payload (plain THOUGHT+```bash``` regex parsing), so sglang's
  `--tool-call-parser qwen25` is inert for this eval — no tool-call-format mismatch is possible
  either.
- **unembed LoRA exclusion — not the differentiator.** ms-swift's own reference run also logs
  `modules_to_save=[]` — it does **not** train the embedding/lm-head, same as torchtune's
  `apply_lora_to_output=False`. Both sides tie in the same way.
- **LoRA scaling factor — identical convention.** ms-swift's logged `LoraConfig` shows
  `use_rslora=False`, `lora_alpha=32`, `r=128` — the same `alpha/rank` scaling torchtune uses
  (`LoRALinear.forward`: `(self.alpha/self.rank) * lora_b(...)`). No 4× effective-LR discrepancy.

**Confirmed real, previously-uncontrolled differences** (pulled from the live ms-swift log's
argument dump vs the torchtune finalist yamls — never matched because the comparison only
controlled for r/alpha/lr/epochs/global-batch):

| knob | torchtune finalists | ms-swift reference (bs8, 19.7%) |
|---|---|---|
| `weight_decay` | **0.0** | **0.1** |
| AdamW `beta2` | 0.999 (torch default, never set) | **0.95** |
| gradient clipping | **none** (`clip_grad_norm: null`) | **clip_grad=1.0** |
| `lora_dropout` | 0.0 | 0.05 |
| LR floor | cosine decays fully to **0** | cosine decays to a **10%-of-peak floor** (min_lr=5e-5) |
| truncation side | right | left (only ~1.5% of examples exceed 32k tokens — negligible) |

Training/val loss curves for both finalists are smooth and healthy (no NaNs/divergence, val_loss
in the same range as the sweep), so this isn't a crash/instability bug — the leading theory is
that **no weight decay + no gradient clipping on a 264M-param (r128) adapter**, trained at a fairly
high LR for 2 full epochs, lets the adapter drift further than the ms-swift run's more-regularized
setup, hurting generalization to held-out agentic tasks in a way that isn't visible in in-distribution
held-out loss. This has **not been confirmed by a rerun yet** — see the confirmatory experiment below.

### Confirmatory experiment (in progress)

One additional torchtune run, `final_lora_r128_lr5e-4_2ep_msmatched.yaml` (F2 config +
`weight_decay=0.1`, `optimizer.betas=[0.9, 0.95]`, `clip_grad_norm=1.0`, `model.lora_dropout=0.05`;
LR schedule left as cosine-to-0, i.e. the schedule-floor delta is *not* controlled for in this pass),
2× A100-40 FSDP, followed by `eval/sbv.sh` (N=5) served via sglang DP=2 across both GPUs
(`MS=torchtune_qwen3_lora_lr5e-4_2ep_msmatched`). If this closes most of the gap to 19.7%, the
optimizer/regularization deltas are confirmed as the primary cause; if not, the residual points to
a deeper Megatron-Core-vs-torchtune LoRA numerics difference or the LR-schedule floor.

## Expectation vs outcome
- ✅ LR is the lever, optimum high (~7e-4) — confirmed; the unswept reference's `5e-4` was near-optimal, `1e-4` far off.
- ✅ Rank not capacity-bound at r≥64 — confirmed flat.
- ❌ **Downstream match failed.** F1 11.2%/28.8%, F2 7.8%/11.9% — well below 22.2%/40.7% target and even below the unswept ms-megatron LoRA (19.7%/38.1%). The training-recipe gap (not serving, not eval harness, not masking/template — see "Root-cause investigation — update" above) is the primary suspect; a matched-hyperparameter confirmatory run is in progress.

## Engine comparison: ms-megatron (SWIFT) vs torchtune — wall-clock + why (code-grounded)
Apples-to-apples (both r128, global batch 8 = micro 1 × accum 4 × DP 2, 2 epochs, 2× A100-40,
activation-checkpointing/recompute on, flash attention, same model + data):

| | ms-megatron (SWIFT, megatron-core 0.17.1) | torchtune (67614f9) |
|---|---|---|
| 2-epoch wall | **11h 28m** | **14h 43m** (F1) |
| s/it | **21.0** (median, `train_speed(s/it)`) | ~24 (pure-train) / ~26 (incl. val+ckpt) |
| iters (2ep) | 1948 (5% val split → ~7637 train) | 2010 (full 8039) |
| peak mem | 26.6 GiB | ~27 active / ~30 reserved |

Megatron is **~22% faster wall / ~13% faster per-iter**. The wall gap is partly incidental
(torchtune trained ~5% more data + periodic val + 2 ckpt saves); the engine gap is the ~13% s/it.
**Compute FLOPs per step are identical** (same model/tokens/recompute/global-batch), so the gap is
*overhead*. Decomposed against the source of both engines:

1. **Kernel fusion — most plausible primary driver (reasoned, not yet profiled).** Megatron runs
   fused kernels: vocab-parallel CE
   ([`fusions/fused_cross_entropy.py#L136`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/fusions/fused_cross_entropy.py#L136), enabled by `--cross_entropy_loss_fusion true`),
   fused RMSNorm ([`fusions/fused_layer_norm.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/fusions/fused_layer_norm.py)),
   fused SwiGLU ([`fusions/fused_bias_swiglu.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/fusions/fused_bias_swiglu.py)),
   fused softmax ([`fusions/fused_softmax.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/fusions/fused_softmax.py)).
   torchtune runs **eager** (`compile: False` in the config →
   [`lora_finetune_distributed.py#L316`](https://github.com/SWE-Spot/swespot/blob/67614f9/train/torchtune/lora_finetune_distributed.py#L316)) with a chunked, unfused
   `LinearCrossEntropyLoss` Python loop
   ([`cross_entropy_loss.py#L87`](https://github.com/SWE-Spot/swespot/blob/67614f9/third_party/torchtune/torchtune/modules/loss/cross_entropy_loss.py#L87)).
   On this memory-bandwidth-heavy regime (norm/activation/elementwise per layer), fewer/larger
   fused kernels cut launch + HBM traffic — DL-consistent as the main contributor.

2. **FSDP gathers the frozen base vs DDP-replicate — real but minor on NVLink.** torchtune
   `shard_model` calls `fully_shard(..., reshard_after_forward=True)` on every transformer layer
   ([`_distributed.py#L661`](https://github.com/SWE-Spot/swespot/blob/67614f9/third_party/torchtune/torchtune/training/_distributed.py#L661),
   wired from [recipe `#L548`](https://github.com/SWE-Spot/swespot/blob/67614f9/train/torchtune/lora_finetune_distributed.py#L548)),
   which shards **all** params incl. the frozen base and re-gathers them in forward *and* backward.
   Megatron (TP1/PP1/DP2) keeps full params per rank in DDP
   ([`distributed_data_parallel.py#L23`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/distributed/distributed_data_parallel.py#L23))
   and the distributed optimizer all-gathers only the *trainable* (LoRA-sized) params
   ([`distrib_optimizer.py#L2686`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.17.1/megatron/core/optimizer/distrib_optimizer.py#L2686)).
   So torchtune moves ~8 GB (the base) ×2/step that megatron doesn't. **But** on A100 NVLink
   (~hundreds of GB/s) that's tens of ms vs a ~24 s step, and FSDP2 prefetches/overlaps it — so it's
   a small contributor here (it would *dominate* on PCIe/Ethernet). Note both engines overlap their
   comm (FSDP2 prefetch ↔ megatron `overlap_grad_reduce`/`overlap_param_gather`), so overlap itself
   is **not** the differentiator — only the comm *volume* is, and that volume is cheap on NVLink.

3. **Activation offloading — measured ~1.6%.** torchtune keeps `enable_activation_offloading=True`
   (Tier-1 OOM-safety); `OffloadActivations` overlaps CPU↔GPU copies on a side CUDA stream
   ([`_activation_offloading.py#L24`](https://github.com/SWE-Spot/swespot/blob/67614f9/third_party/torchtune/torchtune/training/_activation_offloading.py#L24)),
   which is why Tier-1 measured only +1.6% from turning it off. Megatron doesn't offload (recompute only).

4. **CE FLOPs & recompute ≈ a wash.** torchtune's LinearCE projects **only non-masked (assistant)
   tokens** ([`cross_entropy_loss.py#L104`](https://github.com/SWE-Spot/swespot/blob/67614f9/third_party/torchtune/torchtune/modules/loss/cross_entropy_loss.py#L104)) — a FLOP saving vs megatron projecting all
   positions — partly offsetting megatron's fusion advantage on the CE itself. Both do full-layer
   recompute, so that cancels.

**Correction vs my first take:** I had glibly credited "fused kernels + comm overlap." Reading the
source shows comm *overlap* is not the lever (both overlap), and a bandwidth estimate shows the FSDP
base-gather is cheap on NVLink — so the honest story is **fusion-dominated, offloading ~1.6%
(measured), FSDP-gather minor-on-NVLink, CE/recompute a wash**. Magnitudes beyond the offloading
1.6% are *reasoned, not profiled*: a torchtune profiler run (deferred — GPUs busy with the finalists)
would give the exact split, and is the clean way to confirm fusion as the primary driver.

## Eval & serving-throughput (post-training plan)
The 2-ep adapters are evaluated with `eval/sbv.sh` (118 SWE-bench-verified Django instances,
multi-turn agent up to 64 steps, 5 runs V=0..4 for variance). Generation is served by **sglang**
(`/home/alex/serving`, sglang 0.5.9 — vllm is not installed there); a single V run took ~2 h, so
before committing to 5×2 runs we characterize how much generation throughput is left on the table.
Hard constraints: keep `eval_results/` intact (new unique `MS` name per finalist → new subdir, never
overwrite); do not modify the serving `.venv`. Procedure recorded in memory + `LORA_SWEEP_GOAL.md`.

**Throughput experiments (a FEW, via `sglang.bench_serving`, not full sbv runs):**
1. Concurrency sweep {1,4,8,16,32} at **DP=2** vs **DP=1** → output tok/s, TTFT, p50 latency; find
   the per-replica saturation point and whether WORKERS=20 saturates DP=2.
2. **DP=2 sequential vs DP=1 ×2 parallel** (2 servers, 1 GPU each, 2 sbv runs at once) — inferred
   from the per-replica saturation curve (does DP=2 carry cross-replica overhead?).
3. Prefix/radix cache on/off and `mem-fraction-static` 0.80/0.90 (agent reuses the long
   system+repo prefix across 64 steps → potentially large KV/prefix-cache win).
4. Host CPU/mem under WORKERS=20 (gen) + 24 (scoring) docker fleets — is it CPU-bound, not GPU?

**Results** (`sglang.bench_serving`, base Qwen3-4B, random in≈8k/out≈400 single-turn proxy, radix
cache on by default; total tok/s = in+out aggregate across the server):

| concurrency | 1 | 8 | 16 | 32 |
|---|---|---|---|---|
| **DP=2** total tok/s | 2088 | 10836 | 15332 | **17490** |
| DP=2 TTFT p50 / E2E p50 | 0.46s / 4.0s | 0.63s / 6.2s | 0.85s / 8.8s | 2.94s / 15.2s |
| **DP=1** total tok/s | 2156 | 7672 | 8879 | **9210** |
| DP=1 TTFT p50 / E2E p50 | 0.41s / 4.0s | 1.23s / 9.0s | 3.13s / 15.2s | 7.73s / 20.0s |

**Findings:**
- **DP=2 saturates ~concurrency 16–24** (~17.5k tok/s); beyond that throughput rises only ~14%
  while TTFT/E2E blow up. **WORKERS=20 sits right at that knee** — a single DP=2 server is already
  near-optimally loaded; raising workers buys little.
- **DP=1 saturates earlier (~c8–16, ~9.2k).** Two independent DP=1 replicas ≈ **18.4k vs DP=2's
  17.5k (~5% more)** — DP=2's round-robin router carries slight overhead vs fully-independent replicas.
- **Host: 24 CPU / 167 GiB.** Generation (20 agent containers) is LLM-wait-bound → CPU has slack,
  GPU is the gen bottleneck. Scoring (24 swebench containers) is CPU-bound → GPU idle during scoring.

**Recommendation for the 5-run × 2-finalist eval:**
- **Parallelize across the two GPUs — one finalist per GPU:** two DP=1 servers (`:8002` GPU0,
  `:8003` GPU1), eval F1 and F2 concurrently at ~WORKERS 12–16 each (near DP=1 saturation). This
  runs both finalists in ≈ the wall-time of one (~2× total speedup) at ~5% better GPU util than DP=2.
- **Scoring is CPU-bound** (24 containers ≈ 24 cores) → don't run two scoring phases at once; stagger,
  or decouple generation (GPU) from scoring (CPU) so GPUs keep generating the next version while
  CPUs score the previous.
- **Caveat — GPU is not the whole ~2h.** A run is multi-turn agent rollouts (64 steps × 118
  instances, sequential per instance, tool-exec between steps) + SWE-bench scoring; generation and
  scoring currently run serially per version, each leaving the other resource idle. So DP tuning
  helps the generation slice, but there is an orchestration/tool-exec/scoring floor GPU throughput
  can't cut — the real wall-clock lever is **parallelism (across GPUs) + gen/score decoupling**, not
  squeezing tok/s. Also, the random-token bench has *no shared prefix*, so it is a conservative
  floor: the real agent workload reuses a long system+repo prefix across 64 steps, which radix/prefix
  caching (on by default) exploits — real generation throughput is higher than these numbers.

**Eval runs:** F1 (`MS=torchtune_qwen3_lora_lr7e-4_2ep`) and F2 (`MS=...lr5e-4_2ep`), 5 versions
each, with the best serving config. Results: F1 11.2%/28.8%, F2 7.8%/11.9% — see eval results
section above for analysis of the gap vs 22.2%/40.7% target.

## Reproducibility
- Sweep: `sweep_lora.yaml` + `run_sweep.sh` (7 runs, one detached driver). Analysis: `analyze.py`.
- Finalists: `final_lora_r128_lr{7e-4,5e-4}_2ep.yaml` (from `final_lora_2ep_TEMPLATE.yaml`).
- Env: `swespot/.venv` (torch 2.12.0+cu130, torchtune @ 67614f9, torchao 0.17.0). wandb project
  `swespot_torchtune`. Data: `data/torchtune/django_rcx_{8039,val,val_128}.jsonl` (seed-0 split).
