# Goal — Replicate the Django-RCX LoRA SFT on **torchtune** (single-GPU first)

Reproduce, on torchtune, the *same* Qwen3-4B-Instruct-2507 LoRA SFT that ran on the SkyRL **Tinker**
server (`lora_django_trial.py`) and earlier on **ms-swift Megatron** — but on a **single 40 GiB A100**, and
validate the hypothesis (report Part 2D) that torchtune's `LinearCrossEntropyLoss` makes the 32k step fit
on one GPU with **no TP/CP** — the thing SkyRL-Megatron (`ChunkedDistributedLogprob`) and SkyRL-FSDP could
not do.

## Why this should work (the bet)

torchtune's recommended SFT loss
[`LinearCrossEntropyLoss`](https://github.com/pytorch/torchtune/blob/bd2a0fc7/torchtune/modules/loss/cross_entropy_loss.py#L19)
**never materializes the full `[1, 32768, 151936]` logits**: the model skips its output projection
(`skip_output_layer=True` → `unembed()` returns hidden `[1, s, 2560]`,
[transformer.py#L681-L682](https://github.com/pytorch/torchtune/blob/bd2a0fc7/torchtune/modules/transformer.py#L681-L682)),
and the loss masks ignored tokens *before* the projection, then projects+CE the kept tokens in chunks.
See Part 2D of `megatron_vs_skyrl_logits_report.md`.

## Parity targets (must match `lora_django_trial.py` / Tinker)

| Knob | Tinker trial | torchtune mapping |
|---|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` | custom builder (rope_base=5e6 — **not** the stock 1e6) |
| Data | `swespot/sft-v0`, 4 Django RCX units, cap 2048/unit (shuffle seed 0), concat → **8039** ex | pre-built local jsonl, byte-identical selection |
| LoRA | rank 128, "mlp+attn+unembed" | rank 128 on attn(q,k,v,output_proj)+mlp; output is **tied** (no LoRA on unembed — documented gap) |
| max_length | 32768 | tokenizer `max_seq_len=32768`, truncation right |
| Loss mask | `ALL_ASSISTANT_MESSAGES` | `masking_strategy=train_on_assistant`, ignore_index −100 |
| LR | `get_lr(...,is_lora=True)` ≈ **4.9e-4**, **linear** schedule | `optimizer.lr=4.9e-4`, linear LR schedule w/ warmup |
| Global batch | 8 examples/step | `batch_size=1` × `gradient_accumulation_steps=8` (micro-batch=1, like Tinker's per-seq fwd/bwd) |
| Epochs | 1 (paper used 2) | `epochs=1` |

## Phases

- **P0 — Setup.** ✅ done. model on disk (HF cache snapshot); arch verified vs builder (rope_theta 1e6→5e6 → custom builder); **env resolved shim-free** (see "How to run" below).
- **P1 — Data.** Replicate `DjangoRcxBuilder` exactly → `train/data/django_rcx_8039.jsonl`. Verify counts.
- **P2 — Recipe/config.** Custom builder `train/qwen3_2507_builder.py` (rope_base=5e6); copy torchtune `lora_finetune_single_device.py` → `train/`; write `train/qwen3_4b_django_lora.yaml`.
- **P3 — Render parity.** Tokenize K sample convos with Tinker renderer vs torchtune qwen3; diff token ids + loss mask. Resolve or document.
- **P4 — Smoke.** `max_steps=2`, small seq → loss decreases, checkpoint writes, wandb logs.
- **P5 — Memory.** micro-batch=1 at `max_seq_len` ∈ {8k,16k,32k}; record peak GiB (`log_peak_memory_stats`). **Confirm 32k fits one 40 GiB GPU.**
- **P6 — Throughput.** Sweep `num_output_chunks`, `compile`, `enable_activation_offloading` (packed noted separately, breaks per-example parity). Record tokens/s; pick best parity-preserving config.
- **P7 — Full run.** 1 epoch (~1005 steps) with wandb; save adapter.
- **P8 — Document.** Fill Part 2D measured peak + throughput; refresh secret gist.

## Acceptance criteria

- **V1 — Data:** jsonl = 8039 rows; per-unit = {software_design 2048, ctx_impl 1895, evo_replay 2048, rt_alignment 2048}.
- **V2 — Render parity:** assistant-content token ids match Tinker; loss mask aligns. Differences documented & understood.
- **V3 — Memory (headline):** 32k micro-batch=1 step completes on ONE 40 GiB A100, peak < 40 GiB (no OOM, no TP/CP). Contrast: SkyRL-Megatron OOM'd at TP=1 *and* TP=2.
- **V4 — Loss sane:** decreasing train NLL, same ballpark as Tinker (~0.7–0.8 early).
- **V5 — Throughput:** tokens/s measured; best config identified.
- **V6 — End-to-end:** 1-epoch run finishes; adapter checkpoint saved; wandb run present.

## Notes / known gaps to watch

- **rope_base:** 2507 uses 5e6 (config.json), torchtune builder hardcodes 1e6 → custom builder.
- **Tied embeddings:** `apply_lora_to_output` is illegal with `tie_word_embeddings=True`; Tinker's "unembed" LoRA can't be mirrored exactly. Substantive LoRA (attn+mlp r128) is matched.
- **torchtune source:** local checkout `bd2a0fc7` (has Qwen3 + LinearCE); PyPI `0.6.0` likely lacks both — run against the checkout.
- **Packing:** big throughput win but changes batching/masking vs Tinker's one-example-per-seq → keep `packed=False` for parity; measure packed separately.

## How to run (resolved, shim-free)

Env: `.venv-nightly` (torch 2.12.1+cu130, torchvision 0.27.1+cu130, torchao 0.17.0) with torchtune
installed **editable from the git worktree** `/home/alex/swespot/third_party/torchtune-compat` (commit
`41aaca46` — the last commit before torchtune adapted to a post-0.17 torchao nightly; matches stable
torchao 0.17, so **no shim/patch**). Main torchtune checkout stays at HEAD.

```bash
cd /home/alex/SkyRL/.claude/worktrees/torchtune-sft
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_API_KEY=<key> PYTHONPATH=$(pwd)/train \
/home/alex/swespot/third_party/torchtune/.venv-nightly/bin/python \
  train/lora_finetune_single_device.py --config train/qwen3_4b_django_lora.yaml
# data first (once): .venv-nightly/bin/python train/prepare_django_rcx_data.py  (uses SkyRL .venv too; any env with `datasets`)
```

## Results

- **V1 data** ✅ 8039 rows; per-unit {software_design 2048, ctx_impl 1895, evo_replay 2048, rt_alignment 2048}.
- **V2 render parity** ✅ token IDs byte-identical to tinker's qwen3 renderer; loss-mask minor diff (torchtune also trains the assistant header `<|im_start|>assistant\n` + trailing `\n`) — documented.
- **V3 memory (headline)** ✅ *with the relief valve.* Plain LinearCE at 32k is marginal — a real 1-epoch single-GPU run **OOM'd** (>40 GiB) on long high-assistant-token examples (the 50%-masked synthetic probe's 38 GiB under-counted). With **`enable_activation_offloading=True` + 16 CE chunks**, a true 32768-token step peaks at **30.4 GiB active / 33.4 GiB reserved, no OOM** on one 40 GiB A100, no TP/CP. Contrast: SkyRL-Megatron OOM'd at TP=1 *and* TP=2.
- **V4 loss** ✅ decreasing (smoke 1.47→1.10; full run starts 1.22).
- **V5 throughput** ~1300–2100 tok/s (unpacked, parity); compile counterproductive for variable-length; packing is the speed lever but breaks per-example parity.
- **V6 full run** ✅ via **2 GPUs (FSDP)**. Single-GPU (no offloading) OOM'd ~step 15; the fixed **2-GPU run completed 1 epoch** (1005 steps, ~25 s/step, ~7 h, both GPUs ~22–25 GiB), **final loss 0.62**, unmerged LoRA adapter at `train/outputs/torchtune/qwen3_4b_django_lora_2gpu/epoch_0/adapter_model.safetensors`. 1-GPU run command: `qwen3_4b_django_lora.yaml` (now also has offloading on, so it would fit if rerun). 2-GPU: `torchrun --nproc_per_node=2 train/lora_finetune_distributed.py --config train/qwen3_4b_django_lora_2gpu.yaml`.

