# Goal — leave-one-out ablation: which of the 4 matched hyperparameters actually matter?

## Context

Phase 3b (`FINAL_REPORT.md`) confirmed that matching 4 optimizer/regularization hyperparameters
(`weight_decay`, AdamW `beta2`, `clip_grad_norm`, `lora_dropout`) to ms-swift's values took
torchtune's LoRA recipe from 7.8% to 20.0% pass@1, closing the gap to the ms-swift reference
(19.7%) almost exactly. We changed all 4 at once, so we don't know each knob's individual
contribution. This ablation isolates each one via leave-one-out (LOO): each new run = the winning
`final_lora_r128_lr5e-4_2ep_msmatched_clean.yaml` config with exactly one knob reverted to
torchtune's default, others held matched.

**Decision: start with the 2 highest-hypothesized-impact knobs first** — `weight_decay` and
`clip_grad_norm` (~2.4 days) — and decide whether to run `beta2`/`lora_dropout` as a follow-up round
based on those results, rather than committing to all 4 (~4.7 days) up front.

## 1. Pre-registered hypotheses (state predictions *before* running, don't rationalize after)

| Arm | Reverts | Hypothesis & mechanism | Predicted effect on pass@1 if reverted | Confidence |
|---|---|---|---|---|
| **LOO-wd0** | `weight_decay` 0.1→0.0 | Weight decay pulls the LoRA A/B matrices back toward their near-zero init every step, capping how far the adapter drifts from the base model. The downstream eval is genuinely OOD relative to training (different SWE-bench tasks, not held-out same-distribution rows), so this is the one knob whose job is explicitly "generalize past the training distribution." | **Largest drop, ~3-6pp** | Medium-high |
| **LOO-noclip** | `clip_grad_norm` 1.0→null | Clipping caps rare large-gradient-norm steps (long/unusual sequences in this 32k-context agentic dataset, bf16 noise) from producing a destabilizing update. At LR=5e-4 over 2010 steps this is a real risk; a few bad updates late in training could disproportionately hurt a model evaluated via multi-turn agentic rollout (errors compound). | **Large drop, ~2-5pp**, possibly with higher across-version variance | Medium-high |
| LOO-dropout0 (round 2) | `lora_dropout` 0.05→0.0 | Dropout here only regularizes the *input to the LoRA path* (`self.lora_a(self.dropout(x))`), not the frozen base — a narrow, gentle intervention on an already-small (264M/4B) adapter. | **Small drop, ~0-2pp** — plausibly indistinguishable from noise at N=5 | Low-medium |
| LOO-beta2default (round 2) | `betas` [0.9,0.95]→[0.9,0.999] | beta2 sets AdamW's 2nd-moment EMA window: 0.95≈20 steps, 0.999≈1000 steps. beta2=0.95 is standard for *long* LLM pretraining (loss landscape is genuinely non-stationary over many-thousand-step runs). Our run is only 2010 steps — 0.999's 1000-step window is already comparable to the *whole run*, so its usual "staleness" downside is muted here, while 0.95's short window may just be noisier at this horizon. **No confident directional prior** — could plausibly show near-zero or even a slightly *positive* effect if reverted. | **Smallest/most uncertain effect**, most likely arm to come back inconclusive | Low |

**Predicted ranking (largest → smallest expected effect): weight_decay ≳ clip_grad_norm > lora_dropout ≳ beta2**, with beta2 flagged as uncertain in *direction*, not just magnitude.

## 2. Explicit hyperparameters (no implicit defaults)

Every new config states all 4 knobs explicitly, including the one being "reverted to default" —
e.g. `betas: [0.9, 0.999]` written out with a comment noting it's AdamW's default, never an absent
line relying on the framework to fill it in silently. Verified `final_lora_r128_lr5e-4_2ep.yaml`
(the original all-default F2 run) has no `betas:` line at all — so `[0.9, 0.999]` is confirmed as
the actual value that was already tested, not just a documented assumption.

## 3. Execution strategy — parallel single-device, gated by a memory probe

**Verified feasible** (read `lora_finetune_single_device.py` directly): it uses the exact same
generic `config.instantiate` pattern for `model`/`optimizer`/`loss`/`lr_scheduler` as the
distributed recipe, supports `clip_grad_norm` and activation checkpointing/offloading identically,
has **no distributed-process-group init at all** (plain `python` script, no torchrun), and needs
only one substantive config change: `gradient_accumulation_steps: 4→8` to preserve global batch=8
with 1 GPU instead of 2 (batch_size=1 × accum=8 × world_size=1 = 8). It does **not** support
`dataset_val`/`run_val_every_n_steps` (grep found zero references) — drop those fields, no loss
since we've already shown val_loss doesn't discriminate these knobs anyway. Same epochs_run
increment-order pattern exists here too (`self.epochs_run += 1` after the loop) — the same
"don't trust `resume_from_checkpoint: True` without checking `recipe_state.pt` directly" caution
applies if a run needs resuming.

**The honest catch, found while costing this out:** parallelizing *training* alone barely reduces
total wall-clock. Single-GPU training has no data-parallel split, so it runs the full 8039-example
epoch on one GPU instead of splitting it across two — roughly **2x slower per arm**. Running 2 arms
in parallel at 2x the per-arm time is a wash against running them sequentially at 1x each on 2 GPUs.
The real lever is *also* parallelizing eval (currently ~13h14m per arm split 3-versions/2-versions
across both GPUs) into single-GPU-per-arm (~21.5h per arm, all 5 versions sequential on that arm's
own GPU, run simultaneously with the other arm's eval) — that's genuinely new code (not a reuse of
the proven split-eval pattern), for a **~9% total time reduction** (≈10 hours off a multi-day run).

**Decision:** do the parallel-single-device training (cheap to set up), but reuse the *existing,
already-proven* 2-GPU-split eval phase sequentially per arm afterward rather than building a new
single-GPU eval loop — take the "config is trivial" win on the training side without taking on new,
unvalidated eval code for a marginal further gain.

**Memory is the one real unknown — gate on a quick probe before committing.** No single-device run
of this model has been done before (an earlier attempt in `TORCHTUNE_SFT_GOAL.md` was abandoned for
the 2-GPU FSDP approach, reason not recorded). The 2-GPU FSDP run reserved ~27-30GiB/GPU *while
sharding* the frozen base model; single-device must hold the full unsharded ~8GiB base weights the
whole time, so reserved memory could plausibly climb to ~33-34GiB on a 40GiB A100 — likely fits, but
tight enough to verify rather than assume. **First step: run a 40-step probe** (mirroring this
project's own Tier-1 throughput-probe methodology) on one GPU before launching the real ablation.

- `train/torchtune/probe_single_device.yaml`: copy of `msmatched_clean.yaml`, minus
  `dataset_val`/`run_val_every_n_steps`, `gradient_accumulation_steps: 8`, `max_steps_per_epoch: 40`,
  `epochs: 1`, throwaway `output_dir`.
- Run: `CUDA_VISIBLE_DEVICES=0 python train/torchtune/lora_finetune_single_device.py --config train/torchtune/probe_single_device.yaml` (~15-20 min).
- **Gate:** if it completes without OOM and `GPU peak memory reserved` has reasonable headroom (comfortably under 40GiB) → proceed with the parallel-single-device plan below. If OOM or too tight → fall back to the proven sequential 2-GPU-FSDP pattern (byte-identical hypotheses/knob-diffs, just executed via `torchrun --nproc_per_node=2` + `lora_finetune_distributed.py` looped one arm at a time, exactly mirroring `run_msmatched_clean.sh`'s existing 4-phase structure).

**Probe result (2026-07-28): PASSED.** 40/40 steps completed, no OOM, no errors. `GPU peak memory
reserved: 34.10 GiB` / 40 GiB (85% utilized — real but workable headroom; this is a random-shuffle
40-step sample, so a longer sequence later in the full 2010-step run could push this higher, but
activation checkpointing caps most of the growth). Average **38.3s/step** — 2010 steps ≈ **21.4h**
projected training time per arm (better than the ~30h worst-case estimate in §3, since the observed
single-GPU slowdown vs. the 2-GPU run is closer to ~1.4x than 2x). **Decision: proceed with
parallel single-device for round 1.** If a later OOM does occur mid-ablation on a longer sequence,
treat it as that arm's failure per the driver's continue-on-failure design and re-run that one arm
via the FSDP fallback rather than treating it as a plan failure.

## 4. Epoch scale — keep the full 2 epochs, don't truncate to 1

Considered and rejected truncating each arm to 1 epoch (~26% cheaper: saves ~7.5h training per arm,
but eval — the other ~13h14m — doesn't shrink, so the saving is smaller than it sounds). Two
independent reasons to reject it:

1. **This project already has a precedented short-run bias for a closely related case.** The
   original LR sweep (`LORA_SWEEP_GOAL.md`) explicitly found short runs bias the *apparent* optimum
   and corrected for it by hedging one LR notch down for the full run. The 4 knobs here are subject
   to the same category of risk, mechanistically: weight_decay's cumulative pull, clip_grad's
   cumulative spike-prevention opportunity, and beta2's EMA-window-vs-run-length ratio *all*
   plausibly compound or shift specifically between epoch 1 and epoch 2 — training half as long
   doesn't just add noise, it risks measuring a materially different regime than the one we actually
   deployed and care about (the 2-epoch recipe).
2. We already know val_loss doesn't discriminate these knobs (checked directly: F2 vs
   `msmatched_clean`'s early val_loss at steps 500/1000 are nearly identical — 0.567/0.537 vs
   0.567/0.537 — despite a 3x downstream difference). So a 1-epoch arm still needs the *full*
   downstream eval to get any signal at all, meaning epoch-truncation only ever saves the training
   half of the cost, not the eval half — a smaller win than "half the epochs" suggests, for a real
   representativeness risk.

**Decision: full 2 epochs for every arm**, matching the deployed recipe exactly.

## 5. Concrete config plan

Base: `train/torchtune/final_lora_r128_lr5e-4_2ep_msmatched_clean.yaml`. New files (all in
`train/torchtune/`), each = that file with `dataset_val`/`run_val_every_n_steps` removed,
`gradient_accumulation_steps: 8`, unique `output_dir`/`metric_logger.name`, plus exactly one knob
reverted (explicitly, per §2):

| File | Reverted field (explicit value) | Round |
|---|---|---|
| `final_lora_r128_lr5e-4_2ep_loo_wd0.yaml` | `weight_decay: 0.0` | **1 — run now** |
| `final_lora_r128_lr5e-4_2ep_loo_noclip.yaml` | `clip_grad_norm: null` | **1 — run now** |
| `final_lora_r128_lr5e-4_2ep_loo_dropout0.yaml` | `lora_dropout: 0.0` | 2 — deferred, decide after round 1 |
| `final_lora_r128_lr5e-4_2ep_loo_beta2default.yaml` | `betas: [0.9, 0.999]  # AdamW default, explicit` | 2 — deferred, decide after round 1 |

Write all 4 config files now (cheap, mechanical, and useful documentation regardless of whether
round 2 runs), but only **launch training for the round-1 pair**.

`eval_results/sbv/django_e13b714/` MS names (verified no collision with existing dirs):
`torchtune_qwen3_lora_lr5e-4_2ep_loo_wd0`, `..._loo_noclip`, `..._loo_dropout0`, `..._loo_beta2default`.

## 6. Concrete driver script plan

1. `train/torchtune/probe_single_device.yaml` + a one-line launch command (§3) — run and inspect manually first, this is the go/no-go gate.
2. `train/torchtune/run_loo_ablation.sh` — single detached driver for **round 1 only** (`loo_wd0` on GPU0, `loo_noclip` on GPU1), each launched as plain `CUDA_VISIBLE_DEVICES=N python lora_finetune_single_device.py --config ... &`, waited on with `wait`. Sanity check per arm (no val_loss lines available on this recipe): confirm the log reached step 2010 and the `epoch_1` checkpoint dir exists; hard-fail (skip to eval for whatever succeeded, log the failure) rather than aborting the whole script, matching the continue-on-failure design already used for `run_finalists.sh`. After both arms finish training, evaluate them **sequentially**, reusing `run_msmatched_clean.sh`'s exact proven eval phase (two DP=1 sglang servers, N=5 split 3/2 across GPUs, `eval/sbv.sh`'s existing `flock` handles scoring serialization automatically — no new eval code). Per-arm `results.tsv` row, per-arm log subdirectory. Launched via one backgrounded Bash call, no nohup/disown wrapper. Structure the arm list as a small array/loop (not hardcoded to exactly 2) so extending to round 2 later is a one-line addition, not a rewrite.
3. Fallback variant (only if the probe shows OOM): swap step 2 for the sequential-2-GPU-FSDP form of the same configs (train/serve/eval one config at a time via `torchrun --nproc_per_node=2`, identical structure to `run_msmatched_clean.sh` looped over an `ARMS` array like `run_finalists.sh`'s `CONFIGS` convention) — same hypotheses, same eval methodology, just slower.
4. Round 2 (`loo_dropout0` + `loo_beta2default`) reuses the exact same script/pattern with the arm list extended — a follow-up decision after seeing round-1 results, not built now.

## 7. Cost summary

| Path | Round 1 (2 arms: wd0 + noclip) | All 4 (if round 2 follows) |
|---|---|---|
| **Recommended: parallel single-device train + sequential proven eval** | ~30h train (parallel) + 26h28m eval (2×13h14m sequential) ≈ **56h28m ≈ 2.4 days** | ≈113h ≈ 4.7 days total (round 2 adds another ~2.4 days) |
| Sequential 2-GPU FSDP (fallback, fully proven already) | 28h23m × 2 ≈ **56h46m ≈ 2.4 days** | ≈113h32m ≈ 4.7 days total |

Round 1 (this plan's actual scope) costs **~2.4 days** either way — the parallel-single-device path
doesn't beat sequential FSDP on raw speed (see §3's honest catch), but validates the cheaper
approach for future use. Round 2 (`beta2`, `lora_dropout`) is a separate, later decision — go/no-go
based on how informative round 1's results turn out to be, not committed to here.

## 8. Statistical honesty

Recomputed per-run pstdev directly from `FINAL_REPORT.md`'s resolved-instance counts: F2 1.36pp,
`msmatched_clean` 0.86pp, confounded run 2.54pp, F1 3.61pp (V4 outlier-inflated). At N=5, the
minimum reliably-detectable effect (80% power) is **~1.5-2.5pp** under good/moderate noise, worse
under high noise. The total 4-knob effect is 12.2pp; if contributions are uneven (as hypothesized in
§1), 1-2 arms — most likely `lora_dropout` and/or `beta2` — may come back statistically
indistinguishable from the `msmatched_clean` baseline even with perfect execution. **Pre-register
this as an expected, acceptable outcome**, not a design failure. Cheap fix if an arm is borderline:
its checkpoint already exists, so extend N post-hoc (extra eval versions, ~4.3h each) rather than
retraining.

## Verification

- Probe: confirm `train.log` shows step 40/40 with no CUDA OOM traceback, and note `GPU peak memory reserved`.
- Each arm post-training: `epoch_1/adapter_model.safetensors` exists; log shows no Python traceback; step count reached 2010 (grep the final `2|2010|Loss:` line).
- Each arm post-eval: all 5 `eval_results/.../v{0..4}/report.log` files exist with non-empty `resolved_instances` (matches the "verify evidence" lesson — don't trust driver rc=0 alone, given the earlier docker-race incident).
- Final: compute pass@1 mean/pstdev per arm (same script pattern used for `msmatched_clean`), compare each against its §1 pre-registered prediction, update `FINAL_REPORT.md` with a "Phase 4 — LOO ablation" section including the hypothesis-vs-outcome table.
