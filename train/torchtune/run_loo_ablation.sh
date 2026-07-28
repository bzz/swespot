#!/bin/bash
# LOO ablation driver — round 1 (weight_decay + clip_grad_norm), per LOO_ABLATION_GOAL.md.
# Phase 1: train both round-1 arms in PARALLEL, one GPU each, via the single-device recipe
# (verified feasible + memory-safe by probe_single_device.yaml: 34.1GiB/40GiB peak, no OOM).
# Phase 2: evaluate each arm SEQUENTIALLY, reusing run_msmatched_clean.sh's exact proven eval
# phase unchanged (two DP=1 sglang servers, N=5 split 3/2 across GPUs; eval/sbv.sh's flock
# handles scoring serialization automatically). Continue-on-failure: one arm's failure doesn't
# abort the other. Round 2 (loo_dropout0 + loo_beta2default) is a separate future decision —
# extend the ARMS array below and re-run when ready.
set -u
VENV=/home/alex/swespot/.venv/bin
SVENV=/home/alex/serving/.venv/bin/python3
RECIPE=/home/alex/swespot/train/torchtune/lora_finetune_single_device.py
DIR=/home/alex/swespot/train/torchtune
BASE=Qwen/Qwen3-4B-Instruct-2507
ROOT=/tmp/tt_loo_ablation
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
RESULTS=$ROOT/results.tsv
[ -f "$RESULTS" ] || echo -e "arm\tphase\trc\tend" > "$RESULTS"

log() { echo "[$(date +%m-%d_%H:%M:%S)] $1" | tee -a "$ROOT/driver.log"; }
gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')" -lt 1000 ]; do sleep 5; done
}

# Round 1 arms: "CFG MS GPU"
ARMS=(
  "final_lora_r128_lr5e-4_2ep_loo_wd0     torchtune_qwen3_lora_lr5e-4_2ep_loo_wd0     0"
  "final_lora_r128_lr5e-4_2ep_loo_noclip  torchtune_qwen3_lora_lr5e-4_2ep_loo_noclip  1"
)

# ---- Phase 1: train all arms in parallel (one GPU each, single-device recipe) ----
gpu_free
log "TRAIN START — ${#ARMS[@]} arms in parallel"
declare -A TPID
for arm in "${ARMS[@]}"; do
  set -- $arm; CFG=$1; MS=$2; GPU=$3
  mkdir -p "$LOGDIR/$CFG"
  (
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$DIR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    WANDB_PROJECT=swespot_torchtune \
    "$VENV/python" "$RECIPE" --config "$DIR/$CFG.yaml" \
    > "$LOGDIR/$CFG/train.log" 2>&1
    echo $? > "$LOGDIR/$CFG/train.rc"
  ) &
  TPID[$CFG]=$!
  log "  [$CFG] launched on GPU$GPU (pid ${TPID[$CFG]})"
done
for arm in "${ARMS[@]}"; do
  set -- $arm; CFG=$1
  wait "${TPID[$CFG]}"
done
log "TRAIN — all arms' processes exited"

# ---- Sanity gate + disk hygiene per arm ----
GOOD_ARMS=()
for arm in "${ARMS[@]}"; do
  set -- $arm; CFG=$1; MS=$2; GPU=$3
  rc=$(cat "$LOGDIR/$CFG/train.rc" 2>/dev/null || echo 1)
  echo -e "$CFG\ttrain\t$rc\t$(date +%m-%d_%H:%M:%S)" >> "$RESULTS"
  CKPT="$DIR/outputs/$CFG/epoch_1"
  reached=$(tr '\r' '\n' < "$LOGDIR/$CFG/train.log" | grep -c "^2|2010|Loss")
  if [ "$rc" != "0" ] || [ ! -d "$CKPT" ] || [ "$reached" -lt 1 ]; then
    log "  [$CFG] TRAIN FAILED (rc=$rc, checkpoint=$([ -d "$CKPT" ] && echo ok || echo missing), reached_2010=$reached) — skipping eval for this arm"
    continue
  fi
  log "  [$CFG] TRAIN OK — reached step 2010, checkpoint saved at $CKPT"
  rm -rf "$DIR/outputs/$CFG/epoch_0"
  GOOD_ARMS+=("$arm")
done

if [ ${#GOOD_ARMS[@]} -eq 0 ]; then
  log "NO ARMS SURVIVED TRAINING — aborting before eval"
  exit 1
fi

# ---- Phase 2: eval each surviving arm SEQUENTIALLY, reusing the proven 2-GPU-split pattern ----
launch_server() { # cfg gpu port ckpt -> echoes pid
  CUDA_VISIBLE_DEVICES=$2 $SVENV -m sglang.launch_server \
    --model-path "$BASE" --served-model-name qwen34i \
    --tensor-parallel-size 1 --data-parallel-size 1 --context-length 48000 \
    --max-loras-per-batch 2 --lora-paths "{\"lora_name\":\"django\",\"lora_path\":\"$4\",\"pinned\":true}" \
    --max-lora-rank 128 --lora-target-modules all --tool-call-parser qwen25 \
    --host 0.0.0.0 --port "$3" --api-key swespot > "$LOGDIR/$1/server_gpu$2.log" 2>&1 &
  echo $!
}
ready() { until curl -s "http://localhost:$1/health" -H "Authorization: Bearer swespot" >/dev/null 2>&1; do sleep 5; done; }
test_gen() {
  curl -s "http://localhost:$1/v1/chat/completions" -H "Authorization: Bearer swespot" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$BASE:django\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK\"}],\"max_tokens\":8,\"temperature\":0}"
}
cd /home/alex/swespot
run_versions() { # config versions...
  local cfg=$1; shift
  for V in "$@"; do
    log "    EVAL $MS V=$V ($cfg) START"
    RUN_EVAL=true VERSION=$V WORKERS=14 MS=$MS \
      MODEL="openai/$BASE:django" CONFIG=eval/$cfg REPO=django HASH=e13b714 \
      bash eval/sbv.sh > "$LOGDIR/$ARM_CFG/eval_v${V}.log" 2>&1
    rc=$?
    echo -e "$ARM_CFG\teval_v$V\t$rc\t$(date +%m-%d_%H:%M:%S)" >> "$RESULTS"
    log "    EVAL $MS V=$V END rc=$rc"
  done
}

for arm in "${GOOD_ARMS[@]}"; do
  set -- $arm; ARM_CFG=$1; MS=$2; ARM_GPU=$3
  CKPT="$DIR/outputs/$ARM_CFG/epoch_1"
  gpu_free
  log "[$ARM_CFG] SERVE START (DP=1 x2, GPU0:8002 GPU1:8003)"
  P0=$(launch_server "$ARM_CFG" 0 8002 "$CKPT")
  P1=$(launch_server "$ARM_CFG" 1 8003 "$CKPT")
  log "[$ARM_CFG] server pids: gpu0=$P0 gpu1=$P1 — waiting for health"
  ready 8002; ready 8003
  R0=$(test_gen 8002); echo "$R0" > "$LOGDIR/$ARM_CFG/validation_gpu0.json"
  R1=$(test_gen 8003); echo "$R1" > "$LOGDIR/$ARM_CFG/validation_gpu1.json"
  if ! echo "$R0" | grep -q '"choices"' || ! echo "$R1" | grep -q '"choices"'; then
    log "[$ARM_CFG] VALIDATION FAILED — skipping eval for this arm"; kill $P0 $P1 2>/dev/null; continue
  fi
  log "[$ARM_CFG] VALIDATION OK — starting eval (WORKERS=14/server, versions split across GPUs)"
  run_versions sbv_host_lora2.yaml 0 2 4 &
  J0=$!
  run_versions sbv_host_lora_8003.yaml 1 3 &
  J1=$!
  wait $J0; wait $J1
  log "[$ARM_CFG] TEARDOWN"
  kill $P0 $P1 2>/dev/null; sleep 3; pkill -9 -f "sglang.launch_server" 2>/dev/null
  log "[$ARM_CFG] ARM COMPLETE"
done

log "LOO ABLATION ROUND 1 COMPLETE"
grep _instances "$LOGDIR"/*/eval_v*.log 2>/dev/null | tee -a "$ROOT/report.log"
