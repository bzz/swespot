#!/bin/bash
# Phase 3b — clean (bug-free) rerun of the ms-swift-matched config, from scratch, to get the real
# result uncounfounded by the earlier checkpoint-resume bug (see FINAL_REPORT.md "Phase 3b").
# Single detached driver: train (2 GPUs, FSDP, resume_from_checkpoint: False throughout) -> serve
# (two DP=1 sglang servers) -> eval N=5 split across both GPUs -> teardown. No manual scoring
# serialization needed this time: eval/sbv.sh now flocks run_evaluation itself.
set -u
VENV=/home/alex/swespot/.venv/bin
SVENV=/home/alex/serving/.venv/bin/python3
RECIPE=/home/alex/swespot/train/torchtune/lora_finetune_distributed.py
DIR=/home/alex/swespot/train/torchtune
CFG=final_lora_r128_lr5e-4_2ep_msmatched_clean
CKPT=$DIR/outputs/$CFG/epoch_1
MS=torchtune_qwen3_lora_lr5e-4_2ep_msmatched_clean
BASE=Qwen/Qwen3-4B-Instruct-2507
ROOT=/tmp/tt_msmatched_clean
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
RESULTS=$ROOT/results.tsv
echo -e "phase\trc\tend" > "$RESULTS"

log() { echo "[$(date +%m-%d_%H:%M:%S)] $1" | tee -a "$ROOT/driver.log"; }
gpu_free() {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')" -lt 1000 ]; do sleep 5; done
}

# ---- Phase 1: train from scratch (2 GPUs, FSDP) ----
gpu_free
log "TRAIN START $CFG (clean, resume_from_checkpoint: False)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_PROJECT=swespot_torchtune \
PYTHONPATH=$DIR \
"$VENV/torchrun" --nproc_per_node=2 --master_port=29552 \
  "$RECIPE" --config "$DIR/$CFG.yaml" \
  > "$LOGDIR/train.log" 2>&1
rc=$?
echo -e "train\t$rc\t$(date +%m-%d_%H:%M:%S)" >> "$RESULTS"
log "TRAIN END rc=$rc"
if [ $rc -ne 0 ]; then log "TRAIN FAILED — aborting before eval"; exit 1; fi
if [ ! -d "$CKPT" ]; then log "CHECKPOINT MISSING at $CKPT — aborting"; exit 1; fi

# sanity: a genuinely clean, uninterrupted 2-epoch run should show exactly 4 "Validation loss"
# lines (steps 500/1000/1500/2000) and reach global step 2010, not more (8 lines / >2010 would mean
# it got interrupted and resumed again, re-tripping the epochs_run resume bug).
nval=$(tr '\r' '\n' < "$LOGDIR/train.log" | grep -c "Validation loss")
maxstep=$(tr '\r' '\n' < "$LOGDIR/train.log" | grep -oE "^[0-9]\|[0-9]+\|Loss" | grep -oE "[0-9]+" | sed -n '2~2p' | sort -n | tail -1)
log "SANITY: $nval validation checkpoints logged, max global step observed = ${maxstep:-unknown} (expect 4 and 2010)"

# ---- Phase 2: serve (two independent DP=1 sglang servers, one per GPU) ----
gpu_free
launch() { # gpu port -> echoes pid
  CUDA_VISIBLE_DEVICES=$1 $SVENV -m sglang.launch_server \
    --model-path "$BASE" --served-model-name qwen34i \
    --tensor-parallel-size 1 --data-parallel-size 1 --context-length 48000 \
    --max-loras-per-batch 2 --lora-paths "{\"lora_name\":\"django\",\"lora_path\":\"$CKPT\",\"pinned\":true}" \
    --max-lora-rank 128 --lora-target-modules all --tool-call-parser qwen25 \
    --host 0.0.0.0 --port "$2" --api-key swespot > "$LOGDIR/server_gpu$1.log" 2>&1 &
  echo $!
}
ready() { until curl -s "http://localhost:$1/health" -H "Authorization: Bearer swespot" >/dev/null 2>&1; do sleep 5; done; }
test_gen() {
  curl -s "http://localhost:$1/v1/chat/completions" -H "Authorization: Bearer swespot" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$BASE:django\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK\"}],\"max_tokens\":8,\"temperature\":0}"
}

log "SERVE START (DP=1 x2, GPU0:8002 GPU1:8003)"
P0=$(launch 0 8002)
P1=$(launch 1 8003)
log "server pids: gpu0=$P0 gpu1=$P1 — waiting for health"
ready 8002; ready 8003
R0=$(test_gen 8002); echo "$R0" > "$LOGDIR/validation_gpu0.json"
R1=$(test_gen 8003); echo "$R1" > "$LOGDIR/validation_gpu1.json"
if ! echo "$R0" | grep -q '"choices"' || ! echo "$R1" | grep -q '"choices"'; then
  log "VALIDATION FAILED — aborting before eval"; kill $P0 $P1 2>/dev/null; exit 1
fi
log "VALIDATION OK — starting eval (WORKERS=14/server, versions split across GPUs, scoring self-serializes via flock)"

# ---- Phase 3: eval (sbv.sh, N=5, versions split across both GPUs) ----
cd /home/alex/swespot
run_versions() { # config versions...
  local cfg=$1; shift
  for V in "$@"; do
    log "EVAL $MS V=$V ($cfg) START"
    RUN_EVAL=true VERSION=$V WORKERS=14 MS=$MS \
      MODEL="openai/$BASE:django" CONFIG=eval/$cfg REPO=django HASH=e13b714 \
      bash eval/sbv.sh > "$LOGDIR/eval_v${V}.log" 2>&1
    rc=$?
    echo -e "eval_v$V\t$rc\t$(date +%m-%d_%H:%M:%S)" >> "$RESULTS"
    log "EVAL $MS V=$V END rc=$rc"
  done
}
run_versions sbv_host_lora2.yaml 0 2 4 &
J0=$!
run_versions sbv_host_lora_8003.yaml 1 3 &
J1=$!
wait $J0; wait $J1

log "TEARDOWN"
kill $P0 $P1 2>/dev/null; sleep 3; pkill -9 -f "sglang.launch_server" 2>/dev/null
log "MSMATCHED CLEAN RERUN COMPLETE"
grep _instances "$LOGDIR"/eval_v*.log 2>/dev/null | tee -a "$ROOT/report.log"
