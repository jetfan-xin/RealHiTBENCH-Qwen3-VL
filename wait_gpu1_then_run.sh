#!/usr/bin/env bash
set -euo pipefail

GPU_ID=1
MEM_THRESHOLD=10          # MiB
STABLE_SECONDS=10         # 连续空闲 10 秒
INTERVAL=1                # 每秒检查一次
HEARTBEAT_EVERY=60        # 每 60 秒记录一次“还在等”

WORKDIR="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference"
LOGDIR="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local"
LOGFILE="${LOGDIR}/image_full_batch3.log"

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_qwen3vl_local.py \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
  --modality image \
  --batch_size 3"

echo "[watcher] waiting GPU=${GPU_ID} mem < ${MEM_THRESHOLD}MiB for ${STABLE_SECONDS}s ..."
echo "[watcher] heartbeat every ${HEARTBEAT_EVERY}s"
echo "[watcher] workdir=${WORKDIR}"
echo "[watcher] logfile=${LOGFILE}"

if [[ ! -d "$WORKDIR" ]]; then
  echo "[watcher][ERROR] workdir not found: $WORKDIR"
  exit 2
fi

stable=0
last_state="unknown"   # "free" or "busy"
tick=0

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -n1 | xargs)
  ts=$(date "+%F %T")

  if [[ "$used" -lt "$MEM_THRESHOLD" ]]; then
    state="free"
    stable=$((stable+1))
  else
    state="busy"
    stable=0
  fi

  # 只有状态变化才写日志
  if [[ "$state" != "$last_state" ]]; then
    echo "[watcher] ${ts} GPU${GPU_ID} state changed: ${last_state} -> ${state} (used=${used}MiB)"
    last_state="$state"
  fi

  # 心跳：每 HEARTBEAT_EVERY 秒写一次“还在等”
  tick=$((tick+INTERVAL))
  if (( tick % HEARTBEAT_EVERY == 0 )); then
    echo "[watcher] ${ts} still waiting... GPU${GPU_ID} used=${used}MiB stable=${stable}/${STABLE_SECONDS}"
  fi

  # 触发
  if [[ "$stable" -ge "$STABLE_SECONDS" ]]; then
    echo "[watcher] ${ts} GPU${GPU_ID} free for ${STABLE_SECONDS}s -> starting job now"
    mkdir -p "$LOGDIR"
    cd "$WORKDIR"
    nohup bash -lc "$CMD" > "$LOGFILE" 2>&1 &
    echo "[watcher] started, pid=$!"
    exit 0
  fi

  sleep "$INTERVAL"
done
