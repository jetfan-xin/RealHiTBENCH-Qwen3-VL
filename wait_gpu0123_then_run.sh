#!/usr/bin/env bash
set -euo pipefail

GPUS=(0 1 2 3)
MEM_THRESHOLD=10          # MiB
STABLE_SECONDS=10         # 连续稳定 10 秒
INTERVAL=1                # 每秒检查一次
HEARTBEAT_EVERY=60        # 每 60 秒记录一次“还在等”

WORKDIR="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/inference"
LOGDIR="/export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/result/qwen3vl_local"
LOGFILE="${LOGDIR}/image_full_batch4.log"

CUDA_VISIBLE="0,1,2,3"
CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} python inference_qwen3vl_local.py \
  --model_dir /data/pan/4xin/models/Qwen3-VL-8B-Instruct \
  --data_path /data/pan/4xin/datasets/RealHiTBench \
  --qa_path /export/home/pan/4xin/RealHiTBENCH-Qwen3-VL/data \
  --modality image \
  --batch_size 4"

echo "[watcher] waiting GPUs ${CUDA_VISIBLE} all mem < ${MEM_THRESHOLD}MiB for ${STABLE_SECONDS}s ..."
echo "[watcher] heartbeat every ${HEARTBEAT_EVERY}s"
echo "[watcher] workdir=${WORKDIR}"
echo "[watcher] logfile=${LOGFILE}"
echo "[watcher] cmd=${CMD}"

if [[ ! -d "$WORKDIR" ]]; then
  echo "[watcher][ERROR] workdir not found: $WORKDIR"
  exit 2
fi

# 连续满足条件的秒数
stable=0
last_state="unknown"  # "all_free" or "not_free"
tick=0

while true; do
  ts=$(date "+%F %T")

  # 检查 0/1/2/3 是否都 < threshold
  all_free=1
  status_parts=()
  for g in "${GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | head -n1 | xargs)
    status_parts+=("g${g}=${used}MiB")
    if [[ "$used" -ge "$MEM_THRESHOLD" ]]; then
      all_free=0
    fi
  done

  if [[ "$all_free" -eq 1 ]]; then
    state="all_free"
    stable=$((stable+1))
  else
    state="not_free"
    stable=0
  fi

  # 状态变化才写日志
  if [[ "$state" != "$last_state" ]]; then
    echo "[watcher] ${ts} state changed: ${last_state} -> ${state} (${status_parts[*]})"
    last_state="$state"
  fi

  # 心跳
  tick=$((tick+INTERVAL))
  if (( tick % HEARTBEAT_EVERY == 0 )); then
    echo "[watcher] ${ts} still waiting... stable=${stable}/${STABLE_SECONDS} (${status_parts[*]})"
  fi

  # 触发
  if [[ "$stable" -ge "$STABLE_SECONDS" ]]; then
    echo "[watcher] ${ts} GPUs ${CUDA_VISIBLE} free for ${STABLE_SECONDS}s -> starting job now"
    mkdir -p "$LOGDIR"
    cd "$WORKDIR"
    nohup bash -lc "$CMD" > "$LOGFILE" 2>&1 &
    echo "[watcher] started, pid=$!"
    exit 0
  fi

  sleep "$INTERVAL"
done
