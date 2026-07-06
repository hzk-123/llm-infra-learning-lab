#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-0.5b-instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$HOME/llm-sprint/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE"
mkdir -p logs/08_serving_vllm results/08_serving_vllm

echo "vLLM server config"
echo "  MODEL_ID=$MODEL_ID"
echo "  SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "  HOST=$HOST"
echo "  PORT=$PORT"
echo "  DTYPE=$DTYPE"
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "  VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-<empty>}"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  HF_HOME=$HF_HOME"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo

extra_args=()
if [[ -n "$VLLM_EXTRA_ARGS" ]]; then
  # VLLM_EXTRA_ARGS is intended for simple extra flags such as:
  #   --enforce-eager
  #   --disable-log-requests
  # Avoid values that need nested quoting here.
  # shellcheck disable=SC2206
  extra_args=($VLLM_EXTRA_ARGS)
fi

vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  "${extra_args[@]}"
