#!/usr/bin/env bash
# Six USB checkpoints, each evaluated at execute_steps=16 and model-owned H.
# All trials are recorded. Only the 0.3 mm socket penetration stop is disabled.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="${SIM_CODE_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
MODEL_DIR=/cpfs_infra/user/chenxianchi/code/egosteer
OPENPI_DIR=/cpfs_infra/user/chenxianchi/code/openpi
MODEL_PYTHON=/cpfs_infra/user/chenxianchi/miniconda3/envs/egosteer_touch/bin/python3.10
EGOSTEER_RUN=/nas/chenxianchi/egosteer/usb_insert/0914_200/2026.09.15/12.26.11
PI05_CHECKPOINTS=/nas/chenxianchi/datasets/sim/usb_insert/pi05/0914_200/checkpoints/pi05_kaihand_usb_0914_200/usb_insert_full_trim7
NORMALIZER=/nas/chenxianchi/egosteer/normalizer/usb_insert/0914_200/normalizer.pkl
PRETRAINED_VLM=/cpfs_infra/user/chenxianchi/code/T-Rex/weights/Qwen3-VL-2B-Instruct
PORT="${USB_MATRIX_PORT:-18786}"
SEEDS=({0..19})
SERVER_PID=""
EGO_CAMERA_ARGS=()

if [[ -n "${EGOSTEER_CAMERA_VIEWS:-}" ]]; then
  if [[ "$EGOSTEER_CAMERA_VIEWS" != head && "$EGOSTEER_CAMERA_VIEWS" != head,right_wrist ]]; then
    echo "EGOSTEER_CAMERA_VIEWS must be head or head,right_wrist" >&2
    exit 2
  fi
  EGO_CAMERA_ARGS=(--camera-views "$EGOSTEER_CAMERA_VIEWS")
fi

FAMILY="${1:-all}"
case "$FAMILY" in
  all) FAMILIES=(egosteer pi05) ;;
  egosteer|pi05) FAMILIES=("$FAMILY") ;;
  *) echo "Usage: bash $0 [all|egosteer|pi05]" >&2; exit 2 ;;
esac
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "USB_MATRIX_PORT must be a TCP port in 1..65535" >&2
  exit 2
fi

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_busy() { ( : >/dev/tcp/127.0.0.1/"$PORT" ) 2>/dev/null; }
if port_busy; then
  echo "Port $PORT is already in use; choose USB_MATRIX_PORT or stop that server." >&2
  exit 1
fi

RUN_ROOT="${USB_MATRIX_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/usb_last3_exec16_H_N20_video20_diagnostic_no_penetration_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"

run_batch() {
  local evaluator=$1 destination=$2 video_count=$3 execute_steps=$4
  shift 4
  PYTHONPATH="$SIM_DIR/src" "$SIM_DIR/.venv/bin/python" \
    "$SIM_DIR/scripts/workcell/$evaluator" \
    --server "ws://127.0.0.1:$PORT" \
    --deployment-manifest "$MANIFEST" \
    --seeds "$@" \
    --execute-steps "$execute_steps" \
    --max-sim-seconds 50 \
    --trial-wall-limit 1800 \
    --video-count "$video_count" \
    --record-fps 10 \
    --xy-jitter-mm 10 \
    --yaw-jitter-deg 5 \
    --disable-penetration-guard \
    --output-dir "$destination"
}

check_smoke() {
  local destination=$1
  "$SIM_DIR/.venv/bin/python" -c '
import json
import sys
from pathlib import Path

trial = Path(sys.argv[1]) / "seed_000"
summary = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
stats = summary.get("stats", {})
if stats.get("requests", 0) < 1 or stats.get("action_steps", 0) < 1:
    raise SystemExit(f"smoke did not complete inference and action execution: {trial}")
if summary.get("sim_seconds", 0) <= 0 or not isinstance(summary.get("evaluation"), dict):
    raise SystemExit(f"smoke has no simulation time or evaluation metrics: {trial}")
review = trial / "review"
for name in ("review.mp4", "review.json", "frames.jsonl", "first_frame.png", "last_frame.png"):
    artifact = review / name
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise SystemExit(f"smoke artifact missing or empty: {artifact}")
print("SMOKE_OK", trial, "requests=", stats["requests"], "actions=", stats["action_steps"])
' "$destination"
}

wait_for_server() {
  local step=$1 log=$2 ready=0
  for ((i=0; i<600; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 60 "$log"
      echo "Server exited while loading $step" >&2
      return 1
    fi
    if port_busy; then
      ready=1
      break
    fi
    sleep 1
  done
  if (( ready == 0 )); then
    tail -n 60 "$log"
    echo "Server did not listen on port $PORT within 600 seconds" >&2
    return 1
  fi
}

for MODEL_FAMILY in "${FAMILIES[@]}"; do
  if [[ "$MODEL_FAMILY" == egosteer ]]; then
    STEPS=(4000 4200 4400)
    EVALUATOR=evaluate_usb_policy_batch.py
  else
    STEPS=(15000 20000 22000)
    EVALUATOR=evaluate_usb_pi05_policy_batch.py
  fi

  for STEP in "${STEPS[@]}"; do
    if port_busy; then
      echo "Port $PORT became occupied before $MODEL_FAMILY step=$STEP" >&2
      exit 1
    fi
    DEPLOY="$RUN_ROOT/${MODEL_FAMILY}_step${STEP}_deployment"
    if [[ "$MODEL_FAMILY" == egosteer ]]; then
      "$SIM_DIR/.venv/bin/python" \
        "$SIM_DIR/scripts/workcell/prepare_usb_egosteer_deployment.py" \
        --checkpoint "$EGOSTEER_RUN/checkpoints/update_step=$STEP" \
        --model-config "$EGOSTEER_RUN/.hydra/config.yaml" \
        --normalizer "$NORMALIZER" \
        --model-project "$MODEL_DIR" \
        --model-python "$MODEL_PYTHON" \
        --pretrained-vlm "$PRETRAINED_VLM" \
        --port "$PORT" \
        --output-dir "$DEPLOY" \
        "${EGO_CAMERA_ARGS[@]}"
    else
      cd "$OPENPI_DIR"
      "$OPENPI_DIR/.venv-pi05/bin/python" \
        "$SIM_DIR/scripts/workcell/prepare_usb_pi05_deployment.py" \
        --checkpoint "$PI05_CHECKPOINTS/$STEP" \
        --openpi-root "$OPENPI_DIR" \
        --hash-workers 4 \
        --output-dir "$DEPLOY"
    fi

    MANIFEST="$DEPLOY/deployment_manifest.json"
    H=$("$SIM_DIR/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["prediction_horizon"])' \
      "$MANIFEST")
    if [[ "$MODEL_FAMILY" == egosteer ]]; then
      EGO_SERVER_CAMERAS=$("$SIM_DIR/.venv/bin/python" -c \
        'import json,sys; print(",".join(json.load(open(sys.argv[1]))["observation_contract"]["cameras"]))' \
        "$MANIFEST")
    fi
    if (( H < 16 )); then
      echo "$MODEL_FAMILY step=$STEP declares H=$H, less than execute_steps=16" >&2
      exit 1
    fi
    echo "MODEL=$MODEL_FAMILY STEP=$STEP H=$H SEEDS=0..19 VIDEOS=20"
    "$SIM_DIR/.venv/bin/python" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print("contract:",d["model_family"],"action_dim=",d["action_dim"],"cameras=",d["observation_contract"]["cameras"],"checkpoint_sha256=",d["checkpoint_sha256"])' \
      "$MANIFEST"

    SERVER_LOG="$RUN_ROOT/${MODEL_FAMILY}_step${STEP}_server.log"
    if [[ "$MODEL_FAMILY" == egosteer ]]; then
      cd "$MODEL_DIR"
      CUDA_VISIBLE_DEVICES=0 \
      EGOSTEER_CAMERA_VIEWS="$EGO_SERVER_CAMERAS" \
      EGOSTEER_INFERENCE_CONFIG="$DEPLOY/inference.yaml" \
      HF_HOME=/cpfs_infra/user/chenxianchi/.cache/hf \
      HUGGINGFACE_HUB_CACHE=/cpfs_infra/user/chenxianchi/.cache/hf/hub \
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
        "$MODEL_PYTHON" -m src.serving.serve_policy \
        >"$SERVER_LOG" 2>&1 &
    else
      cd "$OPENPI_DIR"
      CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI_DIR/.venv-pi05/bin/python" \
        "$SIM_DIR/scripts/workcell/serve_usb_pi05_policy.py" \
        --deployment-manifest "$MANIFEST" --port "$PORT" \
        >"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!
    wait_for_server "$MODEL_FAMILY step=$STEP" "$SERVER_LOG"

    for EXEC in 16 "$H"; do
      echo "SMOKE model=$MODEL_FAMILY step=$STEP execute_steps=$EXEC"
      SMOKE_DIR="$RUN_ROOT/${MODEL_FAMILY}_step${STEP}_exec${EXEC}_smoke"
      run_batch "$EVALUATOR" "$SMOKE_DIR" 1 "$EXEC" 0
      check_smoke "$SMOKE_DIR"
      echo "EVAL model=$MODEL_FAMILY step=$STEP execute_steps=$EXEC N=20 video_count=20"
      run_batch "$EVALUATOR" \
        "$RUN_ROOT/${MODEL_FAMILY}_step${STEP}_exec${EXEC}_N20" \
        20 "$EXEC" "${SEEDS[@]}"
    done
    stop_server
  done
done

echo "COMPLETE RUN_ROOT=$RUN_ROOT"
