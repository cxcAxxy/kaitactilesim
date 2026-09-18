#!/usr/bin/env bash
# Six PickPlace checkpoints, each at execute_steps=16 and model-owned H, N=20 with all videos.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM="${SIM_CODE_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
EGO=/cpfs_infra/user/chenxianchi/code/egosteer
OPENPI=/cpfs_infra/user/chenxianchi/code/openpi
EGO_PYTHON=/cpfs_infra/user/chenxianchi/miniconda3/envs/egosteer_touch/bin/python3.10
EGO_RUN=/nas/chenxianchi/egosteer/pickplace/0914_200/2026.09.16/13.42.35
PI05_CKPTS=/nas/chenxianchi/datasets/sim/pickplace/pi05/0914_200/checkpoints/pi05_kaihand_pickplace/pickplace_full_0914_200_fsdp8
NORMALIZER=/nas/chenxianchi/egosteer/normalizer/pickplace/0914_200/normalizer.pkl
VLM=/cpfs_infra/user/chenxianchi/code/T-Rex/weights/Qwen3-VL-2B-Instruct
PORT="${PICKPLACE_MATRIX_PORT:-18787}"
RUN_ROOT="${PICKPLACE_MATRIX_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/pickplace_last3_exec16_H_N20_video20_$(date +%Y%m%d_%H%M%S)_$$}"
FAMILY="${1:-all}"
DIAGNOSTIC_RELAX_IK="${PICKPLACE_MATRIX_DIAGNOSTIC_RELAX_IK:-0}"
RESUME="${PICKPLACE_MATRIX_RESUME:-0}"
ALLOW_SOURCE_CHANGE="${PICKPLACE_MATRIX_ALLOW_SOURCE_CHANGE:-0}"
SERVER_PID=""
SEEDS=({0..19})
EGO_CAMERA_ARGS=()

if [[ -n "${EGOSTEER_CAMERA_VIEWS:-}" ]]; then
  if [[ "$EGOSTEER_CAMERA_VIEWS" != head && "$EGOSTEER_CAMERA_VIEWS" != head,right_wrist ]]; then
    echo "EGOSTEER_CAMERA_VIEWS must be head or head,right_wrist" >&2
    exit 2
  fi
  EGO_CAMERA_ARGS=(--camera-views "$EGOSTEER_CAMERA_VIEWS")
fi

case "$FAMILY" in
  all) FAMILIES=(egosteer pi05) ;;
  egosteer|pi05) FAMILIES=("$FAMILY") ;;
  *) echo "Usage: bash $0 [all|egosteer|pi05]" >&2; exit 2 ;;
esac
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "PICKPLACE_MATRIX_PORT must be in 1..65535" >&2
  exit 2
fi
if [[ "$DIAGNOSTIC_RELAX_IK" != 0 && "$DIAGNOSTIC_RELAX_IK" != 1 ]]; then
  echo "PICKPLACE_MATRIX_DIAGNOSTIC_RELAX_IK must be 0 or 1" >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  echo "PICKPLACE_MATRIX_RESUME must be 0 or 1" >&2
  exit 2
fi
if [[ "$ALLOW_SOURCE_CHANGE" != 0 && "$ALLOW_SOURCE_CHANGE" != 1 ]]; then
  echo "PICKPLACE_MATRIX_ALLOW_SOURCE_CHANGE must be 0 or 1" >&2
  exit 2
fi
if [[ "$ALLOW_SOURCE_CHANGE" == 1 && "$RESUME" != 1 ]]; then
  echo "PICKPLACE_MATRIX_ALLOW_SOURCE_CHANGE=1 requires PICKPLACE_MATRIX_RESUME=1" >&2
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
if port_busy; then echo "Port $PORT is occupied" >&2; exit 1; fi
if [[ "$RESUME" == 1 ]]; then
  [[ -d "$RUN_ROOT" ]] || { echo "Resume root does not exist: $RUN_ROOT" >&2; exit 1; }
else
  mkdir "$RUN_ROOT"
fi
echo "RUN_ROOT=$RUN_ROOT" >&2

wait_for_server() {
  local log=$1
  for ((i=0; i<600; i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 60 "$log"
      echo "Server exited during startup" >&2
      return 1
    fi
    if port_busy; then return 0; fi
    sleep 1
  done
  tail -n 60 "$log"
  echo "Server startup exceeded 600 seconds" >&2
  return 1
}

evaluate() {
  local destination=$1 count=$2 execute=$3
  shift 3
  local diagnostic_args=()
  local resume_args=()
  if [[ "$MODEL" == egosteer && "$DIAGNOSTIC_RELAX_IK" == 1 ]]; then
    diagnostic_args=(--diagnostic-relax-ik)
  fi
  if [[ -e "$destination" ]]; then
    if [[ "$RESUME" != 1 ]]; then
      echo "Evaluation destination already exists: $destination" >&2
      return 1
    fi
    resume_args=(--resume)
    if [[ "$ALLOW_SOURCE_CHANGE" == 1 ]]; then
      resume_args+=(--allow-source-change)
    fi
  fi
  PYTHONPATH="$SIM/src" "$SIM/.venv/bin/python" \
    "$SIM/scripts/workcell/evaluate_pickplace_policy_batch.py" \
    --server "ws://127.0.0.1:$PORT" --deployment-manifest "$MANIFEST" \
    --seeds "$@" --execute-steps "$execute" --max-sim-seconds 50 \
    --trial-wall-limit 1800 --video-count "$count" --record-fps 10 \
    --output-dir "$destination" "${diagnostic_args[@]}" "${resume_args[@]}"
}

summary_complete() {
  local summary=$1
  [[ -f "$summary" ]] || return 1
  "$SIM/.venv/bin/python" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("complete") is True else 1)' \
    "$summary"
}

check_smoke() {
  "$SIM/.venv/bin/python" -c '
import json,sys
from pathlib import Path
trial=Path(sys.argv[1])/"seed_000"
report=json.loads((trial/"summary.json").read_text(encoding="utf-8"))
if report.get("status") not in ("success","task_not_completed"):
    raise SystemExit(f"smoke task status invalid: {trial}")
if report.get("stats",{}).get("requests",0)<1 or report.get("stats",{}).get("action_steps",0)<1:
    raise SystemExit(f"smoke did not infer and act: {trial}")
if report.get("sim_seconds",0)<=0 or not isinstance(report.get("evaluation"),dict):
    raise SystemExit(f"smoke has no simulation time or evaluation: {trial}")
for name in ("review.mp4","review.json","frames.jsonl","first_frame.png","last_frame.png"):
    path=trial/"review"/name
    if not path.is_file() or path.stat().st_size==0:
        raise SystemExit(f"smoke artifact missing or empty: {path}")
print("SMOKE_OK",trial,flush=True)
' "$1"
}

for MODEL in "${FAMILIES[@]}"; do
  if [[ "$MODEL" == egosteer ]]; then STEPS=(4200 5800 7600); else STEPS=(20000 25000 29999); fi
  for STEP in "${STEPS[@]}"; do
    if port_busy; then echo "Port $PORT became occupied" >&2; exit 1; fi
    DEPLOY="$RUN_ROOT/${MODEL}_step${STEP}_deployment"
    if [[ "$RESUME" == 1 && -f "$DEPLOY/deployment_manifest.json" ]]; then
      echo "RESUME deployment MODEL=$MODEL STEP=$STEP" >&2
    elif [[ "$MODEL" == egosteer ]]; then
      "$SIM/.venv/bin/python" "$SIM/scripts/workcell/prepare_pickplace_egosteer_deployment.py" \
        --checkpoint "$EGO_RUN/checkpoints/update_step=$STEP" \
        --model-config "$EGO_RUN/.hydra/config.yaml" \
        --normalizer "$NORMALIZER" --model-project "$EGO" \
        --model-python "$EGO_PYTHON" --pretrained-vlm "$VLM" \
        --port "$PORT" --output-dir "$DEPLOY" \
        "${EGO_CAMERA_ARGS[@]}"
    else
      cd "$OPENPI"
      "$OPENPI/.venv-pi05/bin/python" \
        "$SIM/scripts/workcell/prepare_pickplace_pi05_deployment.py" \
        --checkpoint "$PI05_CKPTS/$STEP" --openpi-root "$OPENPI" \
        --hash-workers 4 --output-dir "$DEPLOY"
    fi
    MANIFEST="$DEPLOY/deployment_manifest.json"
    H=$("$SIM/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["prediction_horizon"])' "$MANIFEST")
    if [[ "$MODEL" == egosteer ]]; then
      EGO_SERVER_CAMERAS=$("$SIM/.venv/bin/python" -c \
        'import json,sys; print(",".join(json.load(open(sys.argv[1]))["observation_contract"]["cameras"]))' \
        "$MANIFEST")
    fi
    if (( H < 16 )); then echo "$MODEL step=$STEP H=$H < 16" >&2; exit 1; fi
    echo "MODEL=$MODEL STEP=$STEP H=$H" >&2
    if [[ "$MODEL" == egosteer && "$DIAGNOSTIC_RELAX_IK" == 1 ]]; then
      echo "DIAGNOSTIC ONLY: EgoSteer IK reachability abort thresholds disabled" >&2
    fi
    if summary_complete "$RUN_ROOT/${MODEL}_step${STEP}_exec16_N20/summary.json" \
      && summary_complete "$RUN_ROOT/${MODEL}_step${STEP}_exec${H}_N20/summary.json"; then
      echo "SKIP completed MODEL=$MODEL STEP=$STEP" >&2
      continue
    fi
    SERVER_LOG="$RUN_ROOT/${MODEL}_step${STEP}_server.log"
    if [[ "$MODEL" == egosteer ]]; then
      cd "$EGO"
      CUDA_VISIBLE_DEVICES=0 EGOSTEER_CAMERA_VIEWS="$EGO_SERVER_CAMERAS" \
      EGOSTEER_INFERENCE_CONFIG="$DEPLOY/inference.yaml" \
      HF_HOME=/cpfs_infra/user/chenxianchi/.cache/hf \
      HUGGINGFACE_HUB_CACHE=/cpfs_infra/user/chenxianchi/.cache/hf/hub \
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
        "$EGO_PYTHON" -m src.serving.serve_policy >>"$SERVER_LOG" 2>&1 &
    else
      cd "$OPENPI"
      CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$OPENPI/.venv-pi05/bin/python" \
        "$SIM/scripts/workcell/serve_pickplace_pi05_policy.py" \
        --deployment-manifest "$MANIFEST" --port "$PORT" >>"$SERVER_LOG" 2>&1 &
    fi
    SERVER_PID=$!
    wait_for_server "$SERVER_LOG"
    for EXEC in 16 "$H"; do
      SMOKE="$RUN_ROOT/${MODEL}_step${STEP}_exec${EXEC}_smoke"
      if [[ "$RESUME" == 1 && -d "$SMOKE" ]]; then
        echo "SKIP existing smoke MODEL=$MODEL STEP=$STEP EXEC=$EXEC" >&2
      else
        echo "SMOKE MODEL=$MODEL STEP=$STEP EXEC=$EXEC" >&2
        evaluate "$SMOKE" 1 "$EXEC" 0
      fi
      check_smoke "$SMOKE"
      FULL="$RUN_ROOT/${MODEL}_step${STEP}_exec${EXEC}_N20"
      if summary_complete "$FULL/summary.json"; then
        echo "SKIP completed MODEL=$MODEL STEP=$STEP EXEC=$EXEC" >&2
      else
        echo "EVAL MODEL=$MODEL STEP=$STEP EXEC=$EXEC N=20 VIDEOS=20" >&2
        evaluate "$FULL" 20 "$EXEC" "${SEEDS[@]}"
      fi
    done
    stop_server
  done
done
echo "COMPLETE RUN_ROOT=$RUN_ROOT" >&2
