#!/usr/bin/env bash
# Bulb pi0.5 0920: N=20/video=20 at execute_steps=16 and the model horizon.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM="${SIM_CODE_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
OPENPI="${OPENPI_CODE_DIR:-/cpfs_infra/user/chenxianchi/code/openpi}"
CHECKPOINT_ROOT="${BULB_PI05_CHECKPOINT_ROOT:-/nas/chenxianchi/datasets/sim/bulb-screw/pi05/0920_200/checkpoints/pi05_kaihand/bulb-scew_bs128_skip7}"
PORT="${BULB_PI05_PORT:-18783}"
GPU="${BULB_PI05_GPU:-0}"
RUN_ROOT="${BULB_PI05_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/bulb_pi05_0920_N20_video20_$(date +%Y%m%d_%H%M%S)_$$}"
SMOKE_SECONDS="${BULB_PI05_SMOKE_SECONDS:-2}"
MODEL_PYTHON="$OPENPI/.venv-pi05/bin/python"
SIM_PYTHON="$SIM/.pixi/envs/default/bin/python"
SERVER_PID=""

usage() {
  cat <<EOF
Usage: bash $0 [STEP ...]

Default steps: 20000 25000

Optional environment variables:
  BULB_PI05_GPU              CUDA device (default: 0)
  BULB_PI05_PORT             model server port (default: 18783)
  BULB_PI05_OUTPUT_DIR       new output root
  BULB_PI05_CHECKPOINT_ROOT  directory containing step checkpoints
  BULB_PI05_SMOKE_SECONDS    smoke-test simulation seconds (default: 2)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if (( $# )); then
  STEPS=("$@")
else
  STEPS=(20000 25000)
fi

for executable in "$MODEL_PYTHON" "$SIM_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    echo "Python executable is missing: $executable" >&2
    exit 2
  fi
done
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "BULB_PI05_PORT must be in 1..65535" >&2
  exit 2
fi
for step in "${STEPS[@]}"; do
  if ! [[ "$step" =~ ^[0-9]+$ ]] || [[ ! -d "$CHECKPOINT_ROOT/$step" ]]; then
    echo "Checkpoint step is invalid or missing: $CHECKPOINT_ROOT/$step" >&2
    exit 2
  fi
done
if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "Port $PORT is already occupied; set BULB_PI05_PORT to a free port" >&2
  exit 1
fi
mkdir "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT" >&2

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

wait_for_server() {
  local log=$1
  local ready=0
  for ((attempt=0; attempt<600; attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 80 "$log" >&2
      echo "Model server exited during startup" >&2
      return 1
    fi
    # Reading the server's own readiness log avoids opening an invalid raw TCP
    # connection to the WebSocket endpoint.
    if grep -q "server listening" "$log"; then
      ready=1
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    tail -n 80 "$log" >&2
    echo "Model server startup exceeded 600 seconds" >&2
    return 1
  fi
}

evaluate() {
  local output=$1
  shift
  "$SIM_PYTHON" "$SIM/scripts/evaluate/evaluate.py" \
    --task bulb-screw --model-family pi05 \
    --deployment-manifest "$MANIFEST" \
    --output-dir "$output" \
    --cameras head right_wrist \
    "$@" \
    -- --server "ws://127.0.0.1:$PORT"
}

check_smoke() {
  "$SIM_PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
batch = json.loads((root / "summary.json").read_text(encoding="utf-8"))
trial = json.loads((root / "seed_000/summary.json").read_text(encoding="utf-8"))
if batch.get("valid_trials") != 1:
  raise SystemExit(f"smoke trial is invalid: {root}")
stats = trial.get("stats", {})
if stats.get("requests", 0) < 1 or stats.get("action_steps", 0) < 1:
  raise SystemExit(f"smoke did not complete inference and action execution: {root}")
review = root / "seed_000/review"
for name in ("review.mp4", "review.json", "frames.jsonl"):
  artifact = review / name
  if not artifact.is_file() or artifact.stat().st_size == 0:
    raise SystemExit(f"smoke review artifact is missing or empty: {artifact}")
metadata = json.loads((review / "review.json").read_text(encoding="utf-8"))
if metadata.get("model_input_cameras_displayed") != ["head", "right_wrist"]:
  raise SystemExit(f"smoke review omitted a model camera: {review}")
print(f"SMOKE_OK={root}", flush=True)
PY
}

for STEP in "${STEPS[@]}"; do
  DEPLOYMENT="$RUN_ROOT/step${STEP}_deployment"
  MANIFEST="$DEPLOYMENT/deployment_manifest.json"
  SERVER_LOG="$RUN_ROOT/step${STEP}_server.log"
  SMOKE="$RUN_ROOT/step${STEP}_smoke"
  FORMAL="$RUN_ROOT/step${STEP}_N20_video20"

  echo "PREPARE step=$STEP" >&2
  (
    cd "$OPENPI"
    "$MODEL_PYTHON" "$SIM/scripts/workcell/prepare_shared_task_pi05_deployment.py" \
      --task bulb-screw --checkpoint "$CHECKPOINT_ROOT/$STEP" \
      --openpi-root "$OPENPI" --hash-workers 4 --output-dir "$DEPLOYMENT"
  )

  echo "START_SERVER step=$STEP port=$PORT gpu=$GPU" >&2
  (
    cd "$OPENPI"
    exec env CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$MODEL_PYTHON" "$SIM/scripts/workcell/serve_shared_task_pi05_policy.py" \
      --deployment-manifest "$MANIFEST" --port "$PORT"
  ) >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server "$SERVER_LOG"

  echo "SMOKE step=$STEP seconds=$SMOKE_SECONDS" >&2
  if ! evaluate "$SMOKE" \
    --num-trials 1 --video-count 1 --execute-steps 16 \
    --max-sim-seconds "$SMOKE_SECONDS" --record-fps 10; then
    tail -n 100 "$SMOKE/seed_000.log" >&2 || true
    exit 1
  fi
  check_smoke "$SMOKE"

  echo "FORMAL step=$STEP N=20 videos=20 execute_steps=16,horizon" >&2
  if ! evaluate "$FORMAL" \
    --num-trials 20 --video-count 20 --execute-steps 16 horizon \
    --record-fps 10; then
    echo "Formal evaluation failed; inspect $FORMAL and $SERVER_LOG" >&2
    exit 1
  fi
  stop_server
done

echo "COMPLETE RUN_ROOT=$RUN_ROOT" >&2
