#!/usr/bin/env bash
# Poker-draw: 10 paired pi0.5 and pi0.5+T-Rex simulation rollouts.
# Edit only the settings below, then run: bash scripts/workcell/run_poker_pi05_trex_comparison.sh
set -euo pipefail

# ===== 可修改配置 =====
SIM_ROOT="/cpfs_infra/user/chenxianchi/code/kaitactilesim"
OPENPI_ROOT="/cpfs_infra/user/chenxianchi/code/openpi"
TREX_OPENPI_ROOT="/cpfs_infra/user/chenxianchi/yidu/openpi-main"
MODEL_PYTHON="/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python"
SIM_PYTHON="/cpfs_infra/user/chenxianchi/miniconda3/envs/vtla/bin/python"

DEPLOYMENT_MANIFEST="/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/deployments/pi05_25k_20260924/deployment_manifest.json"
BASE_PYTORCH="/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_25k"
TACTILE_DATA_ROOT="/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/0920_200"
EXPERT_CHECKPOINT="/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/tactile_expert_runs/late4_right_resume_3000_to_10000_20260923_203128/step_010000.pt"
REFERENCE_DATASET="/nas/chenxianchi/datasets/sim/poker-draw/pi05/0920_200"

OUTPUT_ROOT="${OUTPUT_ROOT:-/nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/eval}"
RUN_NAME="pi05_vs_trex_$(date +%Y%m%d_%H%M%S)_$$"
NUM_TRIALS=10
SEED_START=0
VIDEO_COUNT=10
SAVE_RAW=1
DRY_RUN="${DRY_RUN:-0}"             # 1 只检查并打印两组推理命令，不启动模型
EXECUTE_STEPS=30
TACTILE_REFINE_EVERY=5
MAX_SIM_SECONDS=50
SUCCESS_HOLD_SECONDS=5
FULL_DURATION_EVALUATION=1       # 记录到成功或 50 秒；保留穿桌/掉牌诊断，结果标为 diagnostic
TRIAL_WALL_LIMIT=3600
RECORD_FPS=10
REVIEW_SECOND_CAMERA="overhead"
XY_JITTER_MM=4.0
YAW_JITTER_DEG=0.5
GPU_ID=0
SERVER_PORT=18784
SERVER_WAIT_SECONDS=600
# ===== 配置结束 =====

RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
SERVER_PID=""

fail() { echo "ERROR: $*" >&2; exit 2; }

for executable in "$MODEL_PYTHON" "$SIM_PYTHON"; do
  [[ -x "$executable" ]] || fail "Python 不可执行: $executable"
done
"$SIM_PYTHON" -c 'import h5py, matplotlib, PIL, torch' || fail "仿真环境缺少记录视频/原始数据所需的 Python 依赖"
[[ -f "$DEPLOYMENT_MANIFEST" ]] || fail "缺少部署清单: $DEPLOYMENT_MANIFEST"
[[ -f "$EXPERT_CHECKPOINT" ]] || fail "缺少触觉专家权重: $EXPERT_CHECKPOINT"
[[ -f "$BASE_PYTORCH/model.safetensors" ]] || fail "缺少转换后的 pi0.5 权重: $BASE_PYTORCH"
[[ -f "$REFERENCE_DATASET/meta/info.json" ]] || fail "参考数据集无效: $REFERENCE_DATASET"
[[ -d "$TACTILE_DATA_ROOT" ]] || fail "触觉数据集不存在: $TACTILE_DATA_ROOT"
[[ -f "$TREX_OPENPI_ROOT/src/openpi/tactile_expert/model.py" ]] || fail "触觉模型代码不存在: $TREX_OPENPI_ROOT"
[[ -f "$SIM_ROOT/scripts/evaluate/evaluate.py" ]] || fail "统一评估入口不存在: $SIM_ROOT"

for integer in NUM_TRIALS SEED_START VIDEO_COUNT EXECUTE_STEPS TACTILE_REFINE_EVERY TRIAL_WALL_LIMIT RECORD_FPS GPU_ID SERVER_PORT SERVER_WAIT_SECONDS SAVE_RAW DRY_RUN; do
  value="${!integer}"
  [[ "$value" =~ ^[0-9]+$ ]] || fail "$integer 必须是非负整数，实际为 $value"
done
(( NUM_TRIALS > 0 && VIDEO_COUNT <= NUM_TRIALS )) || fail "NUM_TRIALS / VIDEO_COUNT 无效"
(( EXECUTE_STEPS > 0 && TACTILE_REFINE_EVERY > 0 && TACTILE_REFINE_EVERY <= EXECUTE_STEPS )) || fail "动作执行步数或触觉刷新间隔无效"
(( SERVER_PORT > 0 && SERVER_PORT <= 65535 && SERVER_WAIT_SECONDS > 0 )) || fail "服务端口或启动等待时间无效"
(( SAVE_RAW == 0 || SAVE_RAW == 1 )) || fail "SAVE_RAW 只能是 0 或 1"
(( DRY_RUN == 0 || DRY_RUN == 1 )) || fail "DRY_RUN 只能是 0 或 1"
(( FULL_DURATION_EVALUATION == 0 || FULL_DURATION_EVALUATION == 1 )) || fail "FULL_DURATION_EVALUATION 只能是 0 或 1"
(( SUCCESS_HOLD_SECONDS > 0 && SUCCESS_HOLD_SECONDS <= MAX_SIM_SECONDS )) || fail "SUCCESS_HOLD_SECONDS 必须在 (0, MAX_SIM_SECONDS] 内"
[[ "$RECORD_FPS" == 5 || "$RECORD_FPS" == 10 ]] || fail "RECORD_FPS 只能是 5 或 10"
[[ "$REVIEW_SECOND_CAMERA" == overhead || "$REVIEW_SECOND_CAMERA" == global ]] || fail "REVIEW_SECOND_CAMERA 只能是 overhead 或 global"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SIM_ROOT/src:$OPENPI_ROOT/src:$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$SIM_ROOT"

if (( ! DRY_RUN )) && "$SIM_PYTHON" - "$SERVER_PORT" <<'PY'
import socket
import sys
with socket.socket() as connection:
    connection.settimeout(0.5)
    sys.exit(0 if connection.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
then
  fail "端口 $SERVER_PORT 已被占用，请在脚本中修改 SERVER_PORT"
fi

mkdir -p "$OUTPUT_ROOT"
mkdir "$RUN_DIR" || fail "输出目录已存在，不覆盖: $RUN_DIR"
cp -- "${BASH_SOURCE[0]}" "$RUN_DIR/launch_script.sh"
cp -- "$DEPLOYMENT_MANIFEST" "$RUN_DIR/deployment_manifest.json"
echo "RUN_DIR=$RUN_DIR"

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
  local attempt
  for ((attempt=0; attempt<SERVER_WAIT_SECONDS; attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      tail -n 80 "$RUN_DIR/pi05_server.log" >&2 || true
      fail "pi0.5 服务启动失败"
    fi
    if "$SIM_PYTHON" - "$SERVER_PORT" <<'PY'
import sys
import urllib.request
try:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{sys.argv[1]}/healthz", timeout=1) as response:
        sys.exit(0 if response.status == 200 and response.read() == b"OK\n" else 1)
except (OSError, ValueError):
    sys.exit(1)
PY
    then
      return 0
    fi
    sleep 1
  done
  tail -n 80 "$RUN_DIR/pi05_server.log" >&2 || true
  fail "等待 pi0.5 服务超过 ${SERVER_WAIT_SECONDS} 秒"
}

evaluate_family() {
  local family="$1"
  local output_dir="$RUN_DIR/$family"
  local -a dry_run_args=()
  if (( DRY_RUN )); then
    dry_run_args+=(--dry-run)
  fi
  local -a extras=(
    --trial-wall-limit "$TRIAL_WALL_LIMIT"
    --success-hold-seconds "$SUCCESS_HOLD_SECONDS"
    --review-second-camera "$REVIEW_SECOND_CAMERA"
    --xy-jitter-mm "$XY_JITTER_MM"
    --yaw-jitter-deg "$YAW_JITTER_DEG"
  )
  if (( SAVE_RAW )); then
    extras+=(--save-raw)
  fi
  if (( FULL_DURATION_EVALUATION )); then
    extras+=(--full-duration-evaluation)
  fi
  if [[ "$family" == pi05 ]]; then
    extras+=(--server "ws://127.0.0.1:$SERVER_PORT")
  else
    extras+=(
      --openpi-root "$TREX_OPENPI_ROOT"
      --base-pytorch "$BASE_PYTORCH"
      --expert-checkpoint "$EXPERT_CHECKPOINT"
      --tactile-root "$TACTILE_DATA_ROOT"
      --tactile-refine-every "$TACTILE_REFINE_EVERY"
    )
  fi
  echo "START family=$family trials=$NUM_TRIALS output=$output_dir"
  "$SIM_PYTHON" scripts/evaluate/evaluate.py \
    --task poker-draw --model-family "$family" \
    --deployment-manifest "$DEPLOYMENT_MANIFEST" \
    --output-dir "$output_dir" \
    --num-trials "$NUM_TRIALS" --seed-start "$SEED_START" \
    --video-count "$VIDEO_COUNT" --execute-steps "$EXECUTE_STEPS" \
    --max-sim-seconds "$MAX_SIM_SECONDS" --record-fps "$RECORD_FPS" \
    --reference-dataset "$REFERENCE_DATASET" \
    "${dry_run_args[@]}" \
    -- "${extras[@]}" 2>&1 | tee "$RUN_DIR/${family}_batch.log"
}

if (( DRY_RUN )); then
  evaluate_family pi05
  evaluate_family pi05+trex
  echo "DRY_RUN COMPLETE RUN_DIR=$RUN_DIR"
  exit 0
fi

echo "START pi0.5 server port=$SERVER_PORT gpu=$GPU_ID"
env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$MODEL_PYTHON" scripts/workcell/serve_poker_pi05_policy.py \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" --port "$SERVER_PORT" \
  >"$RUN_DIR/pi05_server.log" 2>&1 &
SERVER_PID=$!
wait_for_server
evaluate_family pi05
stop_server
evaluate_family pi05+trex
echo "COMPLETE RUN_DIR=$RUN_DIR"
