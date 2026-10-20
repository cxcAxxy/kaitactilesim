#!/usr/bin/env bash
# Whiteboard wipe pi0.5 0923: warmup, recorded smoke, then N=20/video=20.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SIM_PYTHON="$SIM/.pixi/envs/default/bin/python"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/workcell/run_whiteboard_pi05_0923_evaluation.sh [STEP ...]

Default steps: 20000 24999
Each step performs model warmup, a recorded smoke trial, then N=20/video=20.
The WHITEBOARD_PI05_GPU, WHITEBOARD_PI05_PORT, WHITEBOARD_PI05_OUTPUT_DIR,
WHITEBOARD_PI05_CHECKPOINT_ROOT, WHITEBOARD_PI05_REFERENCE_DATASET, and
WHITEBOARD_PI05_SMOKE_SECONDS environment overrides remain available.
WHITEBOARD_PI05_EXECUTE_STEPS selects the replan interval (horizon by default;
the deployment recommends 8). Set
WHITEBOARD_PI05_DIAGNOSTIC_RELAX_CONTACT_GUARDS=1 to continue after the 8 N
pad-force and 0.5 N direct-hand contact guards in diagnostic-only mode. Valid
loaded pad sliding above 8 N still accumulates modelled cleaning progress.
EOF
  exit 0
fi

export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json}"
export SHARED_PI05_TASK=whiteboard-wipe
export BULB_PI05_CHECKPOINT_ROOT="${WHITEBOARD_PI05_CHECKPOINT_ROOT:-/nas/chenxianchi/datasets/sim/whiteboard-wipe/pi05/0923_200/checkpoints/pi05_kaihand/whiteboard-wipe_bs128_25k}"
export BULB_PI05_REFERENCE_DATASET="${WHITEBOARD_PI05_REFERENCE_DATASET:-/nas/chenxianchi/datasets/sim/whiteboard-wipe/pi05/0923_200}"
export BULB_PI05_OUTPUT_DIR="${WHITEBOARD_PI05_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/whiteboard/whiteboard_pi05_0923_$(date +%Y%m%d_%H%M%S)_$$}"
export BULB_PI05_GPU="${WHITEBOARD_PI05_GPU:-0}"
export BULB_PI05_PORT="${WHITEBOARD_PI05_PORT:-18783}"
export BULB_PI05_SMOKE_SECONDS="${WHITEBOARD_PI05_SMOKE_SECONDS:-2}"
export SHARED_PI05_EXECUTE_STEPS="${WHITEBOARD_PI05_EXECUTE_STEPS:-horizon}"
export SHARED_PI05_DIAGNOSTIC_RELAX_WHITEBOARD_CONTACT_GUARDS="${WHITEBOARD_PI05_DIAGNOSTIC_RELAX_CONTACT_GUARDS:-0}"

if [[ ! -x "$SIM_PYTHON" ]]; then
  echo "Simulation Python executable is missing: $SIM_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
  echo "NVIDIA EGL vendor configuration is missing: $__EGL_VENDOR_LIBRARY_FILENAMES" >&2
  exit 2
fi

"$SIM_PYTHON" "$SIM/scripts/collect/collect.py" \
  --check --task whiteboard-wipe --render-backend hardware

if (( $# == 0 )); then
  set -- 20000 24999
fi

exec bash "$SCRIPT_DIR/run_bulb_pi05_0920_evaluation.sh" "$@"
