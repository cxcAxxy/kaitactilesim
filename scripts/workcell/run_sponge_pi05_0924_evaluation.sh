#!/usr/bin/env bash
# Sponge grasp pi0.5 0924: strict smoke plus N=20/video=20 for 20k and final.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SIM_PYTHON="$SIM/.pixi/envs/default/bin/python"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/workcell/run_sponge_pi05_0924_evaluation.sh [STEP ...]

Default steps: 20000 24999
Each step runs a recorded smoke trial followed by N=20/video=20 evaluation.
The SPONGE_PI05_GPU, SPONGE_PI05_PORT, SPONGE_PI05_OUTPUT_DIR, and
SPONGE_PI05_SMOKE_SECONDS environment overrides remain available.
EOF
  exit 0
fi

export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json}"
export SHARED_PI05_TASK=sponge-grasp
export BULB_PI05_CHECKPOINT_ROOT="${SPONGE_PI05_CHECKPOINT_ROOT:-/nas/chenxianchi/datasets/sim/sponge-grasp/pi05/0924_200/checkpoints/pi05_kaihand/sponge-grasp_bs128_25k}"
export BULB_PI05_REFERENCE_DATASET="${SPONGE_PI05_REFERENCE_DATASET:-/nas/chenxianchi/datasets/sim/sponge-grasp/pi05/0924_200}"
export BULB_PI05_OUTPUT_DIR="${SPONGE_PI05_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/sponge/sponge_pi05_0924_$(date +%Y%m%d_%H%M%S)_$$}"
export BULB_PI05_GPU="${SPONGE_PI05_GPU:-0}"
export BULB_PI05_PORT="${SPONGE_PI05_PORT:-18783}"
export BULB_PI05_SMOKE_SECONDS="${SPONGE_PI05_SMOKE_SECONDS:-2}"

if [[ ! -x "$SIM_PYTHON" ]]; then
  echo "Simulation Python executable is missing: $SIM_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
  echo "NVIDIA EGL vendor configuration is missing: $__EGL_VENDOR_LIBRARY_FILENAMES" >&2
  exit 2
fi

"$SIM_PYTHON" "$SIM/scripts/collect/collect.py" \
  --check --task sponge-grasp --render-backend hardware

if (( $# == 0 )); then
  set -- 20000 24999
fi

exec bash "$SCRIPT_DIR/run_bulb_pi05_0920_evaluation.sh" "$@"
