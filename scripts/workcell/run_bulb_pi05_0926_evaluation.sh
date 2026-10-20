#!/usr/bin/env bash
# Evaluate the 0926 Bulb pi0.5 checkpoints with the frozen N=20/video=20 protocol.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SIM_PYTHON="$SIM/.pixi/envs/default/bin/python"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: bash scripts/workcell/run_bulb_pi05_0926_evaluation.sh [STEP ...]

Default steps: 20000 24999
Each step runs a recorded smoke trial followed by N=20/video=20 evaluation.
The standard BULB_PI05_GPU, BULB_PI05_PORT, BULB_PI05_OUTPUT_DIR, and
BULB_PI05_SMOKE_SECONDS environment overrides remain available. Old 0926
checkpoints have no bound action provenance, were trained from v12 mechanics,
and now require
SHARED_PI05_ALLOW_LEGACY_UNBOUND_DIAGNOSTIC=1; their results are diagnostic,
never formal metrics.
EOF
  exit 0
fi

export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json}"
export BULB_PI05_CHECKPOINT_ROOT="${BULB_PI05_CHECKPOINT_ROOT:-/nas/chenxianchi/datasets/sim/bulb-screw/pi05/0926_200/checkpoints/pi05_kaihand/bulb-screw_bs128_25k}"
export BULB_PI05_REFERENCE_DATASET="${BULB_PI05_REFERENCE_DATASET:-/nas/chenxianchi/datasets/sim/bulb-screw/pi05/0926_200}"
export BULB_PI05_OUTPUT_DIR="${BULB_PI05_OUTPUT_DIR:-/cpfs_infra/user/chenxianchi/evaluations/bulb/bulb_pi05_0926_$(date +%Y%m%d_%H%M%S)_$$}"
export BULB_PI05_GPU="${BULB_PI05_GPU:-0}"
export BULB_PI05_PORT="${BULB_PI05_PORT:-18783}"

if [[ ! -x "$SIM_PYTHON" ]]; then
  echo "Simulation Python executable is missing: $SIM_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$__EGL_VENDOR_LIBRARY_FILENAMES" ]]; then
  echo "NVIDIA EGL vendor configuration is missing: $__EGL_VENDOR_LIBRARY_FILENAMES" >&2
  exit 2
fi

"$SIM_PYTHON" "$SIM/scripts/collect/collect.py" \
  --check --task bulb-screw --render-backend hardware

if (( $# == 0 )); then
  set -- 20000 24999
fi

exec bash "$SCRIPT_DIR/run_bulb_pi05_0920_evaluation.sh" "$@"
