#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHON=/cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python
CONVERTER=${REPO_ROOT}/scripts/workcell/convert_kaihand_to_lerobot_v3.py

export HF_HOME=/cpfs_infra/user/chenxianchi/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

run_task() {
  local task=$1
  local dataset_dir=$2
  local repo_id=$3

  "${PYTHON}" "${CONVERTER}" \
    --input-dir "/nas/chenxianchi/datasets/sim/${dataset_dir}/raw/0920_200" \
    --output-dir "/nas/chenxianchi/datasets/sim/${dataset_dir}/lerobot_v3/0920_200" \
    --task "${task}" \
    --repo-id "${repo_id}" \
    --expected-episodes 200 \
    --workers 4 \
    --ffmpeg-threads 1 \
    --resume
}

# Run sequentially so the conversions do not compete for NAS bandwidth.
# Re-running this script resumes unfinished episodes and accepts already
# published, valid outputs.
run_task poker-draw poker-draw kaihand/poker_draw_0920_200
run_task install-ram install-ram kaihand/install_ram_0920_200
run_task bulb-screw bulb-screw kaihand/bulb_screw_0920_200
run_task pick-place pick-place kaihand/pick_place_0920_200
run_task whiteboard-wipe whiteboard-wipe kaihand/whiteboard_wipe_0920_200
