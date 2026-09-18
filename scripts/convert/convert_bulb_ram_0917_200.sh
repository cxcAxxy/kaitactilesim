#!/usr/bin/env bash
set -euo pipefail

SIM_MERGED_ROOT=/cpfs_infra/user/chenxianchi/code/sim_code_merged_20260917
SIM_PYTHON=/cpfs_infra/user/chenxianchi/miniconda3/envs/galaxea_g05/bin/python
OPENPI_ROOT=/cpfs_infra/user/chenxianchi/code/openpi
CONVERSION_STAGE=/cpfs_infra/user/chenxianchi/.conversion_staging
CONVERSION_LOG_ROOT=/cpfs_infra/user/chenxianchi/conversion_logs/0917_200

mkdir -p "${CONVERSION_STAGE}" "${CONVERSION_LOG_ROOT}"
cd "${SIM_MERGED_ROOT}"

run_conversion() {
  local task=$1
  local format=$2
  local input_dir=$3
  local output_dir=$4
  local dataset_name=$5
  local log_path="${CONVERSION_LOG_ROOT}/${task}_${format}.log"

  PYTHONPATH="${SIM_MERGED_ROOT}/src" "${SIM_PYTHON}" \
    scripts/convert/convert.py \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --format "${format}" \
    --task "${task}" \
    --cameras head left_wrist right_wrist \
    --workers 4 \
    --expected-episodes 200 \
    --staging-root "${CONVERSION_STAGE}" \
    --dataset-name "${dataset_name}" \
    --openpi-root "${OPENPI_ROOT}" \
    --pi05-python "${OPENPI_ROOT}/.venv-pi05/bin/python" \
    2>&1 | tee "${log_path}"
}

# Sequential publication avoids four large three-camera jobs competing for NAS I/O.
run_conversion \
  bulb-screw egosteer \
  /nas/chenxianchi/datasets/sim/buld_screw/raw/0917_200 \
  /nas/chenxianchi/datasets/sim/buld_screw/egosteer/0917_200 \
  bulb_screw_0917_200

run_conversion \
  bulb-screw pi05 \
  /nas/chenxianchi/datasets/sim/buld_screw/raw/0917_200 \
  /nas/chenxianchi/datasets/sim/buld_screw/pi05/0917_200 \
  bulb_screw_0917_200

run_conversion \
  install-ram egosteer \
  /nas/chenxianchi/datasets/sim/install_ram/raw/0917_200 \
  /nas/chenxianchi/datasets/sim/install_ram/egosteer/0917_200 \
  install_ram_0917_200

run_conversion \
  install-ram pi05 \
  /nas/chenxianchi/datasets/sim/install_ram/raw/0917_200 \
  /nas/chenxianchi/datasets/sim/install_ram/pi05/0917_200 \
  install_ram_0917_200
