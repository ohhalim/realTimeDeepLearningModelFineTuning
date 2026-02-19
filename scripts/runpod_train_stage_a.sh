#!/usr/bin/env bash
set -euo pipefail

MODE="all"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CONFIG_PATH="${CONFIG_PATH:-configs/roles/lead.yaml}"
ROLE="${ROLE:-lead}"
RAW_MIDI_DIR="${RAW_MIDI_DIR:-data/raw_midi}"
ROLES_DIR="${ROLES_DIR:-data/roles}"
LOG_FILE="${LOG_FILE:-train_stage_a.log}"
OVERWRITE_DATASET=0
TRANSPOSE_ALL_KEYS=1
NOHUP_TRAIN=1
EPOCHS=""
RESUME=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/runpod_train_stage_a.sh [options]

Options:
  --mode <setup|prepare|train|all>   Step to run (default: all)
  --python <python_bin>              Python executable (default: python3)
  --venv <path>                      Venv directory (default: .venv)
  --config <path>                    Training config (default: configs/roles/lead.yaml)
  --role <lead|accompaniment|call_response>
  --raw-midi-dir <path>              Raw MIDI input dir (default: data/raw_midi)
  --roles-dir <path>                 Role dataset root dir (default: data/roles)
  --epochs <n>                       Override epochs
  --resume <checkpoint_path>         Resume from checkpoint
  --log-file <path>                  Training log path (default: train_stage_a.log)
  --overwrite                        Remove and rebuild role dataset
  --no-transpose                     Disable all-keys transpose augmentation
  --no-nohup                         Run training in foreground
  -h, --help                         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --venv) VENV_DIR="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --role) ROLE="$2"; shift 2 ;;
    --raw-midi-dir) RAW_MIDI_DIR="$2"; shift 2 ;;
    --roles-dir) ROLES_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --overwrite) OVERWRITE_DATASET=1; shift ;;
    --no-transpose) TRANSPOSE_ALL_KEYS=0; shift ;;
    --no-nohup) NOHUP_TRAIN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

activate_venv() {
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
}

setup_env() {
  echo "==> [setup] Creating/using venv: ${VENV_DIR}"
  if [[ ! -d "${VENV_DIR}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  activate_venv
  python -m pip install --upgrade pip
  python -m pip install -r requirements-pytorch.txt

  echo "==> [setup] Dependency and GPU check"
  python - <<'PY'
import importlib
mods = ["torch", "yaml", "pretty_midi", "mido"]
for m in mods:
    importlib.import_module(m)
print("Dependencies OK:", ", ".join(mods))
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
}

prepare_dataset() {
  activate_venv
  echo "==> [prepare] Building role dataset for role=${ROLE}"

  cmd=(
    python scripts/prepare_role_dataset.py
    --input_dir "${RAW_MIDI_DIR}"
    --output_dir "${ROLES_DIR}"
    --role "${ROLE}"
    --conditioning_mode lower_register
  )
  if [[ "${TRANSPOSE_ALL_KEYS}" -eq 1 ]]; then
    cmd+=(--transpose_all_keys)
  fi
  if [[ "${OVERWRITE_DATASET}" -eq 1 ]]; then
    cmd+=(--overwrite)
  fi
  "${cmd[@]}"

  local role_dir="${ROLES_DIR}/${ROLE}"
  local sample_count
  sample_count=$(find "${role_dir}" -type f -name "target.mid" | wc -l | tr -d ' ')
  if [[ "${sample_count}" -eq 0 ]]; then
    echo "ERROR: No role samples generated under ${role_dir}" >&2
    exit 1
  fi
  echo "==> [prepare] role samples: ${sample_count}"
}

start_training() {
  activate_venv
  local role_dir="${ROLES_DIR}/${ROLE}"
  if [[ ! -d "${role_dir}" ]]; then
    echo "ERROR: role dataset not found: ${role_dir}" >&2
    exit 1
  fi

  cmd=(
    python scripts/train_pytorch_transformer.py
    --config "${CONFIG_PATH}"
    --use_role_dataset
    --role "${ROLE}"
    --midi_dir "${role_dir}"
  )
  if [[ -n "${EPOCHS}" ]]; then
    cmd+=(--epochs "${EPOCHS}")
  fi
  if [[ -n "${RESUME}" ]]; then
    cmd+=(--resume "${RESUME}")
  fi

  echo "==> [train] Command: ${cmd[*]}"
  if [[ "${NOHUP_TRAIN}" -eq 1 ]]; then
    nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
    local pid=$!
    echo "==> [train] Started in background"
    echo "PID: ${pid}"
    echo "Log: ${LOG_FILE}"
    echo "Monitor: tail -f ${LOG_FILE}"
  else
    "${cmd[@]}"
  fi
}

case "${MODE}" in
  setup)
    setup_env
    ;;
  prepare)
    setup_env
    prepare_dataset
    ;;
  train)
    setup_env
    start_training
    ;;
  all)
    setup_env
    prepare_dataset
    start_training
    ;;
  *)
    echo "Invalid --mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

