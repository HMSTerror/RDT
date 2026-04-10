#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon.yaml}"

# New stage-1 run that is currently at 20k and should resume to 50k.
: "${NEW_OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_stage1_20k}"
: "${NEW_RESUME_CHECKPOINT:=}"
: "${TARGET_MAX_STEPS:=50000}"

# Old baseline run used for the final 50k comparison.
: "${OLD_OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_50k}"

# Training settings.
: "${NUM_PROCESSES:=2}"
: "${MIXED_PRECISION:=bf16}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${EVAL_BATCH_SIZE:=16}"
: "${SAVE_STEPS:=10000}"
: "${EVAL_STEPS:=5000}"
: "${LOGGING_STEPS:=50}"

# Eval / compare settings.
: "${SPLIT:=test}"
: "${COMPARE_BATCH_SIZE:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${TOPK:=5,10,20}"
: "${GROUP_STRATEGY:=equal_items}"
: "${FREQUENCY_SOURCE_SPLIT:=train}"
: "${EXCLUDE_HISTORY_ITEMS:=1}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG_PATH="logs/genrec_stage1_resume_to50k_${RUN_TAG}.log"
NEW_EVAL_LOG_DIR="logs/genrec_stage1_new50k_raw_${RUN_TAG}"
OLD_EVAL_LOG_DIR="logs/genrec_stage1_old50k_raw_${RUN_TAG}"
COMPARE_DIR="logs/genrec_stage1_50k_compare"

mkdir -p "$(dirname "${TRAIN_LOG_PATH}")" "${NEW_EVAL_LOG_DIR}" "${OLD_EVAL_LOG_DIR}" "${COMPARE_DIR}"
echo "${RUN_TAG}" > "${COMPARE_DIR}/last_run_tag.txt"

find_latest_checkpoint_dir() {
  local output_dir="$1"
  mapfile -t checkpoint_dirs < <(find "${output_dir}" -maxdepth 1 -type d -name "checkpoint-*" | sort -V)
  if [[ "${#checkpoint_dirs[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${checkpoint_dirs[-1]}"
}

resolve_eval_checkpoint_dir() {
  local output_dir="$1"
  if [[ -f "${output_dir}/checkpoint-50000/pytorch_model.bin" ]]; then
    printf '%s\n' "${output_dir}/checkpoint-50000"
    return 0
  fi
  if [[ -f "${output_dir}/final/pytorch_model.bin" ]]; then
    printf '%s\n' "${output_dir}/final"
    return 0
  fi
  if latest_dir="$(find_latest_checkpoint_dir "${output_dir}")"; then
    printf '%s\n' "${latest_dir}"
    return 0
  fi
  return 1
}

run_raw_eval() {
  local checkpoint_dir="$1"
  local output_dir="$2"
  local log_dir="$3"

  CHECKPOINT="${checkpoint_dir}" \
  OUTPUT_DIR="${output_dir}" \
  SPLIT="${SPLIT}" \
  BATCH_SIZE="${COMPARE_BATCH_SIZE}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  TOPK="${TOPK}" \
  GROUP_STRATEGY="${GROUP_STRATEGY}" \
  FREQUENCY_SOURCE_SPLIT="${FREQUENCY_SOURCE_SPLIT}" \
  EXCLUDE_HISTORY_ITEMS="${EXCLUDE_HISTORY_ITEMS}" \
  POPULARITY_PENALTY="0" \
  LOG_DIR="${log_dir}" \
  bash scripts/run_genrec_hybrid_diffusion_eval.sh
}

if [[ -n "${NEW_RESUME_CHECKPOINT}" ]]; then
  RESUME_CHECKPOINT_DIR="${NEW_RESUME_CHECKPOINT}"
else
  if ! RESUME_CHECKPOINT_DIR="$(find_latest_checkpoint_dir "${NEW_OUTPUT_DIR}")"; then
    echo "No checkpoint-* directory found under ${NEW_OUTPUT_DIR}."
    echo "Set NEW_RESUME_CHECKPOINT manually or make sure the 20k run exists."
    exit 1
  fi
fi

echo "========== Resume New Stage-1 Training To 50k =========="
echo "new_output_dir     : ${NEW_OUTPUT_DIR}"
echo "resume_checkpoint  : ${RESUME_CHECKPOINT_DIR}"
echo "target_max_steps   : ${TARGET_MAX_STEPS}"
echo "save_steps         : ${SAVE_STEPS}"
echo "train_log          : ${TRAIN_LOG_PATH}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  scripts/train_genrec_hybrid_diffusion.py \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${NEW_OUTPUT_DIR}" \
  --resume_from_checkpoint "${RESUME_CHECKPOINT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_train_steps "${TARGET_MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --eval_steps "${EVAL_STEPS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  2>&1 | tee "${TRAIN_LOG_PATH}"

if ! NEW_COMPARE_CHECKPOINT="$(resolve_eval_checkpoint_dir "${NEW_OUTPUT_DIR}")"; then
  echo "Could not resolve a new-model checkpoint under ${NEW_OUTPUT_DIR} after training."
  exit 1
fi
if ! OLD_COMPARE_CHECKPOINT="$(resolve_eval_checkpoint_dir "${OLD_OUTPUT_DIR}")"; then
  echo "Could not resolve an old baseline checkpoint under ${OLD_OUTPUT_DIR}."
  exit 1
fi

echo "========== Raw Eval: New 50k =========="
echo "checkpoint: ${NEW_COMPARE_CHECKPOINT}"
run_raw_eval "${NEW_COMPARE_CHECKPOINT}" "${NEW_OUTPUT_DIR}" "${NEW_EVAL_LOG_DIR}"

echo "========== Raw Eval: Old 50k =========="
echo "checkpoint: ${OLD_COMPARE_CHECKPOINT}"
run_raw_eval "${OLD_COMPARE_CHECKPOINT}" "${OLD_OUTPUT_DIR}" "${OLD_EVAL_LOG_DIR}"

NEW_EVAL_LOG_DIR="${NEW_EVAL_LOG_DIR}" \
OLD_EVAL_LOG_DIR="${OLD_EVAL_LOG_DIR}" \
COMPARE_DIR="${COMPARE_DIR}" \
RUN_TAG="${RUN_TAG}" \
python - <<'PY'
import glob
import json
import os

METRICS = ["mean_rank", "hit@5", "ndcg@5", "hit@10", "ndcg@10", "hit@20", "ndcg@20"]
GROUPS = ["cold", "mid", "hot"]


def latest_json(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    if not files:
        raise SystemExit(f"no json found in {folder}")
    return files[-1]


def load_payload(folder):
    path = latest_json(folder)
    with open(path, "r", encoding="utf-8") as fp:
        return path, json.load(fp)


def fmt(value):
    return "NA" if value is None else f"{value:.6f}"


old_dir = os.environ["OLD_EVAL_LOG_DIR"]
new_dir = os.environ["NEW_EVAL_LOG_DIR"]
compare_dir = os.environ["COMPARE_DIR"]
run_tag = os.environ["RUN_TAG"]

old_path, old_data = load_payload(old_dir)
new_path, new_data = load_payload(new_dir)

lines = []
lines.append(
    "Comparison note: delta = new - old; mean_rank lower is better; all other metrics higher are better."
)
lines.append(f"RUN_TAG   : {run_tag}")
lines.append(f"OLD_JSON  : {old_path}")
lines.append(f"NEW_JSON  : {new_path}")
lines.append("")
lines.append("===== OVERALL =====")
for metric in METRICS:
    old_value = old_data["overall_metrics"].get(metric)
    new_value = new_data["overall_metrics"].get(metric)
    delta = None if old_value is None or new_value is None else new_value - old_value
    lines.append(
        f"{metric:>10} | old={fmt(old_value)} | new={fmt(new_value)} | delta={fmt(delta)}"
    )

for group in GROUPS:
    old_group = old_data["group_metrics"][group]
    new_group = new_data["group_metrics"][group]
    lines.append("")
    lines.append(f"===== GROUP: {group} =====")
    lines.append(
        f"sample_count | old={old_group.get('sample_count')} | new={new_group.get('sample_count')}"
    )
    for metric in METRICS:
        old_value = old_group.get(metric)
        new_value = new_group.get(metric)
        delta = None if old_value is None or new_value is None else new_value - old_value
        lines.append(
            f"{metric:>10} | old={fmt(old_value)} | new={fmt(new_value)} | delta={fmt(delta)}"
        )

report = "\n".join(lines)
latest_report = os.path.join(compare_dir, "compare_latest.txt")
dated_report = os.path.join(compare_dir, f"compare_{run_tag}.txt")

with open(latest_report, "w", encoding="utf-8") as fp:
    fp.write(report)
with open(dated_report, "w", encoding="utf-8") as fp:
    fp.write(report)

print(report)
print("")
print(f"compare_latest = {latest_report}")
print(f"compare_copy   = {dated_report}")
PY

