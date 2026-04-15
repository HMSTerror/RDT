#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon.yaml}"
: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_50k}"
: "${CHECKPOINT:=}"
: "${SPLIT:=test}"
: "${NUM_PROCESSES:=1}"
: "${MIXED_PRECISION:=no}"
: "${BATCH_SIZE:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${TOPK:=5,10,20}"
: "${GROUP_STRATEGY:=equal_items}"
: "${FREQUENCY_SOURCE_SPLIT:=train}"
: "${EXCLUDE_HISTORY_ITEMS:=1}"
: "${MAX_EVAL_BATCHES:=0}"
: "${PRINT_EVERY:=20}"
: "${EVAL_SEED:=42}"
: "${ABLATION_LOG_ROOT:=logs/genrec_ablation_suite}"
: "${VARIANT_SPECS:=full=;no_text=text;no_image=image;no_cf=cf}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ABLATION_LOG_ROOT}/${STAMP}"
mkdir -p "${RUN_ROOT}"

variant_names=()

run_one() {
  local tag="$1"
  local occlude="$2"
  local log_dir="${RUN_ROOT}/${tag}"

  echo "========== Ablation: ${tag} =========="
  CHECKPOINT="${CHECKPOINT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  MIXED_PRECISION="${MIXED_PRECISION}" \
  SPLIT="${SPLIT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  TOPK="${TOPK}" \
  GROUP_STRATEGY="${GROUP_STRATEGY}" \
  FREQUENCY_SOURCE_SPLIT="${FREQUENCY_SOURCE_SPLIT}" \
  EXCLUDE_HISTORY_ITEMS="${EXCLUDE_HISTORY_ITEMS}" \
  MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES}" \
  PRINT_EVERY="${PRINT_EVERY}" \
  EVAL_SEED="${EVAL_SEED}" \
  OCCLUDE_MODALITIES="${occlude}" \
  LOG_DIR="${log_dir}" \
  bash scripts/run_genrec_hybrid_diffusion_eval.sh
}

IFS=';' read -r -a variant_specs <<< "${VARIANT_SPECS}"
for spec in "${variant_specs[@]}"; do
  [[ -n "${spec}" ]] || continue
  tag="${spec%%=*}"
  occlude="${spec#*=}"
  variant_names+=("${tag}")
  run_one "${tag}" "${occlude}"
done

VARIANT_NAMES_CSV="$(IFS=,; echo "${variant_names[*]}")"

RUN_ROOT="${RUN_ROOT}" \
VARIANT_NAMES_CSV="${VARIANT_NAMES_CSV}" \
python - <<'PY'
import json
import os
from pathlib import Path

metric_keys = ["mean_rank", "hit@10", "ndcg@10", "hit@20", "ndcg@20"]
group_metric_keys = ["hit@20", "ndcg@20"]

run_root = Path(os.environ["RUN_ROOT"])
variant_names = [name for name in os.environ["VARIANT_NAMES_CSV"].split(",") if name]


def latest_json(folder: Path) -> Path:
    files = sorted(folder.glob("*.json"))
    if not files:
        raise SystemExit(f"no json found under {folder}")
    return files[-1]


payloads = {}
for variant in variant_names:
    path = latest_json(run_root / variant)
    with open(path, "r", encoding="utf-8") as fp:
        payloads[variant] = (path, json.load(fp))

base_variant = "full" if "full" in payloads else variant_names[0]
base_metrics = payloads[base_variant][1]["overall_metrics"]

lines = []
lines.append(
    f"Ablation note: delta = variant - {base_variant}; mean_rank lower is better; others higher are better."
)
lines.append(f"RUN_ROOT: {run_root}")
lines.append("")
lines.append("===== OVERALL =====")
for variant in variant_names:
    path, payload = payloads[variant]
    metrics = payload["overall_metrics"]
    lines.append(f"[{variant}] {path}")
    for key in metric_keys:
        value = metrics.get(key)
        base = base_metrics.get(key)
        delta = None if value is None or base is None else value - base
        lines.append(f"  {key}: value={value:.6f} delta_vs_{base_variant}={delta:.6f}")
    lines.append("")

lines.append("===== GROUP HIT@20 / NDCG@20 =====")
for group_name in ["cold", "mid", "hot"]:
    lines.append(f"[group={group_name}]")
    base_group = payloads[base_variant][1]["group_metrics"][group_name]
    for variant in variant_names:
        group_metrics = payloads[variant][1]["group_metrics"][group_name]
        hit20 = group_metrics.get("hit@20")
        ndcg20 = group_metrics.get("ndcg@20")
        hit_delta = hit20 - base_group.get("hit@20")
        ndcg_delta = ndcg20 - base_group.get("ndcg@20")
        lines.append(
            f"  {variant}: hit@20={hit20:.6f} ({hit_delta:+.6f}), "
            f"ndcg@20={ndcg20:.6f} ({ndcg_delta:+.6f})"
        )
    lines.append("")

summary = "\n".join(lines).rstrip() + "\n"
summary_path = run_root / "compare_latest.txt"
with open(summary_path, "w", encoding="utf-8") as fp:
    fp.write(summary)

print(summary)
print(f"saved_compare = {summary_path}")
PY

echo "========== Ablation suite done =========="
echo "run_root = ${RUN_ROOT}"
echo "summary  = ${RUN_ROOT}/compare_latest.txt"
