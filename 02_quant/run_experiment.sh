#!/usr/bin/env bash
# 单个优化试验：生成 config → 三段式量化 → 评测 PPL
# 用法: bash run_experiment.sh <name> <group_size> <samples> <kd_enable> [epochs] [edit_flags...]
# 例如: bash run_experiment.sh kd_g64_s1024 64 1024 true 5 --keep-lm-head-fp
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/02_quant"

PY="$ROOT/01_prepare/venv/bin/python"
export PYTHONPATH="$ROOT/02_quant/_autopatch:$ROOT/01_prepare/tools/tools_dopt/dopt_pytorch_py3:${PYTHONPATH:-}"
export CUDA_HOME="$ROOT/01_prepare/cuda_stub"
export PATH="$ROOT/01_prepare/venv/bin:$PATH"

NAME="$1"; GROUP="$2"; SAMPLES="$3"; KD="$4"; EPOCHS="${5:-1}"; EDIT_FLAGS="${@:6}"
TESTCASE="exp_${NAME}"
CFG="${NAME}_config.yaml"
MODEL="$ROOT/01_prepare/models/Qwen2.5-1.5B"

echo "============================================================"
echo "试验 ${NAME}: group=${GROUP} samples=${SAMPLES} kd=${KD} epochs=${EPOCHS} ${EDIT_FLAGS}"
echo "============================================================"

# KD 块：enable=true 时加 teacher_config_path（自蒸馏，关键！不加则 KD 不训练）
KD_BLOCK=""
if [ "$KD" = "true" ]; then
  KD_BLOCK="  teacher:
    teacher_config_path: ${MODEL}
    teacher_model: ${MODEL}"
fi

# 1. 生成 config.yaml
cat > "$CFG" <<YAML
kd:
  enable: ${KD}
  loss: mse
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  weight_decay: 0.0
  warmup_steps: 10
  num_epochs: ${EPOCHS}
  learning_rate: 1.0e-4
  eval_step: 1
  logging_step: 10
  lr_scheduler_type: cosine
  trainable_keys:
    - quant_alpha
    - norm
  no_split_module_classes:
    - Qwen2DecoderLayer
${KD_BLOCK}

dataset:
  train_files: ${TRAIN_FILES:-wikitext2}
  train_samples: ${SAMPLES}
  ptq_samples: ${SAMPLES}

extra_training_config:
  fp16: False

cutoff_len: ${CUTOFF:-128}
num_samples: ${SAMPLES}
quant_param_2: False
embedding_separate: True
lm_head_size:
YAML

# 2. 三段式量化
export TESTCASE CFG
echo "--- stage1 (genconfig) ---"
bash run.sh stage1 2>&1 | tail -1
echo "--- edit dopt_config (group=${GROUP} ${EDIT_FLAGS}) ---"
$PY edit_dopt_config.py "${TESTCASE}/dopt_config.json" "${GROUP}" ${EDIT_FLAGS}
echo "--- stage1 (weight) ---"
bash run.sh stage1 2>&1 | tail -1
echo "--- stage2 (act) ---"
bash run.sh stage2 2>&1 | tail -1
echo "--- stage3 (extract) ---"
bash run.sh stage3 2>&1 | tail -1

# 3. 评测
echo "--- eval PPL ---"
$PY eval_ppl.py "$MODEL" "${TESTCASE}/dopt_config.json" \
  "${TESTCASE}/train_output/trained.pth" --tag "${NAME}" --n_samples 8192 2>&1 \
  | grep -E "FP-|QUANT-|DELTA-|tokens"

echo "============================================================"
echo "试验 ${NAME} 完成"
echo "============================================================"
