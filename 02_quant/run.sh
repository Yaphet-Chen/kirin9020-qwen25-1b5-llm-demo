#!/usr/bin/env bash
# 阶段②：三段式量化（GPU）
# 用法:
#   bash 02_quant/run.sh stage1   # 第一次：生成 dopt_config.json
#   （编辑 dopt_config.json 量化策略，见 02_quant/edit_dopt_config.py）
#   bash 02_quant/run.sh stage1   # 第二次：权重量化
#   bash 02_quant/run.sh stage2   # 激活量化
#   bash 02_quant/run.sh stage3   # 量化参数提取
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/02_quant"

# ===== 路径（按实际调整）=====
qlibs="$ROOT/01_prepare/tools/tools_dopt/dopt_pytorch_py3"
model_path="$ROOT/01_prepare/models/Qwen2.5-1.5B"
testcase="${TESTCASE:-qwen25_1b5_9020}"
cfg="${CFG:-config.yaml}"

export WANDB_DISABLED=true
export HF_DATASETS_OFFLINE=0
export PYTHONPATH="${qlibs}:${PYTHONPATH:-}"
export DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME="$ROOT/01_prepare/cuda_stub"   # stub，让 deepspeed import 通过

output_dir="./${testcase}/train_output"
mkdir -p "$output_dir"
cp "$cfg" "$output_dir/"

quant_stage="${1:?用法: bash run.sh stage1|stage2|stage3}"
dopt_config="./${testcase}/dopt_config.json"
RUN_FILE="${qlibs}/dopt/dopt_lm/opt_main.py"

# 防呆：优先用工作区 venv 的 python，忘了 activate 也能跑
PY="$ROOT/01_prepare/venv/bin/python"
[ -x "$PY" ] || PY="python"

"$PY" -u "$RUN_FILE" \
  --model-path "$model_path" \
  --dopt-config "$dopt_config" \
  --optimize-config "$cfg" \
  --quant-stage "$quant_stage" \
  --block-size 128 \
  --output-dir "$output_dir" 2>&1 | tee "$output_dir/logs.log"
