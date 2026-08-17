#!/usr/bin/env bash
# 阶段③：NPU 亲和 ONNX 导出（GPU）
# 用法: bash 03_onnx_export/export.sh
# 前置: 阶段②产出 fake_quant_weight.pth + dopt_config.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/03_onnx_export"

VENV="$ROOT/01_prepare/venv"
DOPT="$ROOT/01_prepare/tools/tools_dopt/dopt_pytorch_py3"
SRC="$ROOT/../cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Model/npu_tuned_export"

# 1. 拷贝 npu_tuned_export 工程（若未拷过）
if [ ! -d npu_tuned_export ]; then
  cp -r "$SRC" npu_tuned_export
fi

# 2. 用绝对路径填充 yaml（占位符 __ROOT__ → 实际 ROOT）
yaml="model_info_target.yaml"
sed "s|__ROOT__|$ROOT|g" "$yaml" > npu_tuned_export/model_info_target.yaml

# 3. 执行导出
source "$VENV/bin/activate"
export CUDA_HOME="$ROOT/01_prepare/cuda_stub"
export PYTHONPATH="$DOPT:${PYTHONPATH:-}"
mkdir -p dump output

cd npu_tuned_export
python export_model_single_qwen2.py model_info_target.yaml 2>&1 | tee "$ROOT/logs/export.log"

# 4. 把产物拷到 03_onnx_export/output（导出脚本会生成 output_embedding_out_no_output_pos）
OUTDIR="$ROOT/03_onnx_export/output_embedding_out_no_output_pos"
echo ""
echo "✅ 阶段③完成。产物目录: $OUTDIR"
ls -lah "$OUTDIR"/*.onnx "$OUTDIR"/*.pb 2>/dev/null
