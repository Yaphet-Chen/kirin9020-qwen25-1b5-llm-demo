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
#    默认 = instruct 产线（实际部署形态）；base 用 EXPORT_YAML=model_info_base.yaml
yaml="${EXPORT_YAML:-model_info_instruct.yaml}"
sed "s|__ROOT__|$ROOT|g" "$yaml" > "npu_tuned_export/${yaml}"

# 3. 执行导出（直接用 venv 的 python，activate 已修复但绝对路径同样稳妥）
#    产物目录命名规则（由导出脚本按配置拼后缀，勿手工建目录）：
#    yaml 的 output_dir + "_embedding_out_no_output_pos" = 最终产物目录
#      base:     output_embedding_out_no_output_pos/
#      instruct: output_instruct_embedding_out_no_output_pos/
export CUDA_HOME="$ROOT/01_prepare/cuda_stub"
export PYTHONPATH="$DOPT:${PYTHONPATH:-}"

cd npu_tuned_export
"$VENV/bin/python" export_model_single_qwen2.py "$yaml" 2>&1 | tee "$ROOT/logs/export.log"

# 4. 产物目录（导出脚本按 yaml 的 output_dir 加后缀生成 *_embedding_out_no_output_pos）
#    OUT_SUBDIR 可覆盖（instruct 版为 output_instruct_embedding_out_no_output_pos，pipeline.sh 自动传）
OUTDIR="$ROOT/03_onnx_export/${OUT_SUBDIR:-output_instruct_embedding_out_no_output_pos}"
echo ""
echo "✅ 阶段③完成。产物目录: $OUTDIR"
ls -lah "$OUTDIR"/*.onnx "$OUTDIR"/*.pb 2>/dev/null
