#!/usr/bin/env bash
# 阶段④：ONNX → OMC 转换（kirin9020）
# 用法: bash 04_omc_convert/convert.sh
# 前置: 阶段②quant_params_file + 阶段③model.onnx/model.pb
# 关键发现: DDK 自带 omg 所需的全部算子库，不需要额外安装 CANN toolkit
# 路径可用环境变量覆盖: QUANT_DIR / ONNX_DIR / OUTPUT_PREFIX
#（instruct 产线由 pipeline.sh 自动传 Qwen25_1b5_Instruct_kirin9020 等值）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/04_omc_convert"

DDK="$ROOT/01_prepare/tools"
# 默认 = instruct 产线（实际部署形态）；base 显式传：
#   QUANT_DIR=.../qwen25_1b5_base_9020 ONNX_DIR=.../output_embedding_out_no_output_pos \
#   OUTPUT_PREFIX=./Qwen25_1b5_Base_kirin9020 bash convert.sh
QUANT_DIR="${QUANT_DIR:-$ROOT/02_quant/qwen25_1b5_instruct_9020}"         # 量化工程（含 train_output/quant_params_file）
ONNX_DIR="${ONNX_DIR:-$ROOT/03_onnx_export/output_instruct_embedding_out_no_output_pos}"
QUANT_PARAMS="$QUANT_DIR/train_output/quant_params_file"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-./Qwen25_1b5_Instruct_kirin9020}"

# 准备输入（先检查，缺产物时给出明确提示，避免现场对着 omg 报错排查）
for f in "$ONNX_DIR/model.onnx" "$ONNX_DIR/model.pb" "$QUANT_PARAMS"; do
  [ -f "$f" ] || { echo "❌ 缺少 $f —— 先跑 02_quant/run.sh stage3 再跑 03_onnx_export/export.sh"; exit 1; }
done
cp "$ONNX_DIR/model.onnx" ./model.onnx
cp "$ONNX_DIR/model.pb" ./model.pb
cp "$QUANT_PARAMS" ./quant_params_file
# 注：本目录的 model.onnx/.pb/quant_params_file 是每次转换的工作副本（源头在 03_onnx_export
# 与 02_quant/<testcase>/），转换完成后可删（清理协议已按此执行，2026-08-17）。
# 历史：model128.onnx/.pb 与 omg_t16/t128/3tier/e128.log 多档实验失败残留已清理
#（档位被 quant_params_file 锁定 {1,64}，见 QUANTIZATION.md §三）。

# 构造 input_shape / input_type / output_type（28 层）
LAYERS=28; HIDDEN=1536; KVLEN=2048; KVHEAD=2; HEADDIM=128
shape="input_embed:1,-1,${HIDDEN};attention_mask:1,1,-1,${KVLEN};position_ids:1,-1"
itype=""; otype="lm_logits:FP32"
for i in $(seq 0 $((LAYERS-1))); do
  shape="${shape};past_key_in${i}:${KVLEN},${KVHEAD},1,${HEADDIM};past_value_in${i}:${KVLEN},${KVHEAD},1,${HEADDIM}"
  itype="${itype};past_key_in${i}:FP16;past_value_in${i}:FP16"
  otype="${otype};past_key${i}:FP16;past_value${i}:FP16"
done
shape="${shape};new_kv_cache_pos:-1;embed_scales:1,-1,1"
input_type_val="${itype#;}"

rm -rf "$OUTPUT_PREFIX" "${OUTPUT_PREFIX}.omc"

# 关键: 用 env -i 隔离环境，只给 DDK 的 lib（避免任何残留 CANN/其他环境干扰）
# omg 自带 interpreter 已用 patchelf 改为系统 ld
echo "执行 omg 转换（platform=kirin9020, target=omc）..."
env -i HOME="$HOME" PATH="/usr/bin:/bin" \
  LD_LIBRARY_PATH="$DDK/tools_omg/master/lib64:$DDK/platform/kirin9020/lib64" \
  SOC_VERSION=kirin9020 \
  bash "$DDK/tools_omg/omg" \
    --model ./model.onnx --framework 5 \
    --output "$OUTPUT_PREFIX" \
    --input_shape="$shape" \
    --dynamic_dims="1,1,1,1,1;64,64,64,64,64" \
    --input_type="$input_type_val" \
    --output_type="$otype" \
    --compress_conf ./quant_params_file \
    --save_weights_as_external_data=true \
    --platform=kirin9020 \
    --target=omc 2>&1 | tee "$ROOT/logs/omc_convert.log"

echo ""
echo "✅ 阶段④完成。产物:"
ls -lah "$OUTPUT_PREFIX"/*.omc "$OUTPUT_PREFIX"/SubGraph_0.weight 2>/dev/null
