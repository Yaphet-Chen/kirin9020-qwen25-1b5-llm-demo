#!/usr/bin/env bash
# 阶段⑤：汇集端侧集成 7 文件（base 默认；instruct 由 pipeline.sh 传环境变量复用本脚本）
# 位于仓库根——服务器侧产线脚本，不属于交付目录（交付目录只放设备侧要用的东西）。
# 用法:
#   bash pack.sh                                            # base（默认值）
#   OMC_NAME=... EMB_STEM=... OMC_DIR=... DEST_DIR=... bash pack.sh
#      （pipeline.sh pack 自动传 instruct 参数）
# 7 文件: omc + SubGraph_0.weight + embedding_weights + embedding_dequant_scale
#        + tokenizer.json + context.json + executor.json
# context.json / executor.json / 推送脚本不放本脚本拷贝——每个交付目录自带（base 与
# instruct 的 sampler 参数、omc/embedding 文件名不同：*_Base_* / *_Instruct_*）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# 环境变量覆盖（pipeline.sh 传；单独跑时默认 = base 交付版）
OMC_NAME="${OMC_NAME:-Qwen25_1b5_Base_kirin9020}"
OMC_DIR="${OMC_DIR:-$ROOT/04_omc_convert/Qwen25_1b5_Base_kirin9020}"
EMB_DIR="${EMB_DIR:-$ROOT/03_onnx_export/output_embedding_out_no_output_pos}"
EMB_SRC_STEM="${EMB_SRC_STEM:-model_64_2048}"          # 导出侧名（= onnx名_prefill_kvlen，两条产线相同）
EMB_STEM="${EMB_STEM:-model_base_64_2048}"             # 端侧名（带 Base/Instruct 标记，与 executor.json 一致）
MODEL_DIR="${MODEL_DIR:-$ROOT/01_prepare/models/Qwen2.5-1.5B}"
DEST_DIR="${DEST_DIR:-$ROOT/05_device_files_base}"

for f in "$OMC_DIR/${OMC_NAME}.omc" "$OMC_DIR/SubGraph_0.weight" \
         "$EMB_DIR/${EMB_SRC_STEM}.embedding_weights" "$EMB_DIR/${EMB_SRC_STEM}.embedding_dequant_scale" \
         "$MODEL_DIR/tokenizer.json"; do
  [ -f "$f" ] || { echo "❌ 缺少 $f（对照 REPRODUCE.md 验证清单检查前置阶段）"; exit 1; }
done

mkdir -p "$DEST_DIR"
cp "$OMC_DIR/${OMC_NAME}.omc" "$DEST_DIR/"
cp "$OMC_DIR/SubGraph_0.weight" "$DEST_DIR/"
cp "$EMB_DIR/${EMB_SRC_STEM}.embedding_weights"        "$DEST_DIR/${EMB_STEM}.embedding_weights"
cp "$EMB_DIR/${EMB_SRC_STEM}.embedding_dequant_scale"  "$DEST_DIR/${EMB_STEM}.embedding_dequant_scale"
cp "$MODEL_DIR/tokenizer.json" "$DEST_DIR/"

# 校验关键产物体积（传输截断是上机头号故障，REPRODUCE.md 路线B 第3步）
W=$(stat -c%s "$DEST_DIR/SubGraph_0.weight")
echo "✅ [${OMC_NAME}] 端侧 7 文件就绪于 $DEST_DIR（SubGraph_0.weight = ${W} 字节）"
ls -lah "$DEST_DIR" | grep -E 'omc|SubGraph|embedding|tokenizer|context|executor'
echo ""
echo "推送设备: 把 $DEST_DIR 拷到连手机的 Windows 机，运行其中的 push_to_device_next.bat"
echo "  （或 Git Bash: bash push_to_device_next.sh）"
