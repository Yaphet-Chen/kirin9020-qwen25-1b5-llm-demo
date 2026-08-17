#!/usr/bin/env bash
# 阶段⑤：汇集端侧集成 7 文件到 05_device_files/
# 用法: bash 05_device_files/pack.sh
# 前置: 阶段②③④全部完成
# 7 文件: omc + SubGraph_0.weight + embedding_weights + embedding_dequant_scale
#        + tokenizer.json + context.json + executor.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/05_device_files"

OMC_DIR="$ROOT/04_omc_convert/Qwen25_1b5_kirin9020"
EMB_DIR="$ROOT/03_onnx_export/output_embedding_out_no_output_pos"
MODEL_DIR="$ROOT/01_prepare/models/Qwen2.5-1.5B"

# 交付版 embedding 为 model_64_2048（文件名 = prefill档_kv上限；档位锁定 {1,64} 不可改）
for f in "$OMC_DIR/Qwen25_1b5_kirin9020.omc" "$OMC_DIR/SubGraph_0.weight" \
         "$EMB_DIR/model_64_2048.embedding_weights" "$EMB_DIR/model_64_2048.embedding_dequant_scale" \
         "$MODEL_DIR/tokenizer.json"; do
  [ -f "$f" ] || { echo "❌ 缺少 $f（对照 REPRODUCE.md 验证清单检查前置阶段）"; exit 1; }
done

cp "$OMC_DIR/Qwen25_1b5_kirin9020.omc" .
cp "$OMC_DIR/SubGraph_0.weight" .
cp "$EMB_DIR/model_64_2048.embedding_weights" .
cp "$EMB_DIR/model_64_2048.embedding_dequant_scale" .
cp "$MODEL_DIR/tokenizer.json" .
# context.json / executor.json 已在本目录

echo "✅ 端侧集成 7 文件:"
ls -lah
echo ""
echo "推送设备（路线B/NEXT 真机，已验证）: 在连手机的 Windows 机器上运行 push_to_device_next.bat"
echo "  或 Git Bash: bash push_to_device_next.sh"
echo "  executor.json/context_next.json 已含沙箱路径, 详见 NEXT_端侧测试手册.md"
