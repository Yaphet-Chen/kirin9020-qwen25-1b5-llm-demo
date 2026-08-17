#!/usr/bin/env bash
# 阶段①：准备工作 —— 解压 DDK + 组装 tools + 下载模型 + patchelf 修复 omg
# 用法: bash 01_prepare/prepare.sh
# 依赖: dependencies/DDK-tools-next-6.0.1.0.zip, dependencies/kirin9020-plugin-next-6.0.1.0.zip
set -euo pipefail

# 项目根目录（脚本可被任意位置调用）
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEPS="$ROOT/../dependencies"          # 依赖包目录（按实际位置调整）
DDK_ZIP="$DEPS/DDK-tools-next-6.0.1.0.zip"
PLG_ZIP="$DEPS/kirin9020-plugin-next-6.0.1.0.zip"

echo "===== [1/5] 解压 DDK-tools ====="
rm -rf 01_prepare/ddk_extracted 01_prepare/plugin_extracted
unzip -q "$DDK_ZIP" -d 01_prepare/ddk_extracted
unzip -q "$PLG_ZIP" -d 01_prepare/plugin_extracted

echo "===== [2/5] 组装 tools 树（插件放入 platform）====="
cp -r 01_prepare/plugin_extracted/kirin9020 01_prepare/ddk_extracted/tools/platform/
# 实体拷贝到 01_prepare/tools（软链会导致 omg 不可执行）
rm -rf 01_prepare/tools
cp -r 01_prepare/ddk_extracted/tools 01_prepare/tools
echo "tools 树:"; ls 01_prepare/tools/

echo "===== [3/5] 下载两个源模型（base + instruct，各 ~3G，已存在则跳过）====="
mkdir -p 01_prepare/models
for M in Qwen2.5-1.5B Qwen2.5-1.5B-Instruct; do
  if [ ! -f "01_prepare/models/$M/model.safetensors" ]; then
    python -c "from huggingface_hub import snapshot_download as s; \
      s('Qwen/$M', local_dir='01_prepare/models/$M')"
  else
    echo "$M 已存在，跳过"
  fi
done

echo "===== [4/5] patchelf 修复 omg interpreter ====="
# omg 二进制 interpreter 硬编码 /tmp/ld-linux-x86-64-2.35.so.2（跨用户软链 EACCES），
# 改成系统 ld。需要先装 patchelf（uv pip install patchelf）
if ! 01_prepare/venv/bin/patchelf --version >/dev/null 2>&1; then
  echo "请先在 venv 里装 patchelf: uv pip install patchelf （见 REPRODUCE.md 阶段①）"
fi
01_prepare/venv/bin/patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 \
  01_prepare/tools/tools_omg/master/omg
chmod +x 01_prepare/tools/tools_omg/master/omg
echo "omg interpreter 已修复"

echo "===== [5/5] stub CUDA_HOME（让 deepspeed import 通过，无 nvcc 环境）====="
mkdir -p 01_prepare/cuda_stub/bin
cat > 01_prepare/cuda_stub/bin/nvcc <<'NVCC'
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver (stub)"
echo "Cuda compilation tools, release 12.8, V12.8.93"
NVCC
chmod +x 01_prepare/cuda_stub/bin/nvcc

echo ""
echo "✅ 阶段①完成。产物: 01_prepare/{tools, models, cuda_stub}, 01_prepare/venv (需先建)"
