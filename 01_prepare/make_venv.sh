#!/usr/bin/env bash
# 创建 Python 虚拟环境（uv）
# 关键: 本机 RTX 5090 是 Blackwell (sm_120)，requirements.txt 的 torch 2.4 只支持到 sm_90，
#       必须用 torch 2.8.0+cu128（含 sm_120）才能跑量化。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/01_prepare"

# 用 uv 建 venv（若系统无 uv: curl -LsSf https://astral.sh/uv/install.sh | sh）
uv venv --python 3.10 venv
source venv/bin/activate

uv pip install \
  "torch==2.8.0" \
  "transformers==4.51.0" \
  "accelerate==1.3.0" \
  "onnx==1.15.0" \
  "onnxsim==0.4.35" \
  "onnxruntime==1.16.3" \
  "PyYAML==6.0.1" \
  "sentencepiece==0.2.0" \
  "safetensors==0.5.2" \
  "huggingface_hub==0.30.2" \
  "numpy==1.26.4" \
  "protobuf==3.20.2" \
  "datasets==2.19.1" \
  "einops==0.8.0" \
  "peft==0.12.0" \
  "deepspeed==0.15.4" \
  "patchelf"

# 验证
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('arch', torch.cuda.get_arch_list())"
deactivate
echo "✅ venv 建好: 01_prepare/venv"
