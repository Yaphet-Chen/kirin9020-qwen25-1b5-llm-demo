#!/usr/bin/env python3
"""量化仿真模型加载（eval_ppl / chat_test / device_compare 三处共用的单一实现）。

原先三个脚本各自持有一份 get_quanted_model/load_quant 副本，现归一于此（AGENTS.md 规则 2/3）。
env 自举（CUDA_HOME stub + dopt PYTHONPATH）在 import 时完成，脚本可独立运行。
"""
import os, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_HOME", str(ROOT / "01_prepare" / "cuda_stub"))  # 过 deepspeed 检查
sys.path.insert(0, str(ROOT / "01_prepare" / "tools" / "tools_dopt" / "dopt_pytorch_py3"))

import torch
from transformers import AutoModelForCausalLM


def load_quant(model_path, cfg, ckpt, emb=None, device="cuda"):
    """加载 dopt 量化仿真模型（权重 fp32——fp16 在量化算子上长序列溢出 nan）。
    emb 不为空时改用端侧同款 int8 embedding——必须在两个时机替换
    （首版教训：只事后替换无效，optimize_model 会接管/复制模块权重）：
      ① optimize_model 之前 → dopt 接管的就是替换后的权重；
      ② load_state_dict 之后 → 防 ckpt 把 embedding 覆盖回原始值。"""
    from dopt.dopt_lm.do_opt import optimize_model, set_quant_state
    from dopt.dopt_lm.train import set_calibrate_state
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map=device)
    if emb is not None:
        with torch.no_grad():
            base.model.embed_tokens.weight.copy_(emb.to(base.model.embed_tokens.weight.device))
    m = optimize_model(base, str(cfg))
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    if emb is not None:
        with torch.no_grad():
            m.model.embed_tokens.weight.copy_(emb.to(m.model.embed_tokens.weight.device))
    set_quant_state(m, weight_state=True, input_state=True)
    set_calibrate_state(m, False)
    m.eval()
    return m
