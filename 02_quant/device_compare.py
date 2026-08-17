#!/usr/bin/env python3
"""
端侧部署配置下的 FP vs 量化模型续写/对话对比（严格复刻 context.json）。

采样协议 = 05_device_files/context.json 逐项:
  seed=99(每个 prompt 前重置, FP/QUANT 消耗完全相同的随机数)
  repetition_penalty=1.2(logits 除法) -> temperature=0.6 -> top-k=16 -> top-p=0.95(核采样)
FP 用 bf16(本机健康路径), 量化仿真用 fp32(dopt 量化算子在 fp16 下溢出 nan)。

用法(自包含, 无需手工 export 环境变量):
  python device_compare.py                          # 默认: g128+中文校准版, 内置续写 prompt
  python device_compare.py --n 60                   # 短输出快速验证
  python device_compare.py --chat                   # 追加 chat-template 对话测试(base 模型仅看退化程度)
  python device_compare.py \
      --config qwen25_1b5_9020/dopt_config.json \
      --ckpt   qwen25_1b5_9020/train_output/trained.pth   # 换 g64+wiki 校准版对比
可复现性: 同机同 torch 版本下, 量化侧(fp32)输出与 logs/g128zh_device_compare.log 逐字一致。
"""
import os, sys, argparse, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("CUDA_HOME", str(ROOT / "01_prepare" / "cuda_stub"))  # 过 deepspeed 检查
sys.path.insert(0, str(ROOT / "01_prepare" / "tools" / "tools_dopt" / "dopt_pytorch_py3"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = ROOT / "01_prepare" / "models" / "Qwen2.5-1.5B"
# 端侧 context.json 参数（改这里保持与部署一致）
SEED, TOP_K, TOP_P, TEMP, REP = 99, 16, 0.95, 0.6, 1.2

CONT_PROMPTS = [
    "长城是中国古代的伟大工程，", "秋天到了，", "人工智能的发展，",
    "The theory of relativity states that", "Python is a programming language that",
]
CHAT_PROMPTS = ["你好，请介绍一下你自己。", "请用中文解释什么是模型量化。"]


def load_quant(cfg, ckpt):
    from dopt.dopt_lm.do_opt import optimize_model, set_quant_state
    from dopt.dopt_lm.train import set_calibrate_state
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, device_map=DEV)
    m = optimize_model(base, str(cfg))
    m.load_state_dict(torch.load(str(ckpt), map_location="cpu"), strict=True)
    set_quant_state(m, weight_state=True, input_state=True)
    set_calibrate_state(m, False)
    m.eval()
    return m


@torch.no_grad()
def gen(model, tok, prompt, n, chat=False):
    torch.manual_seed(SEED)                      # 每 prompt 重置 → 两侧同随机数
    if chat:
        msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos, toks = tok.eos_token_id, []
    for _ in range(n):
        logits = model(ids).logits[0, -1].float()
        for t in set(toks):
            logits[t] /= REP                     # repetition penalty
        logits = logits / TEMP                   # temperature
        kth = torch.topk(logits, TOP_K).values[-1]   # top-k
        logits[logits < kth] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        sp, si = probs.sort(descending=True)     # top-p 核采样
        cum = torch.cumsum(sp, dim=-1)
        keep = si[:(cum > TOP_P).nonzero()[0].item() + 1]
        mask = torch.zeros_like(probs); mask[keep] = probs[keep]
        nxt = torch.multinomial(mask / mask.sum(), 1).item()
        if nxt == eos:
            break
        toks.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=model.device)], dim=1)
    return tok.decode(toks, skip_special_tokens=True)


def run(model, tok, tag, n, chat):
    prompts = CONT_PROMPTS + (CHAT_PROMPTS if chat else [])
    for p in prompts:
        mode = "对话" if p in CHAT_PROMPTS and chat else "续写"
        print(f"--- [{mode}] {p}\n{gen(model, tok, p, n, chat=(mode == '对话'))}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp_g128_zh/dopt_config.json")
    ap.add_argument("--ckpt", default="exp_g128_zh/train_output/trained.pth")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args()
    cfg, ckpt = ROOT / "02_quant" / a.config, ROOT / "02_quant" / a.ckpt
    assert ckpt.exists(), f"缺 {ckpt}（先跑三段式量化, 见 REPRODUCE.md 阶段②）"
    tok = AutoTokenizer.from_pretrained(MODEL)

    print(f"=========== FP (bf16) ===========", flush=True)
    fp = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEV); fp.eval()
    run(fp, tok, "FP", a.n, a.chat)
    del fp; torch.cuda.empty_cache()

    print(f"\n=========== QUANT (fp32) {a.config} ===========", flush=True)
    qm = load_quant(cfg, ckpt)
    run(qm, tok, "QUANT", a.n, a.chat)


if __name__ == "__main__":
    main()
