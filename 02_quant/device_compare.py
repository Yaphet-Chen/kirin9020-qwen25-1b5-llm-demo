#!/usr/bin/env python3
"""
端侧部署配置下的 FP vs 量化模型续写/对话对比（严格复刻 context.json）。

采样协议 = 05_device_files_base/context.json 逐项:
  seed=99(每个 prompt 前重置, FP/QUANT 消耗完全相同的随机数)
  repetition_penalty=1.2(logits 除法) -> temperature=0.6 -> top-k=16 -> top-p=0.95(核采样)
FP 用 bf16(本机健康路径), 量化仿真用 fp32(dopt 量化算子在 fp16 下溢出 nan)。

用法(自包含, 无需手工 export 环境变量):
  python device_compare.py                          # 默认: g128+中文校准版, 内置续写 prompt
  python device_compare.py --n 60                   # 短输出快速验证
  python device_compare.py --chat                   # 追加 chat-template 对话测试(base 模型仅看退化程度)
  python device_compare.py --greedy --n 600         # 贪心(对齐端侧 do_sample=false), 验证端云数值一致性
  python device_compare.py --emb --greedy --n 600   # 量化仿真改用端侧同款 int8 embedding(跳过 FP 侧)
  python device_compare.py --emb --probe            # 探针: embedding 置零, 验证其真在前向路径上
  python device_compare.py \
      --config qwen25_1b5_instruct_9020/dopt_config.json \
      --ckpt   qwen25_1b5_instruct_9020/train_output/trained.pth   # 换 instruct(g64+对话校准)对比
  DC_MODEL=.../Qwen2.5-1.5B-Instruct DC_TOP_K=20 DC_TOP_P=0.8 DC_TEMP=0.7 DC_REP=1.1 python device_compare.py \
      --config qwen25_1b5_instruct_9020/...               # instruct 模型+其官方采样参数
可复现性: 同机同 torch 版本下, 量化侧(fp32)输出与 logs/g128zh_device_compare.log 逐字一致。
"""
import os, sys, argparse, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

from quant_sim import load_quant   # 量化仿真加载（env 自举 + emb 两时机替换都在里面）

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
MODEL = pathlib.Path(os.environ.get("DC_MODEL", str(ROOT / "01_prepare" / "models" / "Qwen2.5-1.5B")))  # DC_MODEL 可换 instruct 模型
VOCAB, HIDDEN = 151936, 1536
# 端侧 context.json 参数（改这里保持与部署一致）
import os as _os
SEED = int(_os.environ.get("DC_SEED", "99")); TOP_K = int(_os.environ.get("DC_TOP_K", "16")); TOP_P = float(_os.environ.get("DC_TOP_P", "0.95")); TEMP = float(_os.environ.get("DC_TEMP", "0.6")); REP = float(_os.environ.get("DC_REP", "1.2"))  # 环境变量可覆盖, 对齐 instruct 采样参数用

CONT_PROMPTS = [
    "长城是中国古代的伟大工程，", "秋天到了，", "人工智能的发展，",
    "The theory of relativity states that", "Python is a programming language that",
]
CHAT_PROMPTS = ["你好，请介绍一下你自己。", "请用中文解释什么是模型量化。"]


def load_device_embedding(emb_dir, stem):
    """读端侧 int8 embedding 两文件并反量化为 fp32 表（[151936,1536] 行缩放）。"""
    import numpy as np
    d = pathlib.Path(emb_dir)
    w = np.fromfile(d / f"{stem}.embedding_weights", dtype=np.int8)
    s = np.fromfile(d / f"{stem}.embedding_dequant_scale", dtype=np.float32)
    assert w.size == VOCAB * HIDDEN, f"embedding_weights 元素数 {w.size} != {VOCAB*HIDDEN}"
    assert s.size == VOCAB, f"dequant_scale 元素数 {s.size} != {VOCAB}"
    return torch.from_numpy(w.reshape(VOCAB, HIDDEN).astype(np.float32) * s[:, None])


@torch.no_grad()
def gen(model, tok, prompt, n, chat=False, greedy=False):
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
        if greedy:
            nxt = logits.argmax().item()         # do_sample=false: 贪心,验证端云数值一致性用
        else:
            logits = logits / TEMP               # temperature
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


def run(model, tok, tag, n, chat, greedy=False):
    prompts = CONT_PROMPTS + (CHAT_PROMPTS if chat else [])
    for p in prompts:
        mode = "对话" if p in CHAT_PROMPTS and chat else "续写"
        print(f"--- [{mode}] {p}\n{gen(model, tok, p, n, chat=(mode == '对话'), greedy=greedy)}\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen25_1b5_base_9020/dopt_config.json")
    ap.add_argument("--ckpt", default="qwen25_1b5_base_9020/train_output/trained.pth")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="贪心解码(对应端侧 do_sample=false),仅用于端云数值一致性验证")
    ap.add_argument("--emb", action="store_true",
                    help="量化仿真加载端侧同款 int8 embedding(对齐端侧真实输入; 此模式跳过 FP 侧)")
    ap.add_argument("--emb-dir", default=str(ROOT / "05_device_files_base"), help="embedding 两文件所在目录")
    ap.add_argument("--emb-stem", default="model_base_64_2048", help="embedding 文件名主干")
    ap.add_argument("--probe", action="store_true", help="探针: embedding 置零, 验证 embed_tokens 真在前向路径上(配 --emb)")
    a = ap.parse_args()
    cfg, ckpt = ROOT / "02_quant" / a.config, ROOT / "02_quant" / a.ckpt
    assert ckpt.exists(), f"缺 {ckpt}（先跑三段式量化, 见 REPRODUCE.md 阶段②）"
    tok = AutoTokenizer.from_pretrained(MODEL)

    if a.emb:
        emb = load_device_embedding(a.emb_dir, a.emb_stem)
        print(f"device int8 embedding loaded: {tuple(emb.shape)} fp32(反量化)", flush=True)
        qm = load_quant(MODEL, cfg, ckpt, emb=emb)
        if a.probe:
            with torch.no_grad():
                qm.model.embed_tokens.weight.zero_()
            print("PROBE: embed_tokens 已置零, 若输出仍与正常相同则说明该模块不在前向路径上", flush=True)
        with torch.no_grad():
            diff = (qm.model.embed_tokens.weight.float().cpu() - emb).abs().max().item()
        print(f"embed_tokens 与端侧 embedding 最大偏差: {diff} (应为 0 或接近 0)", flush=True)
        print(f"\n=========== QUANT+端侧int8embedding (fp32) {a.config} ===========", flush=True)
        run(qm, tok, "QUANT_EMB", a.n, a.chat, greedy=a.greedy)
        return

    print(f"=========== FP (bf16) ===========", flush=True)
    fp = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEV); fp.eval()
    run(fp, tok, "FP", a.n, a.chat, greedy=a.greedy)
    del fp; torch.cuda.empty_cache()

    print(f"\n=========== QUANT (fp32) {a.config} ===========", flush=True)
    qm = load_quant(MODEL, cfg, ckpt)
    run(qm, tok, "QUANT", a.n, a.chat, greedy=a.greedy)


if __name__ == "__main__":
    main()
