#!/usr/bin/env python3
"""
量化精度评测：对比浮点模型 vs 量化仿真模型的 wikitext2 perplexity (PPL)。
用法:
    python eval_ppl.py <hf_model_path> <dopt_config.json> <quant_ckpt.pth> [--tag NAME]
PPL 越低越好；量化与浮点的 PPL 差距越小，量化精度损失越小。
"""
import sys, os, argparse, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_sim import load_quant   # 量化仿真加载（含 env 自举），本脚本原 get_quanted_model 已并入


@torch.no_grad()
def compute_ppl(model, input_ids, device, chunk=256):
    """按固定 chunk 累计 neg log likelihood 算 PPL。
    chunk=256 控制显存：每次 forward 一个 256 长度片段，独立算 next-token nll。
    （不做 sliding window 拼接——chunk 远小于 max_position，每个片段独立成立）"""
    model.eval()
    total_nll, n_tokens = 0.0, 0
    seq = input_ids[0]
    for i in range(0, seq.size(0) - 1, chunk):
        inp = seq[i:i + chunk].unsqueeze(0).to(device)   # 固定 chunk 长度
        if inp.size(1) < 2:
            continue
        out = model(inp)
        logits = out.logits[:, :-1, :]                     # [1, L-1, V]
        target = inp[:, 1:]                                 # [1, L-1]
        # fp16 log_softmax + gather，省显存（vocab=151936 较大）
        # fp32 log_softmax（fp16 对大 vocab 数值不稳定），nll = -log_prob(target)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
        total_nll += nll.sum().item()
        n_tokens += target.numel()
        del out, logits, target, log_probs, nll
        torch.cuda.empty_cache()
    ppl = math.exp(total_nll / n_tokens) if n_tokens else float("nan")
    return ppl, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("dopt_config")
    ap.add_argument("quant_ckpt", nargs="?", default=None)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_samples", type=int, default=0, help="测试 token 数上限，0=全量")
    ap.add_argument("--data", default="wikitext2", help="wikitext2 或中文测试文本文件路径")
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # 测试集：wikitext2（标准英文）或 --data 指定的文本文件（如中文）
    if args.data == "wikitext2":
        from datasets import load_dataset
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(test["text"])
    else:
        text = open(args.data, encoding="utf-8").read()
    enc = tok(text, return_tensors="pt")
    input_ids = enc.input_ids
    # 截断到模型 max_position 之内（避免 position_ids 越界导致 RoPE nan）
    max_pos = 32768   # Qwen2.5-1.5B(max/instruct 同) max_position_embeddings
    if input_ids.size(1) > max_pos:
        input_ids = input_ids[:, :max_pos]
    if args.n_samples > 0 and input_ids.size(1) > args.n_samples:
        input_ids = input_ids[:, :args.n_samples]
    print(f"[{args.tag}] 测试集({args.data}) token 数: {input_ids.size(1)}")

    # 浮点基线 PPL（fp32 评测，避免 fp16 在量化算子上溢出 nan）
    fp = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, device_map=device)
    fp_ppl, n = compute_ppl(fp, input_ids, device)
    print(f"[FP-{args.tag}] PPL = {fp_ppl:.4f}  (tokens={n})")
    del fp
    torch.cuda.empty_cache()

    # 量化仿真 PPL（fp32：量化算子反量化后 fp32 累加，避免 fp16 溢出）
    if args.quant_ckpt and os.path.exists(args.quant_ckpt):
        qm = load_quant(args.model_path, args.dopt_config, args.quant_ckpt, device=device)
        q_ppl, n2 = compute_ppl(qm, input_ids, device)
        print(f"[QUANT-{args.tag}] PPL = {q_ppl:.4f}  (tokens={n2})")
        print(f"[DELTA-{args.tag}] ΔPPL = {q_ppl - fp_ppl:+.4f}  (相对劣化 {(q_ppl/fp_ppl - 1)*100:+.2f}%)")
    else:
        print(f"[{args.tag}] 未提供 quant_ckpt，仅评测浮点基线")


if __name__ == "__main__":
    main()
