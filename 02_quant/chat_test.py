#!/usr/bin/env python3
"""
GPU 对话测试：对比浮点模型 vs 量化仿真模型的实际生成质量。
用法:
  python chat_test.py <model_path> <dopt_config> <quant_ckpt>
输入问题后回车，FP 和 QUANT 各生成一次，对比输出。输入空行退出。
"""
import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_quanted_model(base_model, dopt_config, quanted_ckpt):
    from dopt.dopt_lm.do_opt import optimize_model, set_quant_state
    from dopt.dopt_lm.train import set_calibrate_state
    model = optimize_model(base_model, dopt_config)
    model.load_state_dict(torch.load(quanted_ckpt, map_location="cpu"), strict=True)
    set_quant_state(model, weight_state=True, input_state=True)
    set_calibrate_state(model, False)
    model.eval()
    return model

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=128):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = [oids[len(iids):] for iids, oids in zip(inputs.input_ids, out_ids)]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

def main():
    mp, cfg, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(mp)

    print("加载浮点模型...", flush=True)
    # 关键：attn_implementation="eager" 避免 sliding window + sdpa 在 generate 时的 KV cache bug
    # （transformers 4.51 + Qwen2.5 sliding_window 会导致生成中途重复乱码）
    fp = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32, device_map=device,
                                              attn_implementation="eager")
    print("加载量化仿真模型...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.float32, device_map=device,
                                                attn_implementation="eager")
    qm = get_quanted_model(base, cfg, ckpt)
    print("模型就绪。输入问题（空行退出）：\n", flush=True)

    prompts = [
        "你好，请做一下自我介绍。",
        "What is 25 multiplied by 37? Think step by step.",
        "写一首关于秋天的五言绝句。",
        "Explain what a large language model is in simple terms.",
        "用Python写一个快速排序函数。",
    ]
    # 交互模式：有命令行问题就用，否则用预设
    user_prompts = sys.argv[4:] if len(sys.argv) > 4 else prompts

    for i, p in enumerate(user_prompts):
        print(f"{'='*70}")
        print(f"[Q{i+1}] {p}")
        print(f"{'='*70}")
        fp_out = generate(fp, tok, p)
        print(f"\n[FP]    {fp_out}\n")
        del_inputs = None
        torch.cuda.empty_cache()
        q_out = generate(qm, tok, p)
        print(f"[QUANT] {q_out}\n")
        torch.cuda.empty_cache()

    # 交互循环
    print("\n--- 交互模式（输入问题回车，空行退出）---")
    while True:
        try:
            p = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not p:
            break
        print(f"\n[FP]    {generate(fp, tok, p)}\n")
        torch.cuda.empty_cache()
        print(f"[QUANT] {generate(qm, tok, p)}\n")
        torch.cuda.empty_cache()

    print("退出。")

if __name__ == "__main__":
    main()
