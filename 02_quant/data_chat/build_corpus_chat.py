#!/usr/bin/env python3
"""构建 instruct 版中文对话校准/测试语料（Qwen2.5-1.5B-Instruct 专用）。

与 data_zh_wiki（zh 维基续写语料，base 版用）的区别：
  - 校准分布对齐部署分布：demos 是对话，故语料必须是 chat template 渲染后的
    ChatML 文本（<|im_start|>role\\n...<|im_end|>），与端侧 app 侧
    apply_chat_template 后喂引擎的 token 流一致。

来源（HF 流式，无需整库下载）:
  BelleGroup/multiturn_chat_0.8M  多轮对话（instruction 内嵌 Human:/Assistant: 标记，需解析）
  BelleGroup/train_0.5M_CN        单轮指令（instruction/input → output）
  混合比例（按对话数）: 多轮:单轮 = 2:1

产物:
  dataset.json  校准集 [{"text": ...}, ...] —— dopt train_files 用
                        ★ 与 data_zh_wiki 同约束：行数 ≥ num_samples(1024)，
                          stage1 拼接全文、stage2 按行索引；每行打包完整对话、
                          宽度 ~480 token 贴近 cutoff_len=512
  test.txt      留出对话测试集（eval_ppl.py --data 用，不参与校准）

用法（需 venv，模型已下载）:
  01_prepare/venv/bin/python 02_quant/data_chat/build_corpus_chat.py
"""
import json
import re
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = "/home/chenyipei/test_omc/qwen25_1b5_run"
TOK_PATH = f"{ROOT}/01_prepare/models/Qwen2.5-1.5B-Instruct"
OUT_DIR = f"{ROOT}/02_quant/data_chat"

# 打包参数：行宽（token）目标，贴近 cutoff_len=512 且留 template 余量
ROW_TOKEN_BUDGET = 480
MIN_CONV_TOKENS, MAX_CONV_TOKENS = 60, 900
TARGET_ROWS = 2400            # ≥1024 且留余量；总量 ≈ 1.1M token（GPTQ 需 524k）
TEST_CONVS = 120              # 留出对话数（~8k+ token，匹配 eval n_samples=8192）
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SINGLE_RATIO = 3              # 每 3 条多轮插入 1 条单轮


def parse_multiturn(instruction, output):
    """把 Belle multiturn 的 'Human:..\\nAssistant:..\\nHuman:..\\nAssistant:' 解析成 turns。
    尾部悬空 'Assistant:' 是生成提示，配 output 作最后一轮 assistant。"""
    parts = re.split(r"(?:^|\n)(Human:|Assistant:)", instruction)
    turns, role = [], None
    for p in parts:
        if p in ("Human:", "Assistant:"):
            role = "user" if p == "Human:" else "assistant"
        elif role and p.strip():
            turns.append({"role": role, "content": p.strip()})
    if not turns or turns[-1]["role"] != "user":
        return None
    turns.append({"role": "assistant", "content": output.strip()})
    if len([t for t in turns if t["role"] == "user"]) < 2:
        return None                      # 只保留真多轮（≥2 问）
    return turns


def iter_conversations():
    """流式产出 (turns,)；多轮:单轮 = 2:1 交错。"""
    mt = load_dataset("BelleGroup/multiturn_chat_0.8M", split="train", streaming=True)
    st = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)
    st_it = iter(st)
    n_mt = n_st = 0
    for art in mt:
        turns = parse_multiturn(art["instruction"], art.get("output", ""))
        if turns:
            yield turns, "mt"
            n_mt += 1
        if n_mt % SINGLE_RATIO == 0:
            try:
                art = next(st_it)
            except StopIteration:
                continue
            user = art["instruction"].strip()
            if art.get("input", "").strip():
                user += "\n" + art["input"].strip()
            out = art.get("output", "").strip()
            if user and len(out) >= 20:
                yield [{"role": "user", "content": user},
                       {"role": "assistant", "content": out}], "st"
                n_st += 1
        if n_mt >= TARGET_ROWS * 2:      # 上限保护（足够填满打包预算）
            break


def main():
    tok = AutoTokenizer.from_pretrained(TOK_PATH)
    tmpl = tok.chat_template  # 确认 instruct 版带 ChatML 模板

    def render(turns):
        msgs = [{"role": "system", "content": SYSTEM}] + turns
        return tok.apply_chat_template(msgs, tokenize=False)

    rows, test_texts = [], []
    seen_first_user = set()
    buf, buf_tok = [], 0
    n_mt = n_st = n_skipped = 0
    total_tok = 0

    for turns, src in iter_conversations():
        key = turns[0]["content"][:64]           # 按首问去重
        if key in seen_first_user:
            continue
        seen_first_user.add(key)

        text = render(turns)
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        if not (MIN_CONV_TOKENS <= n_tok <= MAX_CONV_TOKENS):
            n_skipped += 1
            continue

        if len(test_texts) < TEST_CONVS and len(rows) > 50:  # 前 50 行只进校准，再开始留出
            test_texts.append(text)
            continue

        # 打包：装满一行（一行 ≥1 条完整对话，不跨行拆对话）
        if buf and buf_tok + n_tok > ROW_TOKEN_BUDGET:
            rows.append({"text": "".join(buf)})
            buf, buf_tok = [], 0
        buf.append(text)
        buf_tok += n_tok
        total_tok += n_tok
        if src == "mt":
            n_mt += 1
        else:
            n_st += 1
        if len(rows) >= TARGET_ROWS:
            break
    if buf:
        rows.append({"text": "".join(buf)})

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/dataset.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    with open(f"{OUT_DIR}/test.txt", "w", encoding="utf-8") as f:
        f.write("".join(test_texts))

    toks_per_row = [len(tok(r["text"], add_special_tokens=False)["input_ids"]) for r in rows[:200]]
    print(f"校准集: {len(rows)} 行 (≥1024: {len(rows) >= 1024}), "
          f"多轮 {n_mt} / 单轮 {n_st}, 跳过(长度/去重后不合规) {n_skipped}")
    print(f"行宽抽样: min={min(toks_per_row)} avg={sum(toks_per_row)//len(toks_per_row)} "
          f"max={max(toks_per_row)} (cutoff_len=512)")
    print(f"校准总 token ≈ {total_tok // 1000}k (GPTQ 需 ≥524k)")
    print(f"留出集: {len(test_texts)} 条对话 → test.txt "
          f"({sum(len(t) for t in test_texts) // 1000}k 字符)")


if __name__ == "__main__":
    main()
