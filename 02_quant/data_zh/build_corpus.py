#!/usr/bin/env python3
"""构建中文校准/测试语料（当前交付版语料的生成脚本，可复现）。

来源: HF wikimedia/wikipedia 20231101.zh（流式，无需整库下载）
产物:
  dataset_zh.json  校准集 [{"text": ...}, ...] —— dopt train_files 用
                   ★ 必须切成多行且行数 ≥ num_samples(1024)：stage1 拼接全文用，
                     stage2 按数据集"行"索引样本（整篇一行会在 stage2 越界崩溃）
  test_zh.txt      留出测试集（eval_ppl.py --data 用，不参与校准）

用法: venv 的 python 直接运行（需 datasets）:
  01_prepare/venv/bin/python 02_quant/data_zh/build_corpus.py
"""
import json
from datasets import load_dataset

TARGET_TRAIN_CHARS, TARGET_TEST_CHARS = 2_000_000, 260_000
CHUNK_CHARS = 700          # 每行 ~700 字（≈450-550 token），贴近 cutoff_len=512

ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
train_texts, test_texts = [], []
n_train = n_test = 0
for art in ds:
    t = art["text"].strip()
    if len(t) < 200:            # 过滤超短条目
        continue
    if n_train < TARGET_TRAIN_CHARS:
        train_texts.append(t); n_train += len(t)
    elif n_test < TARGET_TEST_CHARS:
        test_texts.append(t); n_test += len(t)
    else:
        break
print(f"train: {len(train_texts)} 篇/{n_train} 字; test: {len(test_texts)} 篇/{n_test} 字")

# 按段落边界切成 ~CHUNK_CHARS 的行（保序：stage1 的拼接流不变）
rows, buf = [], []
for t in train_texts:
    for para in t.split("\n"):
        para = para.strip()
        if not para:
            continue
        buf.append(para)
        if sum(len(x) for x in buf) >= CHUNK_CHARS:
            rows.append({"text": "\n".join(buf)}); buf = []
if buf:
    rows.append({"text": "\n".join(buf)})

with open("dataset_zh.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)
with open("test_zh.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(test_texts))
print(f"写出 dataset_zh.json: {len(rows)} 行(≥1024 ✓, 平均 {sum(len(r['text']) for r in rows)//len(rows)} 字/行) + test_zh.txt")
