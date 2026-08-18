#!/usr/bin/env python3
# 编辑 dopt_config.json 量化策略（在 stage1 第一次执行后、第二次执行前运行）
# 默认策略: embed→Quant_Embed_MinMax; 所有 Linear→Quant_act_weight_eco(weight4/group,input16); lm_head→Quant_lm_head
# 注意: 不加 output 段（实跑验证：加了 output 反而让 omg 无法识别 s16s4 pattern）
# 可用策略全集(来自 strategy.so): float / Quant_act_weight_eco / Quant_Embed_MinMax /
#   Quant_lm_head / Quant_aigc_ptq / Quant_aigc_qat / Quant_Dummy
import json
import os
import sys
from collections import Counter

p = sys.argv[1] if len(sys.argv) > 1 else "qwen25_1b5_base_9020/dopt_config.json"
group_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64
keep_lm_head_fp = "--keep-lm-head-fp" in sys.argv or os.environ.get("KEEP_LM_HEAD_FP", "1") == "1"
# 可选: --linear-strategy X 把 decoder linear 换成其他策略（默认 Quant_act_weight_eco）
lin_strategy = "Quant_act_weight_eco"
if "--linear-strategy" in sys.argv:
    lin_strategy = sys.argv[sys.argv.index("--linear-strategy") + 1]

d = json.load(open(p))
ls = d["layer_strategy"]

for k, v in ls.items():
    if k == "model.embed_tokens":
        v["quant_strategy"] = "Quant_Embed_MinMax"
    elif k == "lm_head":
        v["quant_strategy"] = "float" if keep_lm_head_fp else "Quant_lm_head"
    elif "Linear" in v.get("type", ""):
        v["quant_strategy"] = lin_strategy
        v["weight"] = {"bit": 4, "group_size": group_size}
        v["input"] = {"bit": 16}

json.dump(d, open(p, "w"), indent=4, ensure_ascii=False)

d2 = json.load(open(p))
print("quant_strategy 分布:", dict(Counter(v["quant_strategy"] for v in d2["layer_strategy"].values())))
print(f"group_size={group_size}, keep_lm_head_fp={keep_lm_head_fp}, linear={lin_strategy}")
print("示例 layer0 q_proj:", json.dumps(d2["layer_strategy"]["model.layers.0.self_attn.q_proj"], ensure_ascii=False))
