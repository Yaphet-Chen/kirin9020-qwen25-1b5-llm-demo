"""KD 蒸馏兼容补丁：修复 dopt Cython 代码在 transformers 4.51 下的两处 API 断裂。"""
try:
    import transformers
    # 1) TrainingArguments.evaluation_strategy → eval_strategy (4.51 改名)
    from transformers import TrainingArguments
    if not hasattr(TrainingArguments, "_kd_patched"):
        _o = TrainingArguments.__init__
        def _p(self, *a, **k):
            if "evaluation_strategy" in k and "eval_strategy" not in k:
                k["eval_strategy"] = k.pop("evaluation_strategy")
            return _o(self, *a, **k)
        TrainingArguments.__init__ = _p
        TrainingArguments._kd_patched = True

    # 2) transformers.AdamW 在 4.51 被移除（deprecated）；用 torch 的 AdamW 替代
    if not hasattr(transformers, "AdamW"):
        import torch
        transformers.AdamW = torch.optim.AdamW
except Exception:
    pass
