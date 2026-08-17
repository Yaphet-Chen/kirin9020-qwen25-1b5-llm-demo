#!/bin/bash
# 真机批量收集回复:向 demo App 注入 prompt(test_prompt.txt 钩子)→ 驱动 UI 发送 →
# 等生成完成 → 拉取 last_reply.txt 与 hilog 性能统计到 device_replies/。
# 与云侧对比时配合 02_quant/device_compare.py（同名采样参数），报告见
# 02_quant/device_compare_report_3way_*.md。
export HDC="/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe"
export MSYS2_ARG_CONV_EXCL="*"
REAL_DIR=/data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files
OUT=/tmp/llm_cmp_device.txt
mkdir -p device_replies
: > "$OUT"

dump_layout() {
  local f
  f=$("$HDC" shell "uitest dumpLayout" 2>/dev/null | grep -oE "layout_[0-9]+\.json" | head -1)
  [ -n "$f" ] && "$HDC" shell "cat /data/local/tmp/$f" 2>/dev/null
}

ensure_foreground() {
  local i; for i in $(seq 1 5); do
    "$HDC" shell "power-shell wakeup" >/dev/null 2>&1; sleep 1
    "$HDC" shell "uitest uiInput swipe 658 2500 658 800 600" >/dev/null 2>&1; sleep 1
    "$HDC" shell "aa start -a EntryAbility -b com.huawei.cannkit.llmengine" >/dev/null 2>&1; sleep 2
    if dump_layout | grep -q "请输入内容"; then return 0; fi
  done
  return 1
}

center_of() {
  dump_layout | grep -oE '"attributes":\{[^{}]*"text":"'"$1"'"[^{}]*\}' | grep -oE '"bounds":"[^"]*"' \
    | grep -oE '[0-9]+' | paste -sd' ' | awk '{printf "%d %d", ($1+$3)/2, ($2+$4)/2}'
}

PROMPTS=("长城是中国古代的伟大工程，" "秋天到了，" "人工智能的发展，" "The theory of relativity states that" "Python is a programming language that")
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1))
  echo "=== PROMPT $i: $p" >> "$OUT"
  # 1. 推 prompt 文件(hdc 对不存在的目标路径行为不定:有时建目录有时直接写文件,两种都兼容)
  printf '%s' "$p" > ./tp_send.txt
  "$HDC" shell "rm -rf /data/local/tmp/tp_stage" >/dev/null 2>&1
  "$HDC" file send tp_send.txt "/data/local/tmp/tp_stage" 2>&1 | grep -q "FileTransfer finish" || echo "PROMPT_PUSH_FAIL" >> "$OUT"
  "$HDC" shell "if [ -d /data/local/tmp/tp_stage ]; then cp /data/local/tmp/tp_stage/tp_send.txt $REAL_DIR/test_prompt.txt; else cp /data/local/tmp/tp_stage $REAL_DIR/test_prompt.txt; fi; rm -rf /data/local/tmp/tp_stage; chmod 755 $REAL_DIR/test_prompt.txt"
  # 校验 prompt 文件内容一致
  "$HDC" shell "cat $REAL_DIR/test_prompt.txt" | grep -qF "$p" || echo "PROMPT_VERIFY_FAIL" >> "$OUT"
  # 2. 重启 App(清上下文)
  "$HDC" shell "aa force-stop com.huawei.cannkit.llmengine" >/dev/null 2>&1; sleep 1
  "$HDC" shell "hilog -r" >/dev/null 2>&1
  "$HDC" shell "aa start -a EntryAbility -b com.huawei.cannkit.llmengine" >/dev/null 2>&1
  for j in $(seq 1 30); do sleep 2; "$HDC" shell "hilog -x" 2>/dev/null | grep -q "LLM Engine Init Done." && break; done
  ensure_foreground || { echo "FOREGROUND_FAILED" >> "$OUT"; continue; }
  # 3. 输入占位符并发送(真实 prompt 走文件钩子)
  "$HDC" shell "uitest uiInput click 481 2655" >/dev/null 2>&1; sleep 1
  "$HDC" shell "uitest uiInput inputText 481 2655 'x'" >/dev/null 2>&1; sleep 1
  XY=$(center_of "发送")
  "$HDC" shell "uitest uiInput click $XY" >/dev/null 2>&1
  echo "sent: $p (send at $XY)"
  # 4. 等生成完成
  for j in $(seq 1 60); do sleep 2; "$HDC" shell "hilog -x" 2>/dev/null | grep -q "LLM Engine, Run Done." && break; done
  # 5. 拉回复 + 统计
  "$HDC" shell "cp $REAL_DIR/last_reply.txt /data/local/tmp/reply_$i.txt && rm -f $REAL_DIR/test_prompt.txt" >/dev/null 2>&1
  "$HDC" file recv "/data/local/tmp/reply_$i.txt" "device_replies/reply_$i.txt" >/dev/null 2>&1
  "$HDC" shell "rm -f /data/local/tmp/reply_$i.txt" >/dev/null 2>&1
  "$HDC" shell "hilog -x" 2>/dev/null | grep "LLM_DEMO" | grep -E "inputTokenCount|outputTokenCount|preFillTimeMs|per token" >> "$OUT"
  wc -c "device_replies/reply_$i.txt" 2>/dev/null >> "$OUT"
done
echo DONE_ALL >> "$OUT"
echo ALL_SENT
