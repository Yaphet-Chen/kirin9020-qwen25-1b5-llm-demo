#!/usr/bin/env bash
# 一键部署: 从云侧拉取 device 文件 -> 推送手机 -> 校验 -> 重启 App 验证
# 用法:
#   bash 05_device_files_base/deploy_from_cloud.sh base       # 部署 base 版(默认)
#   bash 05_device_files_base/deploy_from_cloud.sh instruct   # 部署 instruct 版
# 前置:
#   1. 手机已连 USB 且装过 App(首次用 DevEco 装一次即可)
#   2. 云侧密码: export CANN_SSH_PASS='xxx' (不设则 ssh 交互式询问)
set -uo pipefail

VARIANT="${1:-base}"
CLOUD_HOST="chenyipei@10.232.200.8"
CLOUD_DIR="/home/chenyipei/test_omc/qwen25_1b5_run/05_device_files_${VARIANT}"
HDC="/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe"
REAL_DIR="/data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files"
BUNDLE="com.huawei.cannkit.llmengine"
STAGE="$(cd "$(dirname "$0")" && pwd)/.deploy_tmp"
export MSYS2_ARG_CONV_EXCL="*"

# --- ssh 免交互(有 CANN_SSH_PASS 时) ---
if [ -n "${CANN_SSH_PASS:-}" ]; then
  printf '#!/bin/sh\necho "%s"\n' "$CANN_SSH_PASS" > /tmp/.deploy_askpass.sh
  chmod 700 /tmp/.deploy_askpass.sh
  export SSH_ASKPASS=/tmp/.deploy_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=none
fi
SSH="ssh -o StrictHostKeyChecking=no $CLOUD_HOST"
SCP="scp -o StrictHostKeyChecking=no"

fail() { echo "[FAIL] $*"; rm -rf "$STAGE" /tmp/.deploy_askpass.sh; exit 1; }

echo "== 1/5 检查设备 =="
"$HDC" list targets | grep -q . || fail "没有设备连接, 检查 USB"
"$HDC" shell "ls -d $REAL_DIR" >/dev/null 2>&1 || fail "App 未安装, 先用 DevEco 装一次"

echo "== 2/5 从云侧拉取 $VARIANT 文件 =="
rm -rf "$STAGE"; mkdir -p "$STAGE"
FILES=$($SSH "cd $CLOUD_DIR && ls" ) || fail "云侧目录不可读: $CLOUD_DIR"
echo "$FILES" | grep -q "SubGraph_0.weight" || fail "云侧缺 SubGraph_0.weight"
for f in $FILES; do
  case "$f" in
    *.omc|SubGraph_0.weight|*.embedding_weights|*.embedding_dequant_scale|tokenizer.json|executor.json|context.json)
      $SCP "$CLOUD_HOST:$CLOUD_DIR/$f" "$STAGE/$f" >/dev/null 2>&1 || fail "拉取失败: $f"
      echo "  pulled $f";;
  esac
done
[ -f "$STAGE/context.json" ] || fail "云侧缺 context.json"

echo "== 3/5 md5 校验(云侧 vs 本地) =="
$SSH "cd $CLOUD_DIR && md5sum *" 2>/dev/null | sort > "$STAGE/cloud.md5"
(cd "$STAGE" && md5sum *.omc SubGraph_0.weight *.embedding_weights *.embedding_dequant_scale tokenizer.json executor.json context.json 2>/dev/null) | sort > "$STAGE/local.md5"
join -j1 <(awk '{print $1, $2}' "$STAGE/cloud.md5" | sort -k2) <(awk '{print $1, $2}' "$STAGE/local.md5" | sort -k2) 2>/dev/null | awk '$2!=$3{print $1}' > "$STAGE/diff.txt" || true
# join 对不齐时(云侧多出的非部署文件)不算失败, 只查本地这 7 个文件每个都在云侧 md5 里
while read -r line; do
  h=$(echo "$line" | awk '{print $1}'); f=$(echo "$line" | awk '{print $2}')
  grep -q "^$h " "$STAGE/cloud.md5" || fail "md5 不一致(疑似传输截断): $f"
done < "$STAGE/local.md5"
echo "  md5 全部一致"

echo "== 4/5 推送手机 =="
cd "$STAGE"
for f in *; do
  case "$f" in *.md5|diff.txt) continue;; esac
  dst="$f"
  "$HDC" shell "rm -rf /data/local/tmp/.dep_stage" >/dev/null 2>&1
  # 注意: hdc file send 用裸文件名(相对路径)最稳, 且目标不存在时可能建成目录, 两种形态都兼容
  "$HDC" file send "$f" "/data/local/tmp/.dep_stage" 2>&1 | grep -q "FileTransfer finish" || fail "hdc 发送失败: $f"
  "$HDC" shell "if [ -d /data/local/tmp/.dep_stage ]; then cp /data/local/tmp/.dep_stage/$f $REAL_DIR/$dst; else cp /data/local/tmp/.dep_stage $REAL_DIR/$dst; fi; rm -rf /data/local/tmp/.dep_stage"
  # 逐文件核对端上大小, 防止静默失败
  LOCAL_SZ=$(stat -c%s "$f")
  DEV_SZ=$("$HDC" shell "stat -c%s $REAL_DIR/$dst" 2>/dev/null | tr -d '\r')
  [ "$LOCAL_SZ" = "$DEV_SZ" ] || fail "端上大小不符($LOCAL_SZ vs $DEV_SZ): $f"
  echo "  pushed $f -> $dst ($LOCAL_SZ bytes, 已核验)"
done
cd - >/dev/null
"$HDC" shell "chmod 755 $REAL_DIR/*" >/dev/null 2>&1
# 端上复核大文件
DEV_MD5=$("$HDC" shell "md5sum $REAL_DIR/SubGraph_0.weight" | awk '{print $1}')
grep -q "^$DEV_MD5 " "$STAGE/cloud.md5" || fail "端上 SubGraph_0.weight md5 与云侧不符"
echo "  端上 weight md5 一致"

echo "== 5/5 重启 App 验证加载 =="
"$HDC" shell "aa force-stop $BUNDLE; sleep 1; hilog -r; aa start -a EntryAbility -b $BUNDLE" >/dev/null 2>&1
ok=0
for i in $(seq 1 30); do
  sleep 2
  if "$HDC" shell "hilog -x" 2>/dev/null | grep -q "LLM Engine Init Done."; then ok=1; break; fi
done
rm -rf "$STAGE" /tmp/.deploy_askpass.sh
if [ "$ok" = 1 ]; then
  echo "[DONE] $VARIANT 部署完成, 模型加载成功。打开 App 即可对话。"
else
  echo "[FAIL] 模型未在 60s 内加载完成, 查日志: $HDC shell hilog | grep LLM_DEMO"
  exit 1
fi
