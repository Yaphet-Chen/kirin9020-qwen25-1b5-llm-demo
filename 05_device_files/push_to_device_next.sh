#!/usr/bin/env bash
# 路线B(HarmonyOS NEXT): 推送 7 个模型/配置文件到应用沙箱真实路径
# 前置:
#   1. 手机为 HarmonyOS NEXT(Kirin 9020/x90), USB 连接并开启开发者模式 + USB 调试
#   2. CANNLLMEngineDemoNext 已安装到手机(目录随安装创建, 未装会报错提示)
# 用法(Git Bash): bash 05_device_files/push_to_device_next.sh
#      (Windows 下推荐直接双击 push_to_device_next.bat)
# 注意: hdc v3.x 失败时不输出且退出码为 0, 因此一律以"输出是否为空"判定成败;
#       DevEco 的 hdc 服务可能跑在非默认端口(如 7035), 默认端口看不到设备时
#       自动复用已在运行的 hdc 服务端口, 否则自己起的服务抢不到 USB 设备。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/05_device_files"

# NEXT 应用沙箱: 真实路径(hdc 用) <-> 沙箱路径(App 用 /data/storage/el2/base/haps/entry/files)
REAL_DIR="/data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files"
BUNDLE="com.huawei.cannkit.llmengine"

# 自动定位 hdc: 优先 PATH, 其次 DevEco Studio 默认安装位置
HDC="$(command -v hdc || true)"
if [ -z "$HDC" ]; then
    for CAND in \
        "/c/Program Files/Huawei/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe" \
        "C:\\Program Files\\Huawei\\DevEco Studio\\sdk\\default\\openharmony\\toolchains\\hdc.exe"; do
        if [ -x "$CAND" ] || [ -f "$CAND" ]; then HDC="$CAND"; break; fi
    done
fi
if [ -z "$HDC" ]; then
    echo "❌ 未找到 hdc, 请把 DevEco 的 toolchains 目录加入 PATH 或修改本脚本 HDC 变量"
    exit 1
fi
echo "使用 hdc: $HDC"

echo "== 已连接设备 =="
TARGETS="$("$HDC" list targets 2>&1 | grep -v '^\[Empty\]' || true)"
if [ -z "$TARGETS" ]; then
    # 默认端口无设备: 复用已在运行的 hdc 服务端口(DevEco 常用 7035)
    for pid in $(tasklist //FI "IMAGENAME eq hdc.exe" //FO CSV //NH 2>/dev/null \
                 | sed 's/"//g' | cut -d, -f2 | grep -E '^[0-9]+$'); do
        port="$(netstat -ano 2>/dev/null | grep LISTENING | grep '127.0.0.1:' \
                | grep -E "[[:space:]]${pid}\$" | awk '{print $2}' | cut -d: -f2 | head -1)"
        if [ -n "$port" ]; then
            out="$(HDC_SERVER_PORT="$port" "$HDC" list targets 2>&1 | grep -v '^\[Empty\]' || true)"
            if [ -n "$out" ]; then
                echo "复用 hdc 服务端口: ${port}"
                export HDC_SERVER_PORT="$port"
                TARGETS="$out"
                break
            fi
        fi
    done
fi
if [ -z "$TARGETS" ]; then
    echo "❌ 没有已连接设备。请用 USB 连接手机并允许 USB 调试;"
    echo "   若仍不行, 拔插一次 USB 线(或开关一次 USB 调试)后重跑。"
    exit 1
fi
echo "$TARGETS"

echo "== 检查 App 是否已安装(沙箱目录是否已创建) =="
if [ -z "$("$HDC" shell "ls -d ${REAL_DIR}" 2>&1 || true)" ]; then
    echo "❌ 设备上不存在 ${REAL_DIR}"
    echo "   请先用 DevEco Studio 把 CANNLLMEngineDemoNext 安装到手机, 再运行本脚本。"
    echo "   (可用 bm dump 确认: hdc shell bm dump -n ${BUNDLE})"
    exit 1
fi

# 7 文件: 本地文件 -> 设备文件名(context_next.json 以 context.json 名义写入设备)
push_one() {
    local local_file="$1" remote_name="$2" out=""
    echo ">>> ${remote_name}  ($(du -h "${local_file}" | cut -f1))"
    out="$("$HDC" file send "${local_file}" "${REAL_DIR}/${remote_name}" 2>&1 || true)"
    if [ -z "$out" ]; then
        echo "❌ 推送失败(无输出): ${remote_name}"
        exit 1
    fi
    echo "    ${out}"
}

push_one "${SRC}/Qwen25_1b5_kirin9020.omc"                  "Qwen25_1b5_kirin9020.omc"
push_one "${SRC}/SubGraph_0.weight"                         "SubGraph_0.weight"
push_one "${SRC}/model_64_2048.embedding_weights"           "model_64_2048.embedding_weights"
push_one "${SRC}/model_64_2048.embedding_dequant_scale"     "model_64_2048.embedding_dequant_scale"
push_one "${SRC}/tokenizer.json"                            "tokenizer.json"
push_one "${SRC}/context_next.json"                         "context.json"
push_one "${SRC}/executor.json"                             "executor.json"

# hdc 传入文件权限可能不带其他用户读权限, 统一放开
"$HDC" shell "chmod -R 755 ${REAL_DIR}" >/dev/null 2>&1 || true

echo "== 设备侧最终核验 =="
LISTING="$("$HDC" shell "ls -lh ${REAL_DIR}" 2>&1 || true)"
echo "$LISTING"
MISSING=""
for n in Qwen25_1b5_kirin9020.omc SubGraph_0.weight \
         model_64_2048.embedding_weights model_64_2048.embedding_dequant_scale \
         tokenizer.json context.json executor.json; do
    echo "$LISTING" | grep -q "$n" || MISSING="${MISSING} ${n}"
done
if [ -n "$MISSING" ]; then
    echo "❌ 设备上缺少:${MISSING}"
    exit 1
fi

echo ""
echo "✅ 7 个文件全部核验通过。App 侧读取的是沙箱路径 /data/storage/el2/base/haps/entry/files/"
echo "   (llm_demo.cpp InitParam 与 context_next.json 均已指向该路径)"
