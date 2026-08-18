#!/usr/bin/env bash
# =============================================================================
# 统一产线入口：base / instruct 共用同一套脚本，各自 profile 达到各自最优配方。
#
# 用法:
#   bash pipeline.sh base              # base 全流程（g128 + zh维基校准，= 交付版配方）
#   bash pipeline.sh instruct          # instruct 全流程（g64 + zh对话校准）
#   bash pipeline.sh <p> quant         # 只跑 阶段② 三段式量化（~35min, GPU）
#   bash pipeline.sh <p> eval          # 只跑 PPL 评测（instruct 附带 chat 生成对比）
#   bash pipeline.sh <p> export        # 只跑 阶段③ ONNX 导出（~3min, GPU）
#   bash pipeline.sh <p> convert       # 只跑 阶段④ omc 转换（~2min）
#   bash pipeline.sh <p> pack          # 只跑 阶段⑤ 汇集端侧 7 文件
#   bash pipeline.sh <p> status        # 查看各阶段产物状态
#   FORCE=1 bash pipeline.sh <p> ...   # 跳过"已完成则跳过"保护（重量化必用）
#
# 演示流程（先 base 后 instruct）:
#   bash pipeline.sh base && bash pipeline.sh instruct
#   产物分别在 05_device_files_base/ 与 05_device_files_instruct/，
#   分别拷到连手机的 Windows 机跑各自目录里的 push_to_device_next.bat 即可。
#
# profile 差异（profiles/*.env）:
#   base:     Qwen2.5-1.5B      g128  zh维基校准（续写演示）
#   instruct: Qwen2.5-1.5B-Ins  g64   zh对话校准（ChatML，对话演示）
# 其余（int4/s16激活/s1024/c512/lm_head保fp/PTQ/quant_param_2=False）两条产线一致，
# 均为 base 实验矩阵得出的最优公共项（QUANTIZATION.md §一/§三）。
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

PROFILE="${1:?用法: bash pipeline.sh <base|instruct> [all|quant|eval|export|convert|pack|status]}"
STAGE="${2:-all}"
PROFILE_FILE="$ROOT/profiles/${PROFILE}.env"
[ -f "$PROFILE_FILE" ] || { echo "❌ 未知 profile: $PROFILE（可选 base|instruct）"; exit 1; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"

# 公共环境（评测/仿真要 import dopt → deepspeed，需 CUDA_HOME stub；_autopatch 兼容 transformers 4.51）
export CUDA_HOME="$ROOT/01_prepare/cuda_stub"
export PYTHONPATH="$ROOT/02_quant/_autopatch:$ROOT/01_prepare/tools/tools_dopt/dopt_pytorch_py3:${PYTHONPATH:-}"

MODEL_DIR="$ROOT/01_prepare/models/${PIPE_MODEL}"
PY="$ROOT/01_prepare/venv/bin/python"
STEP() { echo; echo "########## [${PROFILE}] $1 ##########"; }

need_model() {
    [ -f "$MODEL_DIR/config.json" ] || { echo "❌ 缺 $MODEL_DIR（先下载模型，见 01_prepare/prepare.sh）"; exit 1; }
}

# ---------- 阶段② 量化 ----------
do_quant() {
    need_model
    cd "$ROOT/02_quant"
    local QP="./${PIPE_TESTCASE}/train_output/quant_params_file"
    if [ -f "$QP" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "跳过量化（${PIPE_TESTCASE}/train_output/quant_params_file 已存在；重跑用 FORCE=1）"
        return
    fi
    export TESTCASE="$PIPE_TESTCASE" CFG="$PIPE_CFG" MODEL="$MODEL_DIR"   # run.sh 三件套，一次导出
    if [ ! -f "./${PIPE_TESTCASE}/dopt_config.json" ]; then
        STEP "stage1 第一次：生成 dopt_config.json"
        bash run.sh stage1 2>&1 | tail -2
    fi
    STEP "编辑量化策略: int4 per-group ${PIPE_GROUP} + lm_head 保 fp + eco"
    "$PY" edit_dopt_config.py "./${PIPE_TESTCASE}/dopt_config.json" "$PIPE_GROUP" $PIPE_EDIT_FLAGS
    STEP "stage1 权重量化 GPTQ (~19min)"
    bash run.sh stage1 2>&1 | tail -2
    STEP "stage2 激活校准 EMA (~10min)"
    bash run.sh stage2 2>&1 | tail -2
    STEP "stage3 参数提取 (~4min)"
    bash run.sh stage3 2>&1 | tail -2
    ls -lah "./${PIPE_TESTCASE}/train_output/" | grep -E 'trained|quant_params|fake' || true
}

# ---------- 评测 ----------
do_eval() {
    need_model
    cd "$ROOT/02_quant"
    [ -f "./${PIPE_TESTCASE}/train_output/trained.pth" ] || { echo "❌ 先跑 quant（缺 trained.pth）"; exit 1; }
    # GPU 被占时可 EVAL_DEVICE=cpu bash pipeline.sh <p> eval（fp32 CPU 慢但结果等价）
    local DEV="${EVAL_DEVICE:-cuda}"
    STEP "PPL 评测（留出集: ${PIPE_EVAL_DATA}, fp32 仿真, device=${DEV}）"
    "$PY" eval_ppl.py "$MODEL_DIR" "./${PIPE_TESTCASE}/dopt_config.json" \
        "./${PIPE_TESTCASE}/train_output/trained.pth" \
        --tag "${PROFILE}" --device "$DEV" --n_samples 8192 --data "${PIPE_EVAL_DATA}" 2>&1 | tee "$ROOT/logs/eval_${PROFILE}.log" | grep -E "FP-|QUANT-|DELTA-|tokens"
    if [[ "$PIPE_MODEL" == *Instruct* ]]; then
        STEP "chat 生成质量对比（FP vs 量化仿真，chat template）"
        CHAT_DEVICE="$DEV" "$PY" chat_test.py "$MODEL_DIR" "./${PIPE_TESTCASE}/dopt_config.json" \
            "./${PIPE_TESTCASE}/train_output/trained.pth" \
            "你好，请做一下自我介绍。" "用三句话介绍一下长城的历史。" \
            "What is 25 multiplied by 37? Think step by step." \
            "写一个Python函数判断一个数是否是质数。" \
            < /dev/null 2>&1 | tee "$ROOT/logs/chat_${PROFILE}.log" | tail -5
    else
        echo "（base 为续写模型，不走 chat template；生成质量对比见 02_quant/device_compare.py）"
    fi
}

# ---------- 阶段③ ONNX 导出 ----------
do_export() {
    need_model
    cd "$ROOT"
    local FQ="./02_quant/${PIPE_TESTCASE}/train_output/fake_quant_weight.pth"
    if [ ! -f "$FQ" ]; then
        STEP "缺 fake_quant_weight.pth，补跑 stage3（~4min）"
        ( cd 02_quant && TESTCASE="$PIPE_TESTCASE" CFG="$PIPE_CFG" MODEL="$MODEL_DIR" bash run.sh stage3 2>&1 | tail -2 )
    fi
    STEP "阶段③ ONNX 导出（${PIPE_EXPORT_YAML}）"
    EXPORT_YAML="$PIPE_EXPORT_YAML" bash 03_onnx_export/export.sh
}

# ---------- 阶段④ omc 转换 ----------
do_convert() {
    STEP "阶段④ omg 转换（prefix: ${PIPE_OMC_PREFIX}）"
    QUANT_DIR="$ROOT/02_quant/${PIPE_TESTCASE}" \
    ONNX_DIR="$ROOT/03_onnx_export/${PIPE_ONNX_SUBDIR}" \
    OUTPUT_PREFIX="./${PIPE_OMC_PREFIX}" \
    bash 04_omc_convert/convert.sh
}

# ---------- 阶段⑤ 端侧 7 文件 ----------
do_pack() {
    STEP "阶段⑤ 汇集端侧 7 文件 → ${PIPE_DEVICE_DIR}/"
    OMC_NAME="$PIPE_OMC_PREFIX" \
    OMC_DIR="$ROOT/04_omc_convert/${PIPE_OMC_PREFIX}" \
    EMB_DIR="$ROOT/03_onnx_export/${PIPE_ONNX_SUBDIR}" \
    EMB_SRC_STEM="$PIPE_EMB_SRC_STEM" \
    EMB_STEM="$PIPE_EMB_STEM" \
    MODEL_DIR="$MODEL_DIR" \
    DEST_DIR="$ROOT/${PIPE_DEVICE_DIR}" \
    bash pack.sh
}

# ---------- 状态 ----------
do_status() {
    echo "===== [${PROFILE}] 产线状态 ====="
    local q="02_quant/${PIPE_TESTCASE}/train_output"
    for f in "$q/trained_quant_weight.pth:②.3 权重量化" \
             "$q/trained.pth:②.4 激活校准" \
             "$q/quant_params_file:②.5 参数提取" \
             "$q/fake_quant_weight.pth:③前置(fake_quant)" \
             "03_onnx_export/${PIPE_ONNX_SUBDIR}/model.onnx:③ ONNX" \
             "04_omc_convert/${PIPE_OMC_PREFIX}/${PIPE_OMC_PREFIX}.omc:④ omc" \
             "04_omc_convert/${PIPE_OMC_PREFIX}/SubGraph_0.weight:④ weight"; do
        local p="${f%%:*}" label="${f#*:}"
        if [ -e "$ROOT/$p" ]; then
            printf "  ✅ %-28s %s\n" "$label" "$(du -h "$ROOT/$p" | cut -f1)  $p"
        else
            printf "  ⬜ %-28s %s\n" "$label" "$p"
        fi
    done
    echo "  ⑤ 目标目录: ${PIPE_DEVICE_DIR}/"
}

cd "$ROOT"
case "$STAGE" in
    all)    do_quant; do_eval; do_export; do_convert; do_pack ;;
    quant)  do_quant ;;
    eval)   do_eval ;;
    export) do_export ;;
    convert) do_convert ;;
    pack)   do_pack ;;
    status) do_status ;;
    *) echo "❌ 未知阶段: $STAGE（可选 all|quant|eval|export|convert|pack|status）"; exit 1 ;;
esac
STEP "完成: ${PROFILE} / ${STAGE}"
