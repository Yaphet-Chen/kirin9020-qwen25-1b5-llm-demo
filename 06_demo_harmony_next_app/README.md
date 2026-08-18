# 端侧 App（CANNLLMEngineDemoNext）定制源码 llm_demo.cpp

本目录存放端侧 HarmonyOS App 工程 `CANNLLMEngineDemoNext`（来自上游样例库
`cannkit_samplecode_lm_engine_cpp`）中 `llm_demo.cpp` 的**本仓库定制版**，
让克隆者仅凭本仓库即可复现端侧 App，无需访问任何私有改动。

## 相对上游改了什么

基于上游 master `762941b`（!1 合并后）的定制，真机（Kirin 9020）实测通过：

1. **UTF-8 半字符乱码修复**：流式回调的 token 是字节级 BPE，一个中文字符（3 字节）
   可能跨两次回调，残缺半字符直接上报会显示乱码——先攒在缓冲区，拼完整再上报
   （见文件内 `CompleteUtf8PrefixLen` / `g_pendingUtf8`）。
2. **instruct chat template 自动包装**：instruct 模型下，prompt 未含 `<|im_start|>`
   时自动按 ChatML 格式包装（Qwen 官方 system prompt + user/assistant 段），
   配合 `05_device_files_instruct/` 交付版直接对话。
3. **自动化测试钩子**：配合 `同目录 collect_phone_replies.sh` 与
   `02_quant/continuation_compare.py` 做端云三方对比（详见
   `02_quant/reports/report_base_3way_{greedy,sampling}.md`）。

## 怎么用（首次部署端侧 App）

```bash
# 1. 克隆上游样例库到本仓库同级目录（阶段③ 的 npu_tuned_export 导出工程也来自它）
git clone https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp.git \
    ../cannkit_samplecode_lm_engine_cpp

# 2. 套用本仓库的 llm_demo.cpp（二选一）
# 方式 A：直接覆盖（上游演进后仍然可用，推荐）
cp 06_demo_harmony_next_app/llm_demo.cpp \
   ../cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp
# 方式 B：git am 打补丁（保留改动说明，基于上游 762941b；冲突时改用方式 A 手动合并）
cd ../cannkit_samplecode_lm_engine_cpp
git am ../qwen25_1b5_run/06_demo_harmony_next_app/demo_next_llm_demo.patch
```

之后用 DevEco Studio 打开 `CANNLLMEngineDemoNext` 工程构建、安装到手机。

## 工程内 SDK 文件（首次构建前确认）

上游克隆的工程**不含**华为 llm_engine SDK，需从 DDK 包（`DDK-tools-next-6.0.1.0.zip`
的 `ddk/hiai_lm_engine` 目录，下载页见 REPRODUCE.md 前置条件）放入工程两处：

| 文件 | 放到工程内 |
|---|---|
| SDK 头文件（7 个） | `CANNLLMEngineDemoNext/entry/src/main/cpp/include/` |
| `libhiai_llm_engine.so` | `CANNLLMEngineDemoNext/entry/src/main/cpp/lib64/` |

工程兼容性已核对：hvigor 5.0.5 工程 + 本机 DevEco Studio 6.1.1 构建通过
（pages/资源文件完整性一并核对过）。装好 App 后，模型文件推送与运行排障见
`06_demo_harmony_next_app/NEXT_端侧测试手册.md`。

## 文件清单

| 文件 | 用途 |
|---|---|
| `llm_demo.cpp` | 定制版完整源文件（直接覆盖即用） |
| `demo_next_llm_demo.patch` | 同一改动的 git format-patch（`git am` 保留提交说明） |
