# 端侧 App 与实机部署（CANNLLMEngineDemoNext / HarmonyOS NEXT）

本目录 = 端侧 App 的定制源码 + 部署/采集/排障工具 + 操作指南，仅凭本仓库即可复现端侧 App。
模型：Qwen2.5-1.5B(Instruct, int4 g64) · NPU：Kirin 9020 · 工程来自上游样例库。

## 本目录有什么

| 文件 | 用途 |
|---|---|
| `llm_demo.cpp` | 定制版完整源文件（覆盖上游即用；相对上游改了什么见下节） |
| `demo_next_llm_demo.patch` | 同一改动的 git format-patch（`git am` 保留提交说明） |
| `deploy_from_cloud.sh` | **日常部署**：云侧拉取 → md5 校验 → 推送 → 重启验证（一键） |
| `collect_phone_replies.sh` | 真机批量收集回复（端云三方对比用，配合云侧 `02_quant/continuation_compare.py`） |
| `diagnose_load.bat` | 加载失败时重启 App 并抓 hilog 到 `diag/` |

## 相对上游改了什么（基于上游 master `762941b`，真机实测通过）

1. **UTF-8 半字符乱码修复**：流式回调的 token 是字节级 BPE，一个中文字符（3 字节）可能跨两次回调——
   先攒在缓冲区拼完整再上报（`CompleteUtf8PrefixLen` / `g_pendingUtf8`）；顺带修 strdup/delete[] 错配。
2. **instruct chat template 自动包装**：prompt 未含 `<|im_start|>` 时自动按 ChatML 包装
   （Qwen 官方 system prompt + user/assistant 段）——**对话直接输问题即可，无需手动包装**。
3. **自动化测试钩子**：沙箱存在 `test_prompt.txt` 时优先用作 prompt（uitest 无法注入中文），
   完整回复落盘 `last_reply.txt` 供 hdc 拉取。⚠️ **测完删除 test_prompt.txt，否则手动输入被劫持**。
4. InitParam 沙箱路径修正（含缺斜杠笔误）+ 日志格式修正。

## 首次：构建安装 App（一次性）

1. **手机开开发者模式**：设置 → 关于本机 → 连点"版本号"7 次；开发者选项 → 打开 USB 调试；
   USB 连电脑选"传输文件"并允许调试。验证（CMD）：
   ```
   "C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe" list targets
   ```
   能列出序列号即成功。
2. **取工程 + 套定制源码**：
   ```bash
   git clone https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp.git ../cannkit_samplecode_lm_engine_cpp
   cp 06_demo_harmony_next_app/llm_demo.cpp \
      ../cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp
   # 或 git am 06_demo_harmony_next_app/demo_next_llm_demo.patch（冲突时改用 cp）
   ```
3. **工程内放 SDK 文件**（上游工程不含 llm_engine SDK，取自 DDK 包 `ddk/hiai_lm_engine` 目录）：

   | 文件 | 放到工程内 |
   |---|---|
   | SDK 头文件（7 个） | `CANNLLMEngineDemoNext/entry/src/main/cpp/include/` |
   | `libhiai_llm_engine.so` | `CANNLLMEngineDemoNext/entry/src/main/cpp/lib64/` |

   hvigor 5.0.5 工程 + DevEco Studio 6.1.1 构建通过（pages/资源完整性已核对）。
4. **DevEco 编译安装**：Open 打开 `CANNLLMEngineDemoNext` → 首次 Sync 依赖 →
   Project Structure → Signing Configs 勾选 **Automatically generate signature** 登录华为账号（一次性）→
   连手机点 Run ▶。首启因模型未推会显示"模型加载失败"——**预期行为**，继续下一步。

## 日常：部署模型（每次换模型）

**一键部署**（模型文件不落本地，每次从云侧拉最新；云侧为唯一事实源）：

```bash
export CANN_SSH_PASS='云侧密码'          # 不设则 ssh 交互式询问
bash 06_demo_harmony_next_app/deploy_from_cloud.sh    # 默认 instruct；部署 base 则加参数 base
```

自动完成：找 hdc → 检查设备/App → 云侧拉 7 文件逐个 md5 核验 → 推送 → 端上复核 → 重启 App 验证。
约 1.4GB，USB 3.x 约 3-5 分钟（USB 2.0 可能半小时+）。也可用交付目录（05_*）自带的
`push_to_device_next.bat` 推已拷贝的本地目录。

**路径对照**（hdc 与 App 是两套命名空间）：

```
hdc 推送用真实路径:  /data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files
App 内部读沙箱路径:  /data/storage/el2/base/haps/entry/files
```

**验证与性能**：重启 App 第一条消息显示"您好,模型加载完毕"即成功。另开 CMD 看指标：
`hdc shell hilog | findstr LLM_DEMO`（prefillTimeMs=首字延迟；decodeTimeMs per token→tokens/s=1000÷该值）。

## 排障

- **报"设备上不存在 /data/app/el2/100/..."**：App 没装，或多用户索引不是 100——
  `hdc shell "ls /data/app/el2/"` 看实际数字改脚本 `REAL_DIR`。
- **一直"模型加载失败"**：脚本 `ls -lh` 确认 7 文件齐全 → 完全退出 App（任务卡划掉）重开 →
  仍失败用 `diagnose_load.bat` 抓 hilog 看 `LLM Engine, Init...` 后的报错。
- **SubGraph_0.weight 传输截断**（历史案例）：omc 期望 1272421888 字节，少了 6.2MB 报
  `LoadPrivateWeight ... false`——deploy 的 md5/字节核验就是防这个；push 脚本的 `ls` 核验
  只查文件名不查大小，**大文件传输后务必核对**。
- **改生成行为**（max token/贪心等）：改云侧 `05_device_files_*/context.json`、`executor.json`
  重跑 deploy 即可，无需重编 App。
- **instruct 切换**：`SubGraph_0.weight` 与 base 同名，切换 = 整套 7 文件互换（deploy 脚本自动覆盖）。
  采样参数为官方推荐值（top-k 20 / top-p 0.8 / temp 0.7 / rep 1.1，在 `05_device_files_instruct/context.json`）。

## 实测记录（Kirin 9020 真机）

- base（2026-08-17）：输入"你好"，prefill 107ms、decode 27.3ms/token（约 36 tokens/s）。
- instruct（2026-08-17，chat template + test_prompt 钩子）："介绍你自己" 25 token 正常作答、
  `<|im_end|>` 自然停止；"解释模型量化" 297 token 结构化回答、decode 30.2ms/token。

## 变更记录

- 2026-08-18 端云命名统一：本地不再存模型文件（deploy 直拉云侧）；云侧/本地/手机三处文件名一致
  （`Qwen25_1b5_{Base,Instruct}_kirin9020.omc`、`model_{base,instruct}_64_2048.embedding_*`）。
- 2026-08-17 llm_demo.cpp：UTF-8 修复 + chat template 自动包装 + 测试钩子（真机实测通过）；
  `context.json` 内容修正（曾误放 executor 式内容致 `generate_options`/`sampler` 缺失、
  max_gen_tokens 走默认 64——正确内容就是 generate_options + sampler）；
  `stop_sequence` 需含 `<|endoftext|>`（base 不会产出 `<|im_end|>`，只配它则永不停止）；
  executor.json 路径三规则（详见 REPRODUCE 阶段⑥）。
