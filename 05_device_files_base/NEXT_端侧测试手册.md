# 路线 B:HarmonyOS NEXT 手机端侧测试手册

模型:Qwen2.5-1.5B(int8,Kirin9020 NPU)· 工程:`CANNLLMEngineDemoNext`

## 一、已全部搞定(无需再做)

| 事项 | 位置 |
|---|---|
| SDK 头文件(7 个)已放入工程 | `CANNLLMEngineDemoNext/entry/src/main/cpp/include/` |
| `libhiai_llm_engine.so` 已放入工程 | `CANNLLMEngineDemoNext/entry/src/main/cpp/lib64/` |
| `llm_demo.cpp` InitParam 路径已修正(含缺斜杠笔误) | 指向沙箱 `/data/storage/el2/base/haps/entry/files/` |
| 一键部署脚本(云侧拉取 → md5 校验 → 推送 → 重启验证) | `05_device_files/deploy_from_cloud.sh` |
| 工程完整性核对(hvigor 5.0.5 / pages / 资源文件) | 与本机 DevEco Studio 6.1.1 兼容 |

路径对照(记住这两行即可):

```
hdc 推送用真实路径:  /data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files
App 内部读沙箱路径:  /data/storage/el2/base/haps/entry/files
```

## 二、只剩这些步骤需要你动手

### 步骤 0:一键部署(日常换模型就用这个,前面步骤只在首次需要)

**工作原则:以云侧为主。** 模型文件、配置、脚本、报告都以云侧 `qwen25_1b5_run/` 为准;
本地不存放模型文件、不另起副本,只有本地做的 bug fix 需要回同步到云侧。

云侧流水线产出 device 文件后,本地一条命令完成"拉取 → md5 校验 → 推送 → 重启验证":

```bash
export CANN_SSH_PASS='云侧密码'          # 不设则 ssh 交互式询问
bash 05_device_files/deploy_from_cloud.sh base       # 或 instruct
```

脚本特性:逐文件 md5/大小核验(防传输截断)、端云文件名完全一致(云侧叫什么,手机就叫什么)、
临时文件自动清理。看到 `[DONE]` 即部署成功。

### 步骤 1:手机开开发者模式并连接

1. 设置 → 关于本机 → 连点"版本号"7 次开启开发者模式
2. 设置 → 系统和更新 → 开发者选项 → 打开"USB 调试"
3. USB 线连电脑,手机弹窗选择"传输文件"并允许 USB 调试

验证(CMD 或 PowerShell):

```
"C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe" list targets
```

能列出序列号(而不是 `[Empty]` 或空输出)即成功。

### 步骤 2:DevEco Studio 编译安装 App(需要你的华为账号签名)

1. 打开 DevEco Studio → File → Open → 选择
   `CANN_LLM\CANN_LLM_Engine_Demo\CANNLLMEngineDemoNext` 文件夹
2. 首次打开会提示 Sync/Install 依赖,点确认等待完成
3. 签名(一次性):File → Project Structure → Signing Configs →
   勾选 **Automatically generate signature** → 登录华为账号 → OK
4. 手机保持连接,点右上角 **Run ▶**(或 Shift+F10)
   —— 会自动编译 native so + 打 HAP 包 + 安装 + 启动

首次启动因为模型还没推进去,界面第一条消息会显示
"对不起,模型加载失败,请重试。"——**这是预期的**,继续步骤 3。

### 步骤 3:推送模型文件(共约 1.4GB,大头是 SubGraph_0.weight 1.18GB)

用步骤 0 的一键部署脚本即可(模型文件不留在本地,每次从云侧拉最新):

```bash
export CANN_SSH_PASS='云侧密码'
bash 05_device_files/deploy_from_cloud.sh base       # 或 instruct
```

脚本会自动:找 hdc → 检查设备和 App 已安装(沙箱目录存在)→ 从云侧拉 7 个文件并逐文件 md5 核验
→ 推送手机(文件名与云侧完全一致)→ 端上复核 → 重启 App 验证加载。
USB 3.x 口+线推送约 3-5 分钟;若是 USB 2.0 可能要半小时以上,建议换 USB 3 接口。

### 步骤 4:重启 App 测试

推送完成后,从桌面重新打开 App(cannkit llmengine 图标):

- 第一条消息显示 **"您好,模型加载完毕,您可以提问了。"** = 模型加载成功
- 输入问题 → 发送 → 流式返回回答

### 步骤 5:看性能指标(另开一个 CMD 窗口)

```
"C:\Program Files\Huawei\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe" shell hilog | findstr LLM_DEMO
```

每次生成结束会打印:

| 指标 | 含义 |
|---|---|
| `prefillTimeMs` | 首字延迟(输入 prefill 耗时) |
| `decodeTimeMs per token` | 单 token 解码耗时 → tokens/s = 1000 ÷ 该值 |
| `inputTokenCount / outputTokenCount` | 输入/输出 token 数 |
| `totalTimeMs` | 端到端总耗时 |

## 三、常见问题

- **push 脚本报"设备上不存在 /data/app/el2/100/..."**:
  App 没装或路径里 `100` 不是当前用户索引。先确认 App 已安装:
  `hdc shell bm dump -n com.huawei.cannkit.llmengine`,若手机多用户导致索引不同,
  用 `hdc shell "ls /data/app/el2/"` 看实际数字并修改脚本里的 `REAL_DIR`。
- **推送慢**:共约 1.4GB。USB 3.x 口+线约 3-5 分钟;若是 USB 2.0 可能要半小时以上,
  建议换 USB 3 接口。
- **App 一直"模型加载失败"**:脚本最后 `ls -lh` 确认 7 个文件齐全后,
  完全退出 App(任务卡片划掉)再重新打开;仍失败则看 hilog 里 `LLM Engine, Init...` 之后的报错。
- **想改生成行为**(最大 token 数、贪心解码等):改云侧 `05_device_files_base/`(或 `_instruct/`)下的
  `executor.json` / `context.json`,重新跑一遍 deploy 脚本即可,无需重新编译。

## 四、instruct 版本(2026-08-17 已实测通过)

- instruct 模型文件在云侧 `05_device_files_instruct/`(omc + SubGraph_0.weight + model_instruct_* embedding
  + tokenizer + executor.json + context.json),用 `deploy_from_cloud.sh instruct` 部署;注意 `SubGraph_0.weight`
  与 base 同名,切换模型就是整套文件互换(deploy 脚本会覆盖,base/instruct 随时互切)。
- **instruct 需要 chat template 才有正常对话行为**。demo 把输入框文本原样当 prompt,
  手动聊天前需在 `llm_demo.cpp` 的 ModelInfer 里给输入套
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`,或临时用 test_prompt.txt 钩子传入包好模板的文本。
- 实测(套 chat template + test_prompt.txt 钩子):"你好,请用一句话介绍你自己" → 25 token 正常作答,
  `<|im_end|>` 自然停止;"请用中文解释什么是模型量化" → 297 token 结构化回答,decode 30.2ms/token。
- instruct 采样参数(Qwen2.5 官方推荐): top-k 20 / top-p 0.8 / temp 0.7 / rep 1.1,见
  云侧 `05_device_files_instruct/context.json`;`stop_sequence` 用 `<|im_end|>` 即可(instruct 会自然产出)。


## 附:改了哪些文件

- **2026-08-18 端云命名统一**:本地不再存放模型文件(每次由 `deploy_from_cloud.sh` 从云侧拉取),
  删除旧命名残留(`Qwen25_1b5_kirin9020.omc` / `model_64_2048.embedding_*` / `context_next.json` 等约 1.5GB)
  及被 deploy 脚本取代的 `push_to_device*.sh/.bat`、`pack.sh`;`run_device_compare.sh` 更名为
  `collect_phone_replies.sh` 与云侧同名;手机上旧命名文件同步删除。现在云侧、本地脚本、手机三处文件名完全一致
  (`Qwen25_1b5_Base/Instruct_kirin9020.omc`、`model_base/instruct_64_2048.embedding_*`、`context.json`)。
- `CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp`:InitParam 笔误修正 + 日志格式修正
- `CANNLLMEngineDemoNext/entry/src/main/cpp/include/`、`lib64/`:新增 SDK(来自 CANN-Kit-next-6.0.1.0.zip)
- `05_device_files/context_next.json`:NEXT 版 context(2026-08-17 **内容修正**):
  - **正确内容 = `generate_options` + `sampler`**(与 context.json 相同,context 里没有路径,无需"沙箱路径版")
  - 之前误放成 executor 式内容(llm_config/tokenizer/autoregressive),导致 `generate_options`/`sampler` 全部缺失,
    引擎走默认值:**max_gen_tokens 默认 64(输出被截断到 64 token)+ 采样参数丢失(与云侧 seed99/topk16/topp0.95/temp0.6/rep1.2 对不上)**
  - `stop_sequence` 补上 `"<|endoftext|>"`:base 模型不会产出 `<|im_end|>`(那是 instruct 的对话结束符),
    只配 `<|im_end|>` 时生成永不停止、每次都跑满 max_gen_tokens,且 `<|endoftext|>` 会以字面文本混入输出
- `llm_demo.cpp`(2026-08-17):修复流式回调 UTF-8 半字符乱码(native 侧残留缓冲)+ strdup/delete[] 错配;
  新增自动化测试钩子:沙箱存在 `test_prompt.txt` 时优先用作 prompt(uitest 无法注入中文),
  完整回复落盘 `last_reply.txt` 供 hdc 拉取对比。**注意:测试完删除 test_prompt.txt,否则手动输入会被劫持**
- `05_device_files/executor.json` + `context_next.json`(2026-08-17 修正):
  - `autoregressive.model_path` / `weight_path` 改为沙箱绝对路径(原 `/data/local/tmp/qwen25_1b5` 是路线 A 调试路径,App 无权读取,会报 `FileUtil::realpath: ERR` / `model config path error`)
  - `tokenizer.path` 必须为**绝对路径**(相对路径 realpath 直接失败)
  - `embedding_weights` / `embedding_dequant_scale` 必须保持**相对文件名**——引擎会自动拼 `weight_path` 前缀,写绝对路径会被拼成双重路径
- `05_device_files/push_to_device_next.bat`:新增(Windows 双击版推送脚本)
- `05_device_files/push_to_device_next.sh`:新增(Git Bash 版,逻辑相同)
- `05_device_files/push_to_device.sh`:路线 A 用的推送脚本(备用)
- 仓库根目录的 `CANN-Kit-next-6.0.1.0.zip` 已解压完毕,可删可留

## 已解决的问题(2026-08-17)

- `SubGraph_0.weight` 与 `.omc` 不匹配:omc 期望权重 1272421888 字节,本地文件只有 1265893376 字节(少约 6.2MB,
  传输截断),引擎报 `LoadPrivateWeight(419): weightFileSize - constSize >= variableSize false`。
  **已解决**:从量化服务器 `test_omc/qwen25_1b5_run/04_omc_convert/Qwen25_1b5_kirin9020/` 重新拷贝完整文件
  (1272421888 字节,md5 `364df7f5f2aba9fc5cf080b6034b871b`),推送后模型加载成功。
- 端到端验证通过(2026-08-17):输入"你好",prefill 107ms,decode 27.3ms/token(约 36 tokens/s),输出 64 tokens。

## 防止再次踩坑

- 从服务器拷贝大文件后**务必核对字节数或 md5**:`SubGraph_0.weight` 应为 1272421888 字节。
- 推送设备后脚本最后的 `ls -l` 核验只查文件名存在,不查大小;文件被截断时核验会通过但加载会失败。
