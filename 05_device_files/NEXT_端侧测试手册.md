# 路线 B:HarmonyOS NEXT 手机端侧测试手册

模型:Qwen2.5-1.5B(int8,Kirin9020 NPU)· 工程:`CANNLLMEngineDemoNext`

## 一、已全部搞定(无需再做)

| 事项 | 位置 |
|---|---|
| SDK 头文件(7 个)已放入工程 | `CANNLLMEngineDemoNext/entry/src/main/cpp/include/` |
| `libhiai_llm_engine.so` 已放入工程 | `CANNLLMEngineDemoNext/entry/src/main/cpp/lib64/` |
| `llm_demo.cpp` InitParam 路径已修正(含缺斜杠笔误) | 指向沙箱 `/data/storage/el2/base/haps/entry/files/` |
| NEXT 专用 context.json(沙箱路径版) | `05_device_files/context_next.json` |
| 一键推送脚本(自动找 hdc、检查设备和 App、推送后核验) | `05_device_files/push_to_device_next.bat`(Windows 双击)/ `.sh`(Git Bash) |
| 工程完整性核对(hvigor 5.0.5 / pages / 资源文件) | 与本机 DevEco Studio 6.1.1 兼容 |

路径对照(记住这两行即可):

```
hdc 推送用真实路径:  /data/app/el2/100/base/com.huawei.cannkit.llmengine/haps/entry/files
App 内部读沙箱路径:  /data/storage/el2/base/haps/entry/files
```

## 二、只剩这些步骤需要你动手

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

Windows 直接双击:

```
05_device_files\push_to_device_next.bat
```

(或在 CMD/PowerShell 里执行 `C:\Users\Yaphet\Downloads\cannkit_samplecode_lm_engine_cpp\05_device_files\push_to_device_next.bat`)

脚本会自动:找 hdc → 检查设备 → 检查 App 已安装(沙箱目录存在)→ 推 7 个文件
(其中 `context_next.json` 会以 `context.json` 的名字写入设备)→ 放权限 → 逐个核验文件确实到位。
用 Git Bash 的话也可以跑 `bash 05_device_files/push_to_device_next.sh`,逻辑相同。

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
- **想改生成行为**(最大 token 数、贪心解码等):改 `05_device_files/executor.json`
  后重推该文件并重启 App,无需重新编译。

## 附:改了哪些文件

- `CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp`:InitParam 笔误修正 + 日志格式修正
- `CANNLLMEngineDemoNext/entry/src/main/cpp/include/`、`lib64/`:新增 SDK(来自 CANN-Kit-next-6.0.1.0.zip)
- `05_device_files/context_next.json`:新增(NEXT 沙箱路径版 context)
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
