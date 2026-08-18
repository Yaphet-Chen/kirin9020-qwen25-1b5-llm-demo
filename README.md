# kirin9020-qwen25-1b5-llm-demo

**教学 demo：把 Qwen2.5-1.5B 跑上 Kirin 9020（HarmonyOS NEXT）的端侧 LLM 服务全流程。**

从 HuggingFace 原始模型出发，走完「量化 → ONNX 导出（NPU 亲和）→ OMC 离线模型转换 → 端侧部署」
完整链路，在手机上以 NPU 加速跑通 **base（续写）** 与 **instruct（ChatML 对话）** 两条产线。
已在 Kirin 9020 真机验证通过。

## 成果速览

| 产线 | 配方 | 精度 | 端侧产物 |
|---|---|---|---|
| base（续写） | Qwen2.5-1.5B + g128 量化 + zh 维基校准 | 中文 PPL 14.76（较 fp32 劣化 17.1%） | omc ~3.7M + weight 1.27G |
| instruct（对话） | Qwen2.5-1.5B-Instruct + g64 量化 + zh 对话校准 | 评测见 QUANTIZATION.md §八 | omc ~3.7M + weight 1.35G |

端云一致性有三方对比报告（云侧 fp / 云侧量化仿真 / 端侧真机）：
`02_quant/reports/report_base_3way_{greedy,sampling}.md`。

## 快速上手

```bash
bash pipeline.sh base        # base 全流程：量化→导出→转换→汇集交付目录
bash pipeline.sh instruct    # instruct 全流程
bash pipeline.sh <base|instruct> [quant|eval|export|convert|pack|status]   # 单阶段
```

产物落在 `05_device_files_base/`、`05_device_files_instruct/`（各 7 文件 + 推送脚本），
拷到连手机的 Windows 机跑 `push_to_device_next.bat` 即可部署。
**完整操作手册（含每阶段成功标志、产物大小核对、常见问题）见 [REPRODUCE.md](REPRODUCE.md)。**

## 前置条件（摘要）

| 环境 | 要求 |
|---|---|
| 云侧（量化/导出/转换） | Linux x86_64 + NVIDIA GPU（Blackwell/RTX 5090 需 torch 2.8+cu128），磁盘 ≥40G |
| 端侧（部署） | Windows + DevEco Studio + hdc，Kirin 9020 手机（HarmonyOS NEXT） |

外部依赖（均不随仓库分发，获取方式见 REPRODUCE.md 前置条件）：

- **DDK 工具包 + kirin9020 平台插件包**：[官方下载页](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations)，
  放本仓库旁的 `dependencies/`；
- **上游样例库** `cannkit_samplecode_lm_engine_cpp`（gitcode，公开）：克隆到本仓库同级目录，
  提供 ONNX 导出工程（npu_tuned_export）与端侧 App 工程骨架；
- **源模型** Qwen2.5-1.5B / Qwen2.5-1.5B-Instruct：`01_prepare/prepare.sh` 自动下载。

**本仓库已内置端侧 App 的定制改动**（`06_demo_harmony_next_app/`：UTF-8 流式半字符乱码修复、
instruct ChatML 自动包装、端云对比自动化测试钩子）——克隆者仅需上述公开上游 + 本仓库
即可复现全部内容，详见 [06_demo_harmony_next_app/README.md](06_demo_harmony_next_app/README.md)。

## 文档导航

| 文档 | 内容 |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | 全流程操作手册（怎么跑） |
| [QUANTIZATION.md](QUANTIZATION.md) | 量化机制 / 实验矩阵 / 最优配方 / 踩坑记录（为什么这样配） |

## 致谢

- 上游样例与解决方案文档：[HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp](https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp)（gitcode）
- 模型：[Qwen2.5](https://huggingface.co/Qwen) by Alibaba Qwen team
