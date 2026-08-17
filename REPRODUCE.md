# kirin9020 平台 Qwen2.5-1.5B（base + instruct 双产线）全流程复现指引（操作手册）

> 本文只讲**怎么跑**。量化机制、为什么这样配、实验数据、踩坑记录见 **QUANTIZATION.md**；git 工作流见 **VERSION_CONTROL.md**。
> **两条产线、各自最优配方**（统一入口 `pipeline.sh`，详见下节）：
> - **base 交付版 = Qwen2.5-1.5B + g128 + zh维基校准**（续写演示）：中文 PPL 14.76（劣化 17.1%），SubGraph 1.27G；
> - **instruct 交付版 = Qwen2.5-1.5B-Instruct + g64 + zh对话校准**（ChatML，对话演示）：配方与评测见 QUANTIZATION.md §八。
> - 历史口径：英文最优版 = g64 + wikitext2（英文 PPL 19.66，劣化 12.44%）。
> `05_device_files_base/` 与 `05_device_files_instruct/` 中是两套成品（文件名带 `_Base_` / `_Instruct_` 标记）。

## 统一产线入口（演示推荐用法）

```bash
bash pipeline.sh base              # base 全流程（量化已有产物会自动跳过）
bash pipeline.sh instruct          # instruct 全流程
bash pipeline.sh <base|instruct> [quant|eval|export|convert|pack|status]   # 单阶段
FORCE=1 bash pipeline.sh <p> quant # 强制重量化
```

- 同一套阶段脚本（02~05），差异全部收在 `profiles/{base,instruct}.env`：模型 / group_size / 校准语料 / 导出 yaml / 产物命名 / 交付目录。
- **演示流程**：先 `bash pipeline.sh base` 再 `bash pipeline.sh instruct`，产物分别落在 `05_device_files_base/`、`05_device_files_instruct/`；两个目录各自拷到连手机的 Windows 机，分别跑各自的 `push_to_device_next.bat`（换模型必须整组 7 文件重推，`SubGraph_0.weight` 两模型同名）。
- **命名约定**（base/instruct 一眼可区分）：omc `Qwen25_1b5_{Base,Instruct}_kirin9020.omc`；embedding `model_{base,instruct}_64_2048.embedding_*`；量化工程 `02_quant/{exp_g128_zh,qwen25_1b5_instruct_9020}`。

## 现场演示预检

1. **GPU**：`nvidia-smi` 确认空闲。撰写本文时有 14.5G 常驻 python + 1.9G lmstudio 占卡（此前 c1024 实验 OOM 的元凶），跑阶段②前需协调释放。c512 档在 ~14G 空闲下实测可跑（instruct 产线 2026-08-17 验证）。
2. 磁盘 ≥40G 余量（6G×2 pth + 6.7G fake_quant + 5.9G pb 峰值叠加）。
3. **防误重量化**：dopt 工程目录里 `dopt_config.json` 已存在时，第一次 `run.sh stage1` 不会重新生成配置而是直接进入 19 分钟重量化——手动重跑生成配置步骤前先删对应 testcase 目录；不想重跑量化则从 stage3 开始（trained.pth 已在）。`pipeline.sh` 已内置该防呆（未完成的 testcase 自动清除重跑），`run.sh` 裸跑也强制要求显式 `TESTCASE=`。
4. `04_omc_convert/` 的 `model.onnx/.pb/quant_params_file` 是转换工作副本，转换后可删（2026-08-17 已清理，
   含历史多档实验失败残留 model128.*/omg_*.log——档位锁定教训见 QUANTIZATION.md §三）。

## 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA GPU；**sm_120(Blackwell/RTX 5090) 必须 torch 2.8+cu128**，sm_90 及以下可用 torch 2.4 |
| 系统 | Linux x86_64，glibc ≥ 2.35，`uv`、`unzip` |
| 依赖包 | `dependencies/DDK-tools-next-6.0.1.0.zip` + `kirin9020-plugin-next-6.0.1.0.zip`（仅此两个，无需 CANN toolkit） |
| 源码 | 同仓库 `cannkit_samplecode_lm_engine_cpp/`（提供 npu_tuned_export 导出工程） |

## 目录结构

```
qwen25_1b5_run/
├── REPRODUCE.md / QUANTIZATION.md / VERSION_CONTROL.md
├── pipeline.sh                   # ★ 统一产线入口（base|instruct × 各阶段）
├── profiles/                     # ★ 产线差异声明（模型/group/语料/命名/交付目录）
│   ├── base.env  instruct.env
├── 01_prepare/                   # 阶段① 环境
│   ├── make_venv.sh  prepare.sh
│   ├── venv/ tools/ cuda_stub/   # (生成)
│   └── models/Qwen2.5-1.5B/  Qwen2.5-1.5B-Instruct/   # (生成/下载)
├── 02_quant/                     # 阶段② 量化（三段式，run.sh 吃 TESTCASE/CFG/MODEL 环境变量）
│   ├── config.yaml               #   base 默认配方（g128+zh维基，历史）
│   ├── instruct_config.yaml      #   instruct 配方（g64+zh对话校准）
│   ├── data_zh/                  #   base 校准语料（zh维基 2408 行）
│   ├── data_chat/                #   instruct 校准语料（Belle 多轮+单轮 ChatML 2401 行）
│   ├── run.sh  edit_dopt_config.py  run_experiment.sh
│   ├── eval_ppl.py  chat_test.py  device_compare.py
│   ├── _autopatch/               #   transformers 4.51 兼容补丁（KD 用；PTQ 也无害）
│   ├── exp_g128_zh/              #   (生成) base 交付量化工程（g128+zh）
│   └── qwen25_1b5_instruct_9020/ #   (生成) instruct 量化工程（g64+chat）
├── 03_onnx_export/               # 阶段③ export.sh（EXPORT_YAML 可换）+ base/instruct 两份 yaml
├── 04_omc_convert/               # 阶段④ convert.sh（OUTPUT_PREFIX 可换）
├── 05_device_files_base/         # 阶段⑤ base 交付目录（7 文件 + push 脚本 + 测试手册）
├── 05_device_files_instruct/     # 阶段⑤ instruct 交付目录（7 文件 + push 脚本 + README_DEMO）
└── logs/                         # 各阶段日志
```

---

## 阶段 ① 准备（venv/tools/models 已就绪时可跳过）

```bash
bash 01_prepare/make_venv.sh     # uv 建 venv，torch 2.8（输出应含 sm_120）
source 01_prepare/venv/bin/activate
bash 01_prepare/prepare.sh       # 解压 DDK+插件→tools/、下模型、patchelf 修 omg、cuda_stub
```
prepare.sh 四件事：组装 `tools/`（插件进 platform）→ 下载 Qwen2.5-1.5B 与 Qwen2.5-1.5B-Instruct 两个源模型（各 ~3GB，已存在跳过）→ **patchelf 把 omg interpreter 改系统 ld**（原始指向 /tmp 软链会 EACCES）→ 建 stub CUDA_HOME（过 deepspeed 的 CUDA_HOME 检查）。

## 阶段 ② 量化（三段式，GPU，~35 分钟）

> **推荐走统一入口**：`bash pipeline.sh <base|instruct> quant`（自动完成下面 5 步 + 防呆）。
> 校准语料重建（可跳过，若 `data_zh/dataset_zh.json` / `data_chat/dataset_chat_zh.json` 已在）：
> `python 02_quant/data_zh/build_corpus.py`（zh 维基） / `python 02_quant/data_chat/build_corpus_chat.py`（Belle 对话，ChatML 渲染）。
> 手动方式（以 base 为例；instruct 把 testcase/config/model 换成 instruct 版，或看 pipeline.sh do_quant）：

```bash
source 01_prepare/venv/bin/activate
cd 02_quant
TESTCASE=exp_g128_zh CFG=config.yaml bash run.sh stage1   # ①生成 dopt_config.json（198 节点全 float）
python edit_dopt_config.py exp_g128_zh/dopt_config.json 128 --keep-lm-head-fp
                                 # ②改策略：embed=MinMax, 196 linear=eco(g128,in16), lm_head=float
TESTCASE=exp_g128_zh CFG=config.yaml bash run.sh stage1   # ③权重量化 GPTQ ~19min → weight quant done!!!
TESTCASE=exp_g128_zh CFG=config.yaml bash run.sh stage2   # ④激活校准 EMA ~10min → quant done !!!
TESTCASE=exp_g128_zh CFG=config.yaml bash run.sh stage3   # ⑤参数提取 ~4min → quant params file build done
```

**（可选）验证精度与生成质量**
```bash
python eval_ppl.py ../01_prepare/models/Qwen2.5-1.5B \
  exp_g128_zh/dopt_config.json exp_g128_zh/train_output/trained.pth \
  --tag repro --n_samples 8192 --data data_zh/test_zh.txt   # 交付版中文口径（QUANTIZATION.md §三）
python device_compare.py                                     # base 续写对比（端侧采样参数复刻）
python chat_test.py ../01_prepare/models/Qwen2.5-1.5B-Instruct \
  qwen25_1b5_instruct_9020/dopt_config.json \
  qwen25_1b5_instruct_9020/train_output/trained.pth          # 对话对比（chat_test 内置 chat template，instruct 用）
```

## 阶段 ③ ONNX 导出（GPU，~3 分钟）
```bash
bash 03_onnx_export/export.sh                          # base（默认 model_info_target.yaml）
EXPORT_YAML=model_info_instruct.yaml OUT_SUBDIR=output_instruct_embedding_out_no_output_pos \
  bash 03_onnx_export/export.sh                        # instruct（pipeline.sh 自动传）
```

## 阶段 ④ OMC 转换（~2 分钟）
```bash
bash 04_omc_convert/convert.sh                         # base（默认前缀 Qwen25_1b5_Base_kirin9020）
QUANT_DIR=$PWD/02_quant/qwen25_1b5_instruct_9020 \
ONNX_DIR=$PWD/03_onnx_export/output_instruct_embedding_out_no_output_pos \
OUTPUT_PREFIX=./Qwen25_1b5_Instruct_kirin9020 \
  bash 04_omc_convert/convert.sh                       # instruct（pipeline.sh 自动传）
```

## 阶段 ⑤ 端侧 7 文件
```bash
bash 05_device_files_base/pack.sh                      # base（默认参数）
bash pipeline.sh instruct pack                          # instruct（自动传参复用同一 pack.sh）
```

**手机实机部署（路线 B / HarmonyOS NEXT，已在 Kirin9020 真机验证通过）**：

1. `executor.json` / `context.json` 已改为 App 沙箱路径（`/data/storage/el2/base/haps/entry/files/`），
   注意三点：`model_path`/`weight_path` 用沙箱绝对路径；`tokenizer.path` 必须绝对路径；
   `embedding_weights`/`embedding_dequant_scale` 必须保持相对文件名（引擎自动拼 weight_path 前缀）。
2. 在连手机的 Windows 机器上运行交付目录里的 `push_to_device_next.bat`（或 Git Bash 跑 `.sh`）推送 7 文件，
   脚本自动识别目录里的 Base/Instruct 文件名并核验。详细步骤与排错见 `05_device_files_base/NEXT_端侧测试手册.md`。
3. 从本服务器拷贝 `SubGraph_0.weight` 后**务必核对大小**（base g128 = 1272421888 字节；instruct g64 见 pack 输出，
   曾发生传输截断少 6.2MB 导致 `LoadPrivateWeight` 失败）。
4. **换模型演示**：进另一个交付目录重跑 push 脚本（7 文件整组覆盖），完全退出并重启 App。

旧的路线 A（`/data/local/tmp/qwen25_1b5/` + Demo 工程）仅作备用。

---

## 验证清单（每阶段成功标志与产物大小）

| 阶段 | 成功标志 | 产物（实测大小） |
|---|---|---|
| ②.1 | `generate plugin quang config please set quant strategy firstly` | dopt_config.json |
| ②.2 | 分布 `{Quant_Embed_MinMax:1, Quant_act_weight_eco:196, float:1}` | — |
| ②.3 | `weight quant done!!!` | trained_quant_weight.pth ~6.0G |
| ②.4 | `quant done !!!` | trained.pth ~6.0G |
| ②.5 | `quant params file build done` | **quant_params_file：g128≈556M / g64≈1.1G**（690M 说明 quant_param_2 配错）+ fake_quant_weight.pth ~6.7G + embedding 223M/594K |
| ③ | `generating finished` | model.onnx 439K + model.pb ~5.9G + model_64_2048.embedding_*（pack 时改名 model_{base,instruct}_*） |
| ④ | `generate offline model success`，日志 **s16s4 融合数千次（g128 3136 / g64 4704）**、don't support=0 | **omc ~3.7M + SubGraph_0.weight：g128 1.27G / g64 1.35G** |
| ⑤ | 7 文件齐全（omc/embedding 文件名带 `_Base_`/`_Instruct_` 标记） | 总 ~1.5G（g64 版 ~1.6G） |

## 常见问题（详见 QUANTIZATION.md 四）

| 症状 | 原因/解法 |
|---|---|
| `No module named 'datasets'` / deepspeed CUDA_HOME 报错 | 未激活 venv；run.sh 已内置 CUDA_HOME stub |
| omg `Permission denied` | interpreter 未 patchelf（重跑 prepare.sh 第4步） |
| omg 全部 `MatMul don't support` | quant_param_2=True 或加了 output 段 → 改回 False/删 output |
| 评测/仿真 PPL=nan | fp16 溢出 → 必须 fp32（eval_ppl.py 默认已是） |
| 量化/评测 OOM | GPU 被他人占用（nvidia-smi 查）→ 等空闲或降 cutoff_len |
| 对话测试输出乱码 | base 模型不支持 chat template（坑：chat_test 内置 template）→ base 用 device_compare.py 续写口径；chat 对话用 instruct 产线 |
| torch CUDA `no kernel image` | torch 版本不含你的 sm 架构（5090 需 2.8+cu128） |
| 想改 prefill 档位（16/128/三档） | **不支持**——档位在 dopt stage3 的 quant_params_file 里登记锁定为 {decode=1, prefill=64}，多档 omg 编译必失败（QUANTIZATION.md §三 prefill 档位实测） |
