# kirin9020 平台 Qwen2.5-1.5B 全流程复现指引（操作手册）

> 本文只讲**怎么跑**。量化机制、为什么这样配、实验数据、踩坑记录见 **QUANTIZATION.md**；git 工作流见 **VERSION_CONTROL.md**。
> **两个配方口径**（取舍依据见 QUANTIZATION.md §三 中英 2×2 矩阵）：
> - **当前交付版 = g128 + 中文校准**（`02_quant/config.yaml` 现状）：中文 PPL 14.76（劣化 17.1%），SubGraph 1.27G；
> - **英文最优版 = g64 + wikitext2**：英文 PPL 19.66（劣化 12.44%）。
> `05_device_files/` 中的成品是**交付版**（omc 3.7M + weight 1.27G）。

## 现场演示预检

1. **GPU**：`nvidia-smi` 确认空闲。撰写本文时有 14.5G 常驻 python + 1.9G lmstudio 占卡（此前 c1024 实验 OOM 的元凶），跑阶段②前需协调释放。
2. 磁盘 ≥40G 余量（6G×2 pth + 6.7G fake_quant + 5.9G pb 峰值叠加）。
3. 演示 stage1 首跑（生成配置步骤）前先 `rm -rf 02_quant/qwen25_1b5_9020`——**dopt_config.json 已存在时，第一次 `run.sh stage1` 不生成配置而直接进入 19 分钟重量化**。不想重跑量化则从 stage3 开始（trained.pth 已在）。
4. `04_omc_convert/` 下的 `model128.onnx/.pb` 与 `omg_t16/t128/3tier/e128.log` 是分档实验的**失败残留**（4 个日志结尾均 `return FAIL`），不在交付链路上；交付产物 = `Qwen25_1b5_kirin9020/`。

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
├── 01_prepare/                  # 阶段① 环境
│   ├── make_venv.sh  prepare.sh
│   ├── venv/ tools/ models/Qwen2.5-1.5B/ cuda_stub/   # (生成)
├── 02_quant/                    # 阶段② 量化
│   ├── config.yaml              #   交付版配方（g128+中文校准/c512/s1024/quant_param_2=False）
│   ├── data_zh/                 #   中文校准语料（dataset_zh.json，2408 条记录）
│   ├── run.sh                   #   三段式入口（TESTCASE/CFG 环境变量可覆盖）
│   ├── edit_dopt_config.py      #   策略编辑（group/lm_head保fp/策略替换）
│   ├── run_experiment.sh        #   试验驱动（自动 config+三段式+PPL 评测）
│   ├── eval_ppl.py              #   PPL 精度评测（fp32 仿真）
│   ├── chat_test.py             #   生成质量测试（纯续写，base 模型勿用 chat template）
│   ├── _autopatch/              #   transformers 4.51 兼容补丁（KD 用；PTQ 也无害）
│   └── qwen25_1b5_9020/         #   (生成) 量化工程
├── 03_onnx_export/              # 阶段③ export.sh + model_info_target.yaml → (生成) onnx
├── 04_omc_convert/              # 阶段④ convert.sh → (生成) Qwen25_1b5_kirin9020/
├── 05_device_files/             # 阶段⑤ pack.sh + 端侧 7 文件（最终交付）
└── logs/                        # 各阶段日志
```

---

## 阶段 ① 准备（venv/tools/models 已就绪时可跳过）

```bash
bash 01_prepare/make_venv.sh     # uv 建 venv，torch 2.8（输出应含 sm_120）
source 01_prepare/venv/bin/activate
bash 01_prepare/prepare.sh       # 解压 DDK+插件→tools/、下模型、patchelf 修 omg、cuda_stub
```
prepare.sh 四件事：组装 `tools/`（插件进 platform）→ 下载 Qwen2.5-1.5B（3GB）→ **patchelf 把 omg interpreter 改系统 ld**（原始指向 /tmp 软链会 EACCES）→ 建 stub CUDA_HOME（过 deepspeed 的 CUDA_HOME 检查）。

## 阶段 ② 量化（三段式，GPU，~35 分钟）

> 校准语料：交付版用 `data_zh/dataset_zh.json`（config.yaml 现状）；英文版把 `train_files` 改回 `wikitext2`。
> group_size：交付版 **128**，英文最优版 **64**（下按交付版写，英文版把 128 换 64）。

```bash
source 01_prepare/venv/bin/activate
cd 02_quant
bash run.sh stage1               # ①生成 dopt_config.json（198 节点全 float；重跑需先删 qwen25_1b5_9020/，见预检3）
python edit_dopt_config.py qwen25_1b5_9020/dopt_config.json 128 --keep-lm-head-fp
                                 # ②改策略：embed=MinMax, 196 linear=eco(g128,in16), lm_head=float
bash run.sh stage1               # ③权重量化 GPTQ ~19min → weight quant done!!!
bash run.sh stage2               # ④激活校准 EMA ~10min → quant done !!!
bash run.sh stage3               # ⑤参数提取 ~4min → quant params file build done
```

**（可选）验证精度与生成质量**
```bash
python eval_ppl.py ../01_prepare/models/Qwen2.5-1.5B \
  qwen25_1b5_9020/dopt_config.json qwen25_1b5_9020/train_output/trained.pth \
  --tag repro --n_samples 8192   # 期望 QUANT PPL ≈ 19.66（FP=17.49）
python chat_test.py  ../01_prepare/models/Qwen2.5-1.5B \
  qwen25_1b5_9020/dopt_config.json qwen25_1b5_9020/train_output/trained.pth   # 续写对比
```

## 阶段 ③ ONNX 导出（GPU，~3 分钟）
```bash
cd .. && bash 03_onnx_export/export.sh    # 日志: generating finished
```

## 阶段 ④ OMC 转换（~2 分钟）
```bash
bash 04_omc_convert/convert.sh            # 日志: OMG generate offline model success
```

## 阶段 ⑤ 端侧 7 文件
```bash
bash 05_device_files/pack.sh
```

**手机实机部署（路线 B / HarmonyOS NEXT，已在 Kirin9020 真机验证通过）**：

1. `executor.json` / `context_next.json` 已改为 App 沙箱路径（`/data/storage/el2/base/haps/entry/files/`），
   注意三点：`model_path`/`weight_path` 用沙箱绝对路径；`tokenizer.path` 必须绝对路径；
   `embedding_weights`/`embedding_dequant_scale` 必须保持相对文件名（引擎自动拼 weight_path 前缀）。
2. 在连手机的 Windows 机器上运行 `push_to_device_next.bat`（或 Git Bash 跑 `.sh`）推送 7 文件，
   脚本会自动核验。详细步骤与排错见 `05_device_files/NEXT_端侧测试手册.md`。
3. 从本服务器拷贝 `SubGraph_0.weight` 后**务必核对大小 = 1272421888 字节**
   （曾发生传输截断少 6.2MB 导致 `LoadPrivateWeight` 失败）。

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
| ③ | `generating finished` | model.onnx 439K + model.pb ~5.9G + model_64_2048.embedding_* |
| ④ | `generate offline model success`，日志 **s16s4 融合数千次（g128 交付版 3136 / g64 版 4704）**、don't support=0 | **omc ~3.7M + SubGraph_0.weight：g128 1.27G / g64 1.35G** |
| ⑤ | 7 文件齐全 | 总 ~1.5G |

## 常见问题（详见 QUANTIZATION.md 四）

| 症状 | 原因/解法 |
|---|---|
| `No module named 'datasets'` / deepspeed CUDA_HOME 报错 | 未激活 venv；run.sh 已内置 CUDA_HOME stub |
| omg `Permission denied` | interpreter 未 patchelf（重跑 prepare.sh 第4步） |
| omg 全部 `MatMul don't support` | quant_param_2=True 或加了 output 段 → 改回 False/删 output |
| 评测/仿真 PPL=nan | fp16 溢出 → 必须 fp32（eval_ppl.py 默认已是） |
| 量化/评测 OOM | GPU 被他人占用（nvidia-smi 查）→ 等空闲或降 cutoff_len |
| 对话测试输出乱码 | base 模型不支持 chat template → 用 chat_test.py 的纯续写模式 |
| torch CUDA `no kernel image` | torch 版本不含你的 sm 架构（5090 需 2.8+cu128） |
| 想改 prefill 档位（16/128/三档） | **不支持**——档位在 dopt stage3 的 quant_params_file 里登记锁定为 {decode=1, prefill=64}，多档 omg 编译必失败（QUANTIZATION.md §三 prefill 档位实测） |
