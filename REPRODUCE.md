# kirin9020 平台 Qwen2.5-1.5B（base + instruct 双产线）全流程复现指引（操作手册）

> 本文只讲**怎么跑**。量化机制、为什么这样配、实验数据、踩坑记录见 **QUANTIZATION.md**。
> **两条产线、各自最优配方**（统一入口 `pipeline.sh`，详见下节）：
> - **base 交付版 = Qwen2.5-1.5B + g128 + zh维基校准**（续写演示）：中文 PPL 14.76（劣化 17.1%），SubGraph 1.27G；
> - **instruct 交付版 = Qwen2.5-1.5B-Instruct + g64 + zh对话校准**（ChatML，对话演示）：配方与评测见 QUANTIZATION.md §八。
> `05_device_files_base/` 与 `05_device_files_instruct/` 中是两套成品（文件名带 `_Base_` / `_Instruct_` 标记）。

## 统一产线入口（演示推荐用法）

```bash
bash pipeline.sh base              # base 全流程（量化已有产物会自动跳过）
bash pipeline.sh instruct          # instruct 全流程
bash pipeline.sh <base|instruct> [quant|eval|export|convert|pack|status]   # 单阶段
FORCE=1 bash pipeline.sh <p> quant # 强制重量化
```

- 同一套阶段脚本（02~05），差异全部收在 `profiles/{base,instruct}.env`：模型 / group_size / 校准语料 / 导出 yaml / 产物命名 / 交付目录。
- **演示流程**：先 `bash pipeline.sh base` 再 `bash pipeline.sh instruct`，产物分别落在 `05_device_files_base/`、`05_device_files_instruct/`；两个目录各自拷到连手机的 Windows 机，分别跑各自的 `push_to_device_next.bat`。


## 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA GPU；**sm_120(Blackwell/RTX 5090) 必须 torch 2.8+cu128**，sm_90 及以下可用 torch 2.4 |
| 系统 | Linux x86_64，glibc ≥ 2.35，`uv`、`unzip` |
| 依赖包 | DDK 工具包 + kirin9020 平台插件包（仅此两个，无需 CANN toolkit）。官方下载页：https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations ，本演示用的是 `DDK-tools-next-6.0.1.0.zip` + `kirin9020-plugin-next-6.0.1.0.zip`，下载后放本仓库旁的 `dependencies/`（`prepare.sh` 读 `../dependencies`，位置不同改脚本内 `DEPS` 即可） |
| 源码 | 上游样例库克隆到**本仓库同级目录**：`git clone https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp.git ../cannkit_samplecode_lm_engine_cpp`（阶段③ 的 npu_tuned_export 导出工程来自它；`06_demo_harmony_next_app/` 包含端侧 App 的 llm_demo.cpp 定制）。⚠️ 克隆后**必须套用 `06_demo_harmony_next_app/` 的定制 llm_demo.cpp**（覆盖或 `git am`，即未推上游的定制改动：UTF-8 半字符乱码修复 + instruct chat template 自动包装）；缺它则 demo App 不自动包装 chat template、中文可能半字符乱码 |

## 目录结构

```
qwen25_1b5_run/
├── REPRODUCE.md / QUANTIZATION.md
├── pipeline.sh                   # ★ 统一产线入口（base|instruct × 各阶段）
├── pack.sh                       # ★ 阶段⑤ 汇集脚本（仓库根，双产线共用，pipeline.sh 自动传参）
├── profiles/                     # ★ 产线差异声明（模型/group/语料/命名/交付目录）
│   ├── base.env  instruct.env
├── 01_prepare/                   # 阶段① 环境
│   ├── make_venv.sh  prepare.sh
│   ├── venv/ tools/ cuda_stub/   # (生成)
│   └── models/Qwen2.5-1.5B/  Qwen2.5-1.5B-Instruct/   # (生成/下载)
├── 02_quant/                     # 阶段② 量化（三段式，run.sh 吃 TESTCASE/CFG/MODEL 环境变量）
│   ├── base_config.yaml          #   base 配方（g128+zh维基校准；与 instruct_config.yaml 同 schema）
│   ├── instruct_config.yaml      #   instruct 配方（g64+zh对话校准）
│   ├── data_zh_wiki/             #   base 校准语料（build_corpus.py + dataset.json + test.txt）
│   ├── data_chat/                #   instruct 校准语料（build_corpus_chat.py + dataset.json + test.txt）
│   ├── run.sh  edit_dopt_config.py
│   ├── eval_ppl.py  chat_compare.py  continuation_compare.py  quant_sim.py
│   ├── reports/                  #   端云三方对比报告 report_{base,instruct}_3way_<口径>.md
│   ├── _autopatch/               #   transformers 4.51 兼容补丁（KD 用；PTQ 也无害）
│   ├── qwen25_1b5_base_9020/     #   (生成) base 交付量化工程（g128+zh维基）
│   └── qwen25_1b5_instruct_9020/ #   (生成) instruct 量化工程（g64+chat）
├── 03_onnx_export/               # 阶段③ export.sh + model_info_{base,instruct}.yaml
├── 04_omc_convert/               # 阶段④ convert.sh（OUTPUT_PREFIX 可换）
├── 05_device_files_base/         # 阶段⑤ base 交付目录（7 文件 + push 脚本）
├── 05_device_files_instruct/     # 阶段⑤ instruct 交付目录（7 文件 + push 脚本）
├── 06_demo_harmony_next_app/     # 阶段⑥ 端侧 App 与实机部署（llm_demo.cpp 定制 + 手册 + deploy/collect/diagnose 工具）
└── logs/                         # 各阶段日志
```

---

## 阶段 ① 准备（venv/tools/models 已就绪时可跳过）

**干什么**：搭一次性的量化环境，产物全落 `01_prepare/`。建独立 venv（torch 2.8 是 5090/sm_120 的硬要求）；解压 DDK 组装工具树（阶段②的 dopt 量化库 + 阶段④的 omg 编译器都在里面）；下载 base/instruct 两个源模型。

```bash
bash 01_prepare/make_venv.sh     # uv 建 venv，torch 2.8（输出应含 sm_120）
source 01_prepare/venv/bin/activate
bash 01_prepare/prepare.sh       # 解压 DDK+插件→tools/、下模型、patchelf 修 omg、cuda_stub
```
prepare.sh 四件事：组装 `tools/`（插件进 platform）→ 下载 Qwen2.5-1.5B 与 Qwen2.5-1.5B-Instruct 两个源模型（各 ~3GB，已存在跳过）→ **patchelf 把 omg interpreter 改系统 ld**（原始指向 /tmp 软链会 EACCES）→ 建 stub CUDA_HOME（过 deepspeed 的 CUDA_HOME 检查）。

## 阶段 ② 量化（三段式，GPU，~35 分钟）

**干什么**：用 dopt 把 HF 浮点权重转成 NPU int4 量化方案——三段接力，每段落盘、各有去向：

```
stage1  GPTQ 逐层优化 int4 权重（误差补偿，吃校准语料）
          └→ trained_quant_weight.pth    中间产物，只喂 stage2
stage2  EMA MinMax 定激活 int16 scale
          └→ trained.pth                 → 精度评测（PPL/生成对比）仿真 NPU 用
stage3  整理导出
          ├→ fake_quant_weight.pth       → 阶段③ 导 ONNX
          └→ quant_params_file           → 阶段④ omg 注入 s16s4 融合算子
```

三个交付文件的形态（为什么是这个样子）：
- `trained.pth` = 浮点原值 + 量化参数，量化在推理时**动态执行**——评测要能实时仿真 NPU 行为；
- `fake_quant_weight.pth` = 量化**已烘进数值**（每个值 = s_w·round(w/s_w)）——ONNX 图是纯 float、没有量化节点，量化效果必须预先烤进权重常量（omg 后续再量化幂等无损，§2.11）；
- `quant_params_file` = 量化参数表——激活量化参数（s_a）的唯一载体。

（`trained_quant_weight.pth` 留着的唯一价值：重跑 stage2 换 cutoff/样本时免 19min GPTQ，此后可清理，6G。）
机制与最优配方见 QUANTIZATION.md §二/§一。

校准语料重建（可跳过，若 `data_zh_wiki/dataset.json` / `data_chat/dataset.json` 已在）：
```bash
python 02_quant/data_zh_wiki/build_corpus.py #（zh 维基）
python 02_quant/data_chat/build_corpus_chat.py #（Belle 对话，ChatML 渲染）
```

```bash
source 01_prepare/venv/bin/activate
cd 02_quant
# instruct 三件套（与各脚本默认值一致，写出来是为了明确）
export TESTCASE=qwen25_1b5_instruct_9020 CFG=instruct_config.yaml \
       MODEL=$PWD/../01_prepare/models/Qwen2.5-1.5B-Instruct
bash run.sh stage1               # ①生成 dopt_config.json（198 节点全 float）
python edit_dopt_config.py $TESTCASE/dopt_config.json 64 --keep-lm-head-fp
                                 # ②改策略：embed=MinMax, 196 linear=eco(g64,in16), lm_head=float
bash run.sh stage1               # ③权重量化 GPTQ ~19min → weight quant done!!!
bash run.sh stage2               # ④激活校准 EMA ~10min → quant done !!!
bash run.sh stage3               # ⑤参数提取 ~4min → quant params file build done
```

**（可选）验证精度与生成质量**——推荐直接 `bash pipeline.sh <base|instruct> eval`（GPU 被占时加 `EVAL_DEVICE=cpu`）。手动（instruct 口径，PPL 对话留出集劣化 +9.0%，见 QUANTIZATION.md §八；base 的维基口径与续写对比见 §三 + `continuation_compare.py`）：

```bash
python eval_ppl.py ../01_prepare/models/Qwen2.5-1.5B-Instruct \
  qwen25_1b5_instruct_9020/dopt_config.json \
  qwen25_1b5_instruct_9020/train_output/trained.pth \
  --tag instruct --data data_chat/test.txt --n_samples 8192
python chat_compare.py ../01_prepare/models/Qwen2.5-1.5B-Instruct \
  qwen25_1b5_instruct_9020/dopt_config.json \
  qwen25_1b5_instruct_9020/train_output/trained.pth   # 对话对比（内置 chat template）
```

## 阶段 ③ ONNX 导出（GPU，~3 分钟）

**干什么**：加载阶段②的 `fake_quant_weight.pth`（伪量化浮点权重），导出 **"NPU 亲和"的 ONNX 图**（`model.onnx` + 外置权重 `model.pb`），同时落盘分离 embedding 的 int8 表和逐行 fp32 scale。

- **为什么必须经 ONNX**：omg 不认 PyTorch、只接受 ONNX 输入（convert.sh 里 `--framework 5`）——ONNX 是 torch 世界与华为编译器之间的**交换格式**（静态图 + 常量权重），omg 才能在此基础上做算子融合和 shape 编译。
- **"NPU 亲和"的含义**：不是随手 `torch.onnx.export` 导一份，而是 `npu_tuned_export` 工程按 NPU 编译器口味**定向改造过的图**，四件事：
  1. **GEMM→MatMul**——阶段④的 s16s4 融合 pattern 只认 MatMul，用 GEMM 就是全量 `don't support`，量化白做；
  2. **KV cache 显式外露**为输入/输出（FP16）——引擎逐 token 解码循环靠它喂缓存；
  3. **动态长度只留 {decode=1, prefill=64} 两档**——NPU 是静态编译，每档各编专用 kernel；
  4. **embedding 查表移出图**——token id 运行时才知道，数据相关的 Gather 无法静态编译 → CPU 查表、图直接收 `input_embed`（分离出的 int8 表 + 逐行 scale 就是本阶段产物之二）。
- **位宽注意项**：图里**没有**激活量化节点（激活量化是阶段④由 omg 从阶段②的 `quant_params_file` 注入）；Linear 权重 int4、激活 int16、embedding **int8** 逐行缩放（embedding 是查表、无输出误差可优化，int8 足够）。

```bash
bash 03_onnx_export/export.sh        # instruct（= 各脚本默认；pipeline.sh 自动传参）
# base：EXPORT_YAML=model_info_base.yaml OUT_SUBDIR=output_embedding_out_no_output_pos \
#       bash 03_onnx_export/export.sh
```

## 阶段 ④ OMC 转换（~2 分钟）

**干什么**：omg 把 ONNX 图编译成 kirin9020 专用的 `.omc` 离线模型，并按阶段②的 `quant_params_file` 把所有 MatMul 融合成 s16s4 融合算子。产物是 `.omc`（3.7M，图结构）+ `SubGraph_0.weight`（1.3G，外置权重）。

```bash
bash 04_omc_convert/convert.sh       # instruct（= 各脚本默认；pipeline.sh 自动传参）
# base：QUANT_DIR=$PWD/02_quant/qwen25_1b5_base_9020 \
#       ONNX_DIR=$PWD/03_onnx_export/output_embedding_out_no_output_pos \
#       OUTPUT_PREFIX=./Qwen25_1b5_Base_kirin9020 \
#       bash 04_omc_convert/convert.sh
```

## 阶段 ⑤ 端侧 7 文件

**干什么**：把手机上跑起来所需的全部 7 个文件汇集到交付目录（`05_device_files_instruct/`，base 版目录名带 `_base`、文件名带 `_Base_`）——这就是最终交付形态。7 件分别是（以 instruct 为例）：

1. **`Qwen25_1b5_Instruct_kirin9020.omc`**（3.7M）—— NPU 离线模型，图结构（阶段④产物）；
2. **`SubGraph_0.weight`**（1.3G）—— 外置权重：int4 权重 + scale 表（阶段④产物，两模型同名，换模型必须整组重推）；
3. **`model_instruct_64_2048.embedding_weights`**（223M）—— embedding int8 表（阶段③产物）；
4. **`model_instruct_64_2048.embedding_dequant_scale`**（594K）—— embedding 逐行 fp32 scale（阶段③产物）；
5. **`tokenizer.json`**（7M）—— 词表（取自源模型目录，base/instruct 词表相同）；
6. **`context.json`** —— 采样配置：top-k/top-p/温度/重复惩罚/seed（base 用抗复读参数，instruct 用官方推荐值）；
7. **`executor.json`** —— 引擎配置：模型/权重/embedding 的路径与 llm_config（层数/头数/档位等）。

整个目录拷到连手机的 Windows 机，推送与运行见阶段⑥。

```bash
bash pipeline.sh instruct pack     # instruct（= 裸跑 bash pack.sh 的默认值）
bash pipeline.sh base pack         # base 同理（pipeline 自动传 base 一整套参数）
```

## 阶段 ⑥ 端侧 App 与实机部署（App 装一次，模型随时换）

**干什么**：手机侧两件事——**装 App**（一次性）和**推模型**（每次换模型）。本仓库不含 App 完整工程：
工程来自上游样例库，本仓库只保留 `llm_demo.cpp` 的**定制改动**（`06_demo_harmony_next_app/`，未推上游：
UTF-8 半字符乱码修复、instruct chat template 自动包装、三方对比测试钩子——缺它则 App 不自动包装
chat template、中文可能半字符乱码）。装好后换模型只重跑⑤ pack + 本阶段推送，App 不用动。

### 装一次：构建安装 App

```bash
# 1. 克隆上游样例库到本仓库同级目录（阶段③ 的 npu_tuned_export 导出工程也来自它）
git clone https://gitcode.com/HarmonyOS_Samples/cannkit_samplecode_lm_engine_cpp.git ../cannkit_samplecode_lm_engine_cpp
# 2. 套用定制 llm_demo.cpp（覆盖即用，推荐；或 git am 06_demo_harmony_next_app/demo_next_llm_demo.patch 保留提交说明）
cp 06_demo_harmony_next_app/llm_demo.cpp \
   ../cannkit_samplecode_lm_engine_cpp/CANN_LLM/CANN_LLM_Engine_Demo/CANNLLMEngineDemoNext/entry/src/main/cpp/llm_demo.cpp
```

之后用 DevEco Studio 打开工程构建、安装到手机——首次装完 App 沙箱目录才会出现，推送的文件才有落点。
工程内 SDK 文件（7 个头文件 + `libhiai_llm_engine.so`，来自 DDK 包）放置与 hvigor/DevEco 兼容性
见 `06_demo_harmony_next_app/README.md`。

### 每次换模型：推送 7 文件（HarmonyOS NEXT，Kirin 9020 真机验证通过）

在连手机的 Windows 机上，交付目录里双击 `push_to_device_next.bat`（自动识别 Base/Instruct 文件名
并逐项核验），推送后完全退出并重启 App；或用 `06_demo_harmony_next_app/deploy_from_cloud.sh` 一键"云侧拉取→md5 校验→推送→
重启验证"。`SubGraph_0.weight` 两模型同名，换模型必须**整组 7 文件重推**。演示对话**直接输入问题即可**
——定制 App 检测到 prompt 未含 `<|im_start|>` 时自动套 ChatML，引擎侧已配 `<|im_end|>` 停止符。

**executor.json 路径三规则**（已配好，改动时勿破坏；三处分属不同 JSON 块、引擎各自独立解析）：
executor.json 里的"绝对路径"一律指**沙箱绝对路径**（`/data/storage/el2/base/haps/entry/files/...`，
App 进程视角；与 hdc 推送用的设备真实路径 `/data/app/el2/100/...` 是两套命名空间）：
- `model_path` / `weight_path`（autoregressive 块）：沙箱绝对路径，引擎直接加载；
- `tokenizer.path`（tokenizer 块）：同为沙箱绝对路径——写相对路径引擎找不到；
- `embedding_weights` / `embedding_dequant_scale`（llm_config 块）：**相对**文件名——引擎自动拼
  `weight_path` 前缀，写绝对路径会被拼成双重路径。

详细步骤与排障：`06_demo_harmony_next_app/NEXT_端侧测试手册.md`。