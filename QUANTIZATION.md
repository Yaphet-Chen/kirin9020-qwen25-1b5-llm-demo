# Qwen2.5-1.5B（base + instruct）/ kirin9020 int4 量化完全指南（实跑验证版）

> 本文回答一个问题：**到底要怎么量化，结果才是最优的。**
> 所有结论均来自本机（RTX 5090, torch 2.8.0+cu128）GPU 实跑 + wikitext2 PPL 评测 + 续写生成质量对比 + omg 实际转换验证。
> 浮点基线 PPL = 17.49；英文最优量化 **19.66（g64+wiki 校准，劣化 12.44%）**；**base 交付版 = g128 + 中文校准**（中文 PPL 14.76 / 劣化 17.1%，英文 24.42，取舍见 §三 2×2 矩阵）。
> **instruct 交付版（§八）= Qwen2.5-1.5B-Instruct + g64 + 中文对话校准：chat 域劣化仅 +9.0%，greedy 对话零退化**；统一产线入口 `pipeline.sh <base|instruct>`。
> 操作步骤见 REPRODUCE.md；本文是配方、机制（含算法数学原理）、实验与结论。机制部分证据分 [实证]（逆向 so 符号 / 产物文件实测）与 [推断]。
> **全文数学符号统一**（详见 §2.5 对照表）：`s_w` 权重 scale、`α` quant_alpha、`s_a` 激活 scale、`s_e` embedding scale、`γ` norm 增益。

---

## 一、配方速查（两个口径）

> 「最优」指**英文 PPL 口径**（g64 + wikitext2）。**当前交付版为 g128 + 中文校准**（面向中文端侧场景，体积更省），取舍依据见 §三 校准语料域效应 2×2。

| 配置项 | 最优值 | 在哪配置 | 说明 |
|---|---|---|---|
| weight bit | **4** (int4) | dopt_config.json `weight.bit` | 需求即 int4 |
| group_size | **64** | dopt_config.json `weight.group_size` | 比 128 好 1.0 PPL，体积 +45% |
| input bit | **16** | dopt_config.json `input.bit` | 激活 16bit |
| **lm_head** | **不量化 (float)** | dopt_config.json `lm_head.quant_strategy` | 输出层保 fp，好 0.9 PPL |
| embed | Quant_Embed_MinMax | dopt_config.json | 唯一可用 embedding 策略 |
| **校准样本** | **1024** | config.yaml `train/ptq/num_samples` | 收益递减点 |
| **cutoff_len** | **512** | config.yaml `cutoff_len` | **易被忽略的大头**：128→512 好 0.15 PPL |
| **校准语料** | **中文场景: zh维基 / 英文: wikitext2** | config.yaml `train_files` | 域效应：中文校准把中文劣化 25.9%→17.1%，英文塌到 +39.6%（§三 2×2） |
| quant_param_2 | **False** | config.yaml | 必须！True 则 omg 不识别 s16s4 |
| embedding_separate | True | config.yaml | 引擎要求 |
| kd.enable | **False** (PTQ) | config.yaml | 大样本下 KD 无增益（见 §2.9/§三） |
| linear 策略 | Quant_act_weight_eco | dopt_config.json | aigc 系列不兼容（见四.3） |

**产物**（05_device_files_base/ 现有成品 = **base 交付版 g128+中文校准**）：`Qwen25_1b5_Base_kirin9020.omc` 3.7M + `SubGraph_0.weight` 1.27G（omg 成功，s16s4 融合 3136 次；g64 版为 1.35G / 4704 次）。instruct 交付版见 §八。按 REPRODUCE.md / pipeline.sh 全流程可复现等价产物。

---

## 二、量化机制完全解析（算法原理 + 逆向 dopt 二进制 + 实跑验证）

### 2.1 量化基本式：统一符号与对称量化

全文量化函数统一记：

```
量化:   q = Q(x; s, b) = clamp( round(x/s), -2^(b-1), 2^(b-1)-1 )
反量化: x̂ = s · q
```

三类张量、四种关键参数（+norm），各用各的符号：

| 张量 | bit b | 格点 | scale 符号 | scale 粒度 |
|---|---|---|---|---|
| Linear 权重 w | 4 | [-8, 7] | `s_w` | 逐组：每行 1536/64=24 组 |
| Linear 输入激活 x | 16 | [-32768, 32767] | `s_a` | 逐张量（标量） |
| embedding 表 e | 8 | [-128, 127] | `s_e` | 逐行（每词表项 1 个） |
| （非量化）RMSNorm 增益 | — | — | `γ` | 逐通道向量，模型权重 |
| （可训练乘子）quant_alpha | — | — | `α` | 同 `s_w` 逐组，仅权重侧有 |

- **对称、无独立 zero-point**。[实证] 量化器 so 无任何 `zero_point/zp/asym` 符号；
  trained.pth 实测三类 quantizer 的 `offset` **全为 0**。初始化 s=amax/7 时名义用 [-7,7] 共 15 格（-8 弃用），换取 NPU 反量化
  免加法（s16s4 为对称格式）；但 GPTQ 误差补偿会改动未量化列，个别权重可
  越过 -amax 而用到 -8 格（§2.3 实测比值含 8 即此来源）。
- **scale 精度一律 fp32**。[实证] trained.pth 中 s 均为 `torch.float32`；
  embedding_dequant_scale = 151936×4B 与 594K 文件精确吻合。
- **权重 int4 / 激活 int16 分工**：省体积的 4bit 给"死"权重，高精度 16bit
  （65536 格点）给"活"激活，激活量化误差天然远小于权重。
- 各 scale 的计算公式、存储链路、可训练性见 **§2.5 总表**。

### 2.2 三段式流水线：每段做什么、产出什么

| 阶段 | 功能 | 输入 | 输出 | 耗时(s1024/c512) |
|---|---|---|---|---|
| **stage1 权重量化** | GPTQ 分块重建（或 KD 蒸馏训练）：优化每层 int4 权重的量化参数（scale/alpha） | 源模型 + dopt_config + config.yaml | `trained_quant_weight.pth` (6.0G) | ~19 min |
| **stage2 激活量化** | EMA MinMax 校准：跑 num_samples 个样本统计激活分布，定 `s_a` | stage1 产物 | `trained.pth` (6.0G) ← **PPL 仿真/对话测试用它** | ~10 min |
| **stage3 参数提取** | 导出部署格式 | stage2 产物 | `fake_quant_weight.pth`(6.7G, 给 ONNX 导出) + `quant_params_file`(g64≈1.1G / g128≈556M, 给 omg --compress_conf) + `embedding_weights/dequant_scale`(分离 embedding) | ~4 min |

三阶段分发在 `opt_main.py`：stage1→`weight_quant_pipline`、stage2→`inout_quant`、stage3→`build_params`。

### 2.3 stage1 算法：GPTQ（权重量化数学，产出 `s_w`）

[实证] `weight_quant_ptq/ptq_algo_heissen/` 与 GPTQ 原版开源库结构一一对应：
`WeightQuantHeissen.fasterquant`、`cholesky_inverse`、`percdamp`、`actorder`、
`llama_sequential/Catcher/add_batch`、`find_params`；quantizers 库封装为
`GPTQGrid`（add_batch/do_optimize/free）。

**目标函数（对输出，不对权重）**：

```
min_Ŵ  ‖(W − Ŵ)·X‖²_F      X ∈ R^(d_in × n) 为校准激活（n = 样本数 × cutoff_len）
```

朴素 RTN（逐权重四舍五入）单权重误差最小，但误差在 y=Wx 的百万项求和中
同向累积；GPTQ 改为逼近**层输出**。

**Hessian = 误差价格表**：

```
‖(W−Ŵ)X‖² = tr((W−Ŵ)·H·(W−Ŵ)ᵀ)，  H = X·Xᵀ（系数不影响最优解）
H_jk = Σ_t x_j·x_k ：输入 j,k 维在语料上的共现统计
H_jj = Σ_t x_j²   ：在 j 维犯单位误差的代价
```

tr 结构使问题**按 W 的行分解**：每行独立解 `min (w−ŵ)ᵀH(w−ŵ)`，约束 ŵ 落在
int4 格点（离散）→ 贪心。

**`s_w` 的确定（贪心循环之前）**：逐组 MinMax 初始化

```
s_w = amax_g / 7            amax_g = 该 64 权重组的 max|w|
```

[实证] trained.pth 中量化后权重的逐组 amax/s_w 比值 ∈ {6,7,8}、主档恰为 7
（负侧格点可到 -8，故有 8；个别组最大值被补偿修改后有 6），与 amax/7 公式
一致。so 含 `find_params`（GPTQ 原版 scale 搜索入口），是否有进一步 MSE
网格细化无法从数值区分，主档 7 倾向于纯 amax/7。纯 PTQ 下 `s_w` 就此固定，
后续 KD 只经 `α` 间接调整（见 §2.9）。

**逐列贪心 + 误差补偿**（账本类比：每记一笔就把差额转进未定稿账目）：

```
ŵ_i = Q(w_i; s_w, 4)                       ← 该坐标所属组的 s_w
e_i  = ŵ_i − w_i                           ← 不可逆，锁死
w_{i+1:} ← w_{i+1:} − (e_i / [H⁻¹]_{ii}) · [H⁻¹]_{i, i+1:}
```

`[H⁻¹]_{i,j}` = 考虑全部坐标相关性后，坐标 i 的单位误差应由坐标 j 吸收的比例。

**工程项**：`--block-size 128`（H⁻¹ 做 Cholesky 后按 128 列分块，块内量化、
块末统一补偿右侧，防数值漂移并提速）；`percdamp`（H 对角加阻尼防奇异）；
`actorder`（按 H_jj 重排列序）。

**group 与"逐列"是正交概念**：group_size=64 管**谁共享 `s_w`**（静态分组，
每行 24 组）；逐列扫描是**贪心推进次序**（动态流程）。量化第 i 列时该列各行
权重各自用自己组的 `s_w`，互不干扰。

### 2.4 stage2 算法：EMA MinMax（激活校准，产出 `s_a`）与校准数据

[实证] qema.so 类链：`EmaMovingInit.do_collect`（前向收集）→
`EmaMinMaxGrid.get_clip_range` → `clamp_ste`（STE 直通估计器）。

```
amax_t = α·本条 amax + (1−α)·历史 amax     ← 平滑单条离群样本（EMA 系数与 quant_alpha 无关）
s_a    = amax_ema / 32767                   ← int16 对称满格
```

[实证] layer0 q_proj 输入：amax=49.852028，49.852028/32767 = 0.001521410，
与实测 `s_a` 精确一致。`s_a` 是**逐张量标量**（shape=()），不是逐通道/逐
token；min 有记录但对称量化未使用（实测 min=-26.92, max=49.85）。
`s_a` **没有任何可训练参数挂载**（见 §2.5），EMA 定了就定了。

**校准数据**：[实证] train.so 符号链 `AutoTokenizer → encode_function →
concatenate_datasets`（全语料拼 token 流）→ 按 cutoff_len 切块 → 固定种子
（seed_all/data_seed）取前 N 块。**不是随机抽文章**，完全可复现。
s1024×c128 约占全语料 5%~6.5%（分母取 wikitext2 train 标准计数 2.0M 为 6.5%，取 dopt 拼接告警的 2.52M 为 5.2%），c512 约占 1/5~1/4。校准前向是 teacher-forcing 整段并行
计算：覆盖"有上下文条件下的激活"，但 ≠ 真实 decode（逐 token、吃自生成
内容）；int16 逐张量粗粒度对分布漂移不敏感。c512 优于 c128 的部分原因即
长块后段更接近 decode 时刻的条件分布。

**embedding 的 `s_e`**（同属校准产物）：`Quant_Embed_MinMax` 对 embedding 表
逐行 MinMax——

```
s_e = amax_row / 127        amax_row = 该词表行 1536 个权重的 max|e|
```

[实证] trained.pth 实测行 amax/s_e 比值 ∈ {127,128}、主档 127（负侧 -128
同理），与公式一致。embedding 是查表、无"层输出误差"可优化，故不用 GPTQ。

### 2.5 五个关键参数对照表（怎么算 / 存哪里 / 谁可训练）

**先看总表，后看细节。这张表是全文的参数索引：**

| 参数 | 符号 | 身份 | 计算方式（在哪阶段） | 形状 / 精度 | 存储链路（量化期 → 编译期 → 运行期） | KD 可训练 |
|---|---|---|---|---|---|---|
| 权重 scale | `s_w` | int4 权重量化 scale | `s_w = amax_g/7` 初始化（stage1，GPTQ 贪心前） | (36864,1)/层 = 1536行×24组，fp32 | trained.pth `weight_quantizer.s` → quant_params_file → SubGraph_0.weight 内 s16s4 算子参数 | 间接（经 `α`，[推断] 生效值 = s_w·α/4） |
| quant_alpha | `α` | 权重 scale 的**可学习乘子** | PTQ 下恒 4.0；KD 时 LSQ/STE 梯度更新（stage2 内训练回路） | 同 `s_w` 逐组 | trained.pth `weight_quantizer.quant_alpha` → 训练后 stage3 合成进 quant_params_file | **✓（仅此+γ）** |
| 激活 scale | `s_a` | int16 激活量化 scale | `s_a = amax_ema/32767`（stage2，EMA MinMax 校准） | 标量/张量，fp32 | trained.pth `input_quantizer.s` → quant_params_file → 融合算子参数（同 s_w 终点） | **✗**（input_quantizer 无 α 键） |
| embedding scale | `s_e` | int8 embedding 行 scale | `s_e = amax_row/127`（校准期 MinMax 收集） | (151936,1)，fp32 | trained.pth embed `weight_quantizer.s` → **独立运行期文件** embedding_dequant_scale（不进 omc） | ✗ |
| norm 增益 | `γ` | RMSNorm **模型权重**，非量化参数 | 不由量化计算，来自原模型；KD 可微调 | 57×1536，fp32→fp16 | 模型权重链路：trained.pth → fake_quant_weight.pth → model.pb → SubGraph_0.weight（fp16，永不量化） | **✓** |

**实测锚点**（trained.pth，2026-08-16 解剖）：
- `s_w`：量化后权重逐组 amax/s_w ∈ {6,7,8}，主档 7 → `amax_g/7`
- `s_e`：行 amax/s_e ∈ {127,128}，主档 127 → `amax_row/127`
- `s_a`：49.852028/32767 = 0.001521410，与实测精确一致
- `γ`：57 个 norm 键存在（28×2 + model.norm），**quant_op 挂载数 = 0**

**量化器状态解剖**（trained.pth 中每个 Linear 挂两套）：

```
model.layers.N.xxx.quant_op.weight_quantizer.{s_w, offset, bit, alpha, unsigned_quant}
model.layers.N.xxx.quant_op.input_quantizer.{s_a, offset, bit, min, max, unsigned_quant}
```

| 键 | weight_quantizer | input_quantizer |
|---|---|---|
| `s` | `s_w` (36864,1) 逐组 | `s_a` 标量逐张量 |
| `offset` | 全 0（对称实证） | 0 |
| `bit` | 4 | 16 |
| `min/max` | — | EMA 统计（仅 max 参与 `s_a`） |
| `quant_alpha` | `α` (36864,1)，PTQ 恒 4.0 | **不存在此键** |

embedding 的 weight_quantizer：`s_e` (151936,1) 逐行，offset 0，无 α
（MinMax 收集式，带 min/max 键）。

**scale 的三段旅程**（s_w 与 s_a 同文件、同终点；s_e 独走）：

```
s_w:  trained.pth · weight_quantizer.s ─┐
                                          ├→ stage3: quant_params_file (1.1G, g64)
s_a:  trained.pth · input_quantizer.s ──┘     │
                                               ▼
                        阶段④ omg --compress_conf 编译
                                               │
                                               ▼
       SubGraph_0.weight 内 s16s4/MatMuls16s4Gen 融合算子参数（运行期无独立 scale 文件）

s_e:  trained.pth · embed weight_quantizer.s → 独立 bin:
       embedding_weights(int8 表) + embedding_dequant_scale(逐行 fp32) → 运行期 7 文件之二

γ:    trained.pth 的 norm.weight → fake_quant_weight.pth → model.pb → omg → SubGraph_0.weight (fp16)
```

体积佐证：int4 权重 ≈655M + lm_head fp16 ≈467M ≈ 1.12G，SubGraph 1.3G 的
余量容纳 s_w/s_a 表。

### 2.6 config.yaml（量化工程配置）字段全解

```yaml
kd:                                # 蒸馏块（enable=False 时仅 loss/训练超参被忽略）
  enable: False                    # False=纯PTQ(推荐)；True=KD蒸馏(需teacher_config_path!见2.9)
  loss: mse                        # 蒸馏 loss
  num_epochs / learning_rate / lr_scheduler_type / warmup_steps:  # KD 训练超参
  trainable_keys: [quant_alpha, norm]  # KD 可训练范围 = α + γ（精确边界见2.5/2.9：s_a 不可训）
  no_split_module_classes: [Qwen2DecoderLayer]
dataset:
  train_files: wikitext2           # 校准数据（英文版；交付版现指向 02_quant/data_zh_wiki/dataset.json，
                                    #   自定义 json 的记录数必须 ≥ num_samples——stage2 按记录索引样本）
  train_samples: 1024              # 训练样本数（KD 用；PTQ 下与 ptq_samples 等值即可）
  ptq_samples: 1024                # PTQ 重建样本数（stage1，喂 GPTQ 的 Hessian）
extra_training_config: {fp16: False}
cutoff_len: 512                    # ★ 样本序列长度（校准统计的上下文长度，128→512 +0.15 PPL）
num_samples: 1024                  # ★ 激活校准样本数（stage2，定 s_a）
quant_param_2: False               # ★★★ 必须False！True→quant_params_file 格式 omg 无法融合 s16s4
embedding_separate: True           # 必须 True（LLM Engine 要求分离 embedding）
lm_head_size:                      # lm_head 词表补齐（空=不补）
```

配置项 ↔ 参数/算法对应：`--block-size`→GPTQ Cholesky 分块；`weight.bit/
group_size`→`s_w` 的位宽与分组；`input.bit`→`s_a` 位宽；`num_samples`→EMA
校准条数；`cutoff_len`→校准序列长（GPTQ 与 EMA 共用）；`quant_param_2`→非
算法，仅 quant_params_file 格式开关；`trainable_keys`→KD 白名单（匹配到
`α` 与 `γ`，见 §2.5/2.9）。

### 2.7 dopt_config.json（逐层量化策略）结构

stage1 第一次运行自动生成 198 个节点（1 embed + 196 linear + 1 lm_head），全部默认 `float`，**必须手动编辑策略**后再跑第二次 stage1（用 `edit_dopt_config.py`）。

```json
{
  "layer_strategy": {
    "model.embed_tokens": {"quant_strategy": "Quant_Embed_MinMax"},
    "model.layers.0.self_attn.q_proj": {
      "quant_strategy": "Quant_act_weight_eco",
      "weight": {"bit": 4, "group_size": 64},
      "input":  {"bit": 16}
    },
    "lm_head": {"quant_strategy": "float"}
  }
}
```

**策略全集**（逆向 `quantizer/strategy.so` 提取，共 7 种）：
| 策略 | 适用 | 状态 |
|---|---|---|
| `float` | 不量化 | ✅ lm_head 保精度用 |
| `Quant_Embed_MinMax` | embedding | ✅ 唯一可用 |
| `Quant_act_weight_eco` | decoder linear | ✅ **主力策略** |
| `Quant_lm_head` | lm_head | ✅ 可用但掉精度 0.9 PPL |
| `Quant_aigc_ptq` / `Quant_aigc_qat` | — | ❌ **不兼容**（stage2 scale 形状崩溃，见四.3） |
| `Quant_Dummy` | 测试用 | — |

**per-layer 可调键**：`weight{bit, group_size}`、`input{bit}`、`output{bit, per_channel, input_algo}`。
⚠️ **output 块不要加**（指南建议 9020 加以提速，实测加了 omg 反而无法识别 s16s4）。

### 2.8 CLI 参数与 JSON 的优先级

`opt_main.py` 把 CLI 参数设为环境变量做**全局默认**：`--group-size→custom_group_size`、`--block-size→custom_block_size`、`--w-bits→custom_w_bits`、`--act-bits→custom_act_bits`。**dopt_config.json 的显式 per-layer 配置优先生效**。我们 JSON 总是写显式值 → JSON 即真相。`--block-size`(PTQ 重建块) 保持默认 128。

### 2.9 KD 蒸馏：可训练边界（精确版）与数学

**trainable_keys 的实际匹配结果**（哪些参数真能被训练）：

| 目标 | 可训练 | 参数与去向 |
|---|---|---|
| `s_w`（权重 scale） | ✓ 间接 | 经 `α` 调整：`α` 逐组可训练，[推断] 生效 scale = s_w·α/4（恒等初始化乘子）；训练后的值经 stage3 合成写回 quant_params_file |
| `s_a`（激活 scale） | ✗ | input_quantizer **无 α 键**（§2.5 实测），EMA 校准值锁死 |
| `γ`（norm 增益） | ✓ | 模型权重直接微调，KD 后走权重链路到 SubGraph_0.weight (fp16) |

[推断] `α` 均匀初始化 4.0 而 `s_w` 逐组各异，生效 scale 疑为 `s_w·α/4`；
合成公式在闭源 so 内未验证。

**执行位置：嵌在 stage1（weight_quant_pipline 内），不是独立阶段**。
[实证] KD 缺 AdamW 时的报错栈：`opt_main stage1 → weight_quant_pipline(522)
→ train(442) → DoptTrainerCustomization.train(140) → training_loop_block(152)`；
且 KD 运行日志中 Epoch 循环出现在 `weight quant done!!!` 之前（同一次 stage1 调用内）：

```
stage1 GPTQ（s_w 初始化）→ KD 训练回路（training_loop_block，只动 α 和 γ）→ weight quant done
  → stage2 EMA 校准（s_a 固定）
  → stage3 导出训练后的值
```

teacher = 原始 FP 模型（teacher_config_path 指向同一模型目录，自蒸馏；
**缺此字段 KD 静默跳过**，见四.4）。

**梯度**（对生效 scale 记 s，链到 α 时再乘 ∂s/∂α）：

```
前向: ŵ = s · Q(w/s; 4)                   round 用 STE（反向当恒等）
损失: L = Σ_块 ‖f_量化(x; s, γ) − f_FP(x).detach()‖²     （loss: mse）

scale 梯度（LSQ，截断范围内）:  ∂ŵ/∂s ≈ Q(w/s; 4) − w/s
  → 被 round 拉得越狠的权重，调 s 杠杆越大（梯度自动聚焦伤最重处）

γ 梯度（干净真梯度，无需 STE）: y_i = γ_i·x_i/RMS(x)
  ∂L/∂γ_i = Σ (∂L/∂y_i)·(x_i/RMS(x))
```

`γ` 有效的机理：量化误差含**系统性幅度偏差**（int4 后层输出范数偏移、
int16 的 `s_a` 范围略紧截断缩小下游输入），且逐通道不均匀；1536 维逐通道
乘子 γ 恰是一阶修正器，且每个 block 出口必过 RMSNorm，站在"本块误差总闸
门"。文献称 scale/bias correction（AdaRound bias correction、LSQ+）。

**KD 天花板的完整解释**：可训练的只有 `α`（→s_w）与 `γ` 两类**乘性修正**，
`s_a` 锁死、round 落点错误无法修复（需 AdaRound 式 rounding 学习，本工具链
未开放）→ 大样本(1024)下 PTQ 校准已近最优，KD 无增益（19.87 vs 19.81）；
小样本(256)校准粗糙时微调有价值（20.20 vs 20.38）。实验全记录见 §三。

### 2.10 精度评测方法（eval_ppl.py）
- 数据：wikitext2 test，8192 token（截断在 max_position 内）
- 仿真：`dopt.dopt_lm.do_opt.optimize_model`（注意：指南写的 opt_main 是错的）插入量化算子 + 加载 stage2 的 `trained.pth`（quant_op 前向实时用 s_w/s_a 伪量化，不经过 fake_quant_weight）
- **必须 fp32 推理**：fp16 在量化算子上长序列溢出 nan
- 指标：PPL = exp(Σnll/N)，fp32 log_softmax

### 2.11 下游消费与运行期数据流

**编译期链路**：

```
stage3.fake_quant_weight.pth ─→ 03_onnx_export（替换权重导出 NPU 亲和 ONNX，产 model.onnx+model.pb+分离embedding）
stage3.quant_params_file ────→ 04_omc_convert（omg --compress_conf 把 MatMul 融合成 MatMuls16s4Gen）
                                → Qwen25_1b5_{Base,Instruct}_kirin9020.omc + SubGraph_0.weight
                                → 05_device_files_{base,instruct}/ 各 7 文件
```

**fake_quant_weight.pth 的职责边界**：只伪量化**权重**（值 = s_w·q 的 float，
γ 等普通权重原样随行）；导出的 ONNX 是**纯 float 图，无任何激活量化节点**
——`s_a` 的量化由 omg 编译期从 quant_params_file 注入融合，不经过 ONNX。
双重量化无损：q=round(w/s_w) → 存 s_w·q → omg 再 round((s_w·q)/s_w)=q，幂等。

**embedding 运行期分工：查表在 CPU，反量化在 NPU**：

```
CPU/引擎:  token id → 查 int8 行(1.5KB/token) + 取该行 s_e
              → 两个张量作为 input_embed / embed_scales 喂给 NPU 图
NPU 图内:  inputs_embeds = input_embed × embed_scales   ← 图内第一个算子（s_e 反量化）
```

[实证] export_model_wrapper.py 中 `inputs_embeds = input_ids * embed_scale`。
数据相关 Gather（token id 运行时才知道）不适合 NPU 静态编译，CPU 行拷贝几乎
免费；规则张量乘法留在 NPU。两个 embedding 文件 = 引擎每次亲手交给 NPU 的
两个张量。

**KV cache：无量化，全程 FP16**：
- [实证] convert.sh：`past_key_in*/past_value_in*` 与输出 `past_key*/past_value*`
  的 input_type/output_type **均为 FP16**；layer_strategy 只有 embed/196
  Linear/lm_head，KV 不在量化节点清单（无任何 scale/α 挂载）。
- 体积账：28 层 × 2(k,v) × 2048 × 2 头 × 128 维 × 2B ≈ **59MB**，fp16 可承受。
- 原因：K/V 逐 token 动态生成无法离线优化；attention softmax 对 K/V 精度
  敏感；59MB 无 int8 刚需（KV 量化是服务端大 batch 长上下文场景的需求）。

---

## 三、完整实验矩阵（wikitext2 PPL，8160 token，fp32 仿真）

| # | 配置 | PPL | ΔPPL | 劣化% | 结论 |
|---|---|---|---|---|---|
| — | FP 浮点 | 17.49 | — | — | 基准（sha256 校验官方一致） |
| 1 | g128/s16/lm量化/c128（指南默认） | 22.31 | +4.82 | 27.5% | 起点，精度差 |
| 2 | g64/s256 | 21.29 | +3.80 | 21.7% | group↓ 有效 |
| 3 | g64/s256 + KD(3ep) | 20.20 | +2.71 | 15.5% | KD 小样本有效 |
| 4 | g64/s256/lm保fp | 20.38 | +2.89 | 16.6% | lm_head 保 fp 大幅有效 |
| 5 | g64/s1024/lm保fp/c128 | 19.81 | +2.32 | 13.2% | 大样本有效 |
| 6 | g64/s1024/lm保fp + KD(5ep) | 19.87 | +2.38 | 13.6% | KD 大样本无增益 |
| 7 | 同上 KD(2ep) | 20.02 | +2.53 | 14.5% | epoch 少更差 |
| 8 | KD hi-lr(5e-4)/s512/8ep | 19.95 | +2.46 | 14.1% | 高 lr 无突破 |
| **9** | **g64/s1024/lm保fp/c512** | **19.66** | **+2.18** | **12.44%** | **✅ 当前最优** |
| 10 | 同上 c1024 | OOM | — | — | GPU 被占未测成，空闲时可重试 |
| 11 | g128/s1024/lmfp/c512 + **中文校准**(zh维基) | 中文14.76(Δ+2.15,+17.1%) | +2.15 | 17.1% | 中文域显著改善；英文 PPL 24.42(+39.6%) 大幅回退（域转移代价） |
| （对照） | g64/s1024/lmfp/c512 + wiki英文校准 | 中文15.87(+25.9%) | +3.26 | 25.9% | 同一中文留出集上对比：中文校准把中文劣化从 25.9%→17.1% |

**各维度边际收益**（从指南默认到最优的 2.65 PPL 改善来源）：
- group 128→64：**−1.02**（最大单项，代价 volume +45%）
- lm_head 保 fp：**−0.91**（最大单项）
- 样本 16→1024：**−0.57**
- cutoff_len 128→512：**−0.15**

### 生成质量（续写）对比（greedy + 端侧采样）

评测：纯续写（**base 模型不能用 chat template 对话**，见坑6）；greedy 严格对比 + 端侧采样（context.json 同款 top-k20/temp0.6/rep1.2，最接近部署行为）。原始输出见 logs/final_chat_compare.log（greedy）与 logs/final_chat_sampling.log（采样）。

**Greedy（6 prompt）**：
| 维度 | FP (bf16) | 最优量化 (19.66档) | 旧基线 (22.31档) |
|---|---|---|---|
| 英文续写 | 流畅无循环 | 连贯切题，轻度句式重复 | 从头重复、内容扁平 |
| 中文续写 | 流畅 | 部分退化（考试题复读/句式循环） | 整句循环 |
| 与FP前缀逐字一致 token 数 | — | 平均 3.0（greedy 蝴蝶效应所致，非质量指标） | — |

**端侧采样两轮测试**：
- 近似轮（temp0.6/k20/rep1.2，未实现 top_p、未定 seed）：最优量化 3/3 连贯——中文循环消失（"秋天"变流畅健康建议）、选项不再同文复读、英文事实正确（c=299792458）。FP 采样下中文同样漂移考试题格式 → 漂移主要是 base 模型固有特性。
- **严格轮（逐项复刻 context.json：seed99 逐 prompt 重置 / top-k16 / top-p0.95 核采样 / temp0.6 / rep1.2，FP 与 QUANT 消耗完全相同的随机数）**（logs/strict_device_sampling.log）：
  - 英文 2/2 与 FP 前 20 字符完全一致后岔开，岔开后质量同为高质量（相对论→引力弯曲时空/因果律论述；Python→解释型/范式/PyPI 介绍）
  - 中文 2/2 立即岔开但输出全部成型良好（"秋天"→格式工整的三点幸福感建议；"长城"→仍是选择题但四选项互异）
  - **零退化**（无循环/无复读）；量化"长城"选项含史实混淆（把长城说成运河），单点质量瑕疵
  - 结论：部署配置下量化-浮点差距收敛为"同分布采样、路径不同"，无可用性退化

**结论**：量化退化真实存在（PPL 12.4%：英文生成几乎无感、中文风格漂移更频繁），端侧采样参数下基本恢复可用；PPL 排序=生成质量排序，评测方法可信。


### 校准语料的域效应（中英 2×2 实测）

| 校准 \ 测试 | 英文 wikitext2 | 中文 zh维基留出集 |
|---|---|---|
| wiki 英文校准 (g64) | 19.66（+12.4%） | 15.87（+25.9%） |
| **zh 维基中文校准 (g128)** | 24.42（**+39.6%**） | **14.76（+17.1%）** |

- 校准语料决定 GPTQ 的 Hessian 与 EMA 的 `s_a`——**激活统计匹配什么分布，什么分布就受保护**。中文校准把中文劣化 25.9%→17.1%（且这是在 g64→g128 本身掉精度的情况下取得的，纯域效应更大），但英文塌到 +39.6%。
- 端侧部署面向中文场景 → 推荐 g128+中文校准版（产物 SubGraph_0.weight 1.27G，比 g64 版 1.35G 再省 82MB）；若需中英双语均衡，可混合中英语料校准（未实验，遗留项）。
- 中文数据格式坑：dopt 的 stage1 拼接全文用，但 **stage2 按数据集"行"索引样本**——自定义 dataset.json 必须切成 ≥num_samples 行（本版切成 2408 行×~800字，行宽贴近 cutoff_len）。语料构建脚本见 data_zh_wiki/build_corpus.py（wikimedia/wikipedia 20231101.zh 流式拉取，217篇/200万字，留出 28 篇做测试）。



---

## 四、关键坑与结论（踩过的雷，按重要性）

1. **`quant_param_2` 必须 False**。True 产出的 quant_params_file（690M）omg 无法融合 s16s4 → 所有 MatMul don't support。False 的文件：g128+lmfp≈556M（交付版）/ g64+lmfp≈1.1G / g128+lm量化≈655M（历史值）。验证手段：omg 日志 s16s4 计数（成功时应数千次：g128 版 3136 / g64 版 4704）。
2. **dopt_config 不加 output 段**。指南称 9020 加 output 提速，实测加了 omg 反而不识别。
3. **aigc 系列策略（Quant_aigc_ptq/qat）与三段式管线不兼容**：stage1 能跑完，stage2 加载 checkpoint 时 scale 形状不匹配崩溃（`[215040,1] vs [1536,1]`）。两种均实测崩。linear 只能用 Quant_act_weight_eco。
4. **KD 蒸馏的正确打开方式**：config.yaml 的 kd 块里必须显式加
   ```yaml
   teacher:
     teacher_config_path: <模型路径>   # 缺此字段 dopt 静默跳过 KD，不报错！
   ```
   另需 `_autopatch/sitecustomize.py`（修 transformers 4.51 的 evaluation_strategy/AdamW 兼容，AdamW 需物理补进 optimization.py）。KD 结论：小样本(256)时 +0.18 PPL 有效；大样本(1024)时无增益（可训练边界与数学解释见 §2.9）。
5. **fp16 推理在本机（torch 2.8/sm_120）logits=nan 彻底不可用**。PPL 评测/仿真对话必须 fp32（浮点模型本体可用 bf16）。
6. **base 模型不能用 chat template 测对话**。Qwen/Qwen2.5-1.5B（指南链接）是 base 版，套 im_start 格式输出乱码重复是正常行为。排查链：sha256 官方一致 → CPU 同崩（排除 GPU kernel）→ tf4.49 同崩（排除版本）→ 纯续写全对 → 定性为用法错误。
7. omg 转换**不需要 CANN toolkit**（DDK 自带算子库；te_fusion 缺失只是良性告警）；omg 二进制需 patchelf 改 interpreter（prepare.sh 已做）。
8. PPL 评测必须 fp32：fp16 在量化算子上长序列溢出 nan。

---

## 五、端侧部署配置：prefill 档位与 sampler 数学

### 5.1 prefill 档位：工具链锁定 {decode=1, prefill=64}（实测）

executor.json 注释称 prefill_len 可配 16/32/64/128——那是**引擎侧**能力；实际档位在 dopt stage3 生成的
quant_params_file 里**按 shape 登记**，omg 编译每个档位时查表：

| 编译档位 | 结果 |
|---|---|
| 1+64（现行） | ✅ s16s4=3136 |
| 1+128 / 1+16 / 1+64+128 三档 | ❌ `can't find key [shape]`×784，s16s4 掉到 1568（仅 decode 档融合），CPUCL inferShape 崩 |
| ONNX 以 seq_len=128 重导出后再编 1+128 | ❌ 同败 → 锁定点在 quant_params_file（dopt 侧），非导出侧 |

dopt 无 CLI/配置暴露 prefill 形状 → **运行时 executor.prefill_len 必须保持 64**；换档需华为工具链支持。

### 5.2 sampler 数学：从 logits 到 token 的分布变换链

模型每步输出 logits z∈R^V（V≈151936），softmax `p_i = e^{z_i}/Σe^{z_j}`（Boltzmann 分布，logits 即能量）。
context.json 的 sampler 块是一条作用在该分布上的变换链：

```
z ──rep惩罚──> ──温度──> ──top-k──> ──top-p──> 多项采样
```

| 参数(值) | 数学操作 | 作用机制 |
|---|---|---|
| temperature (0.6) | `p_i(T) ∝ e^{z_i/T}` | 缩放能量差：Δz=2 时 T=1 概率比 7:1，T=0.6 变 28:1；T→0 退化为 argmax。保序、熵随 T 单调。T<1 压尾部噪声，副作用逼近 greedy（复读风险↑，故须配惩罚） |
| top-k (16) | 前k大概率集内重归一 | **固定支撑集截断**，零化尾部质量——尾部恰是模型最不可靠且量化相对误差最大区，故对 int4 尤其有效 |
| top-p (0.95) | 累计质量≥p 的最小集内重归一 | **自适应支撑集**：高置信步核心仅1-3个token，低置信步几十个；与k互补（k封顶大小，p封顶质量） |
| repetition_penalty (1.2) | 已生成 token 的 logit 除以 γ | CTRL 乘性先验，概率约乘 `e^{-z(1-1/γ)/T}`；打断自增强回路——正是 base+int4 在 greedy 下的主失效模式（考试题复读） |
| seed (99) | 固定随机数流 | 同 seed+同分布 → 逐 token 可复现；严格对照借此让 FP/QUANT 消耗相同随机数 |

顺序：penalty → T → k → p → 采样（HF 惯例；端侧引擎顺序未公开但等价）。

### 5.3 为什么解码参数对量化观感影响巨大：误差阻尼的形式化

- **greedy 是 0/1 刀锋**：argmax 在近平局处对小扰动不连续，一次翻转即轨迹永久分岔（实测 FP/量化
  greedy 前缀逐字一致仅 ~3 token）；
- **采样是连续阻尼**：同随机数下两侧同 token 概率 ≥ 1−TV(p,q)（总变差距离）——一致性随分布距离
  **连续**变化。配合 top-k 把支撑集压到两分布重叠度最高的头部，即得实测结论：同一份 int4 权重
  greedy 中文退化、端侧采样参数下与浮点只剩"采样路径不同"（§三生成质量对比）。

解码参数不改变模型质量（PPL 不变），只改变 logit 级误差如何放大为序列级差异。

### 5.4 端云一致性：真机 NPU vs 云侧仿真（三方对比，2026-08-17 实测）

对比三方：云侧 FP(bf16)、云侧量化仿真(fp32, continuation_compare.py)、真机 NPU int8 推理
（真机侧由 `05_device_files_base/collect_phone_replies.sh` 自动收集，统一 n=600 口径）。
完整数据见 `02_quant/device_compare_report_3way_{sampling,greedy}.md`。要点：

- **贪心不是逐 token 一致**：5 条 prompt 中 4 条在 0~30 字节内分叉（英文条 ~100 字节后分叉），
  与 repetition_penalty 无关（rep=1.0 对照组前缀基本相同）。
- **embedding 差异已排除**：`continuation_compare.py --emb` 让云侧仿真加载端侧同款 int8 embedding
  （两时机替换 + 置零探针验证生效），输出与未替换**逐字节相同**——端侧 int8 embedding 反量化
  与原始权重最大绝对误差仅 0.0013，不足以翻转任何一步 argmax。
- 结论：端云分叉源于**计算路径本身**（云=权重假量化+fp32 矩阵乘；端=s16s4 融合算子真 int 计算），
  属同分布数值实现差，非缺陷；采样口径下三方输出同档可用。

## 六、精度-体积-耗时权衡

| 方案 | PPL（测试口径） | SubGraph_0.weight | 量化总耗时 |
|---|---|---|---|
| **base 交付版** g128+zh校准+lmfp+c512 | 中文 **14.76**（+17.1%）/ 英文 24.42（+39.6%） | **1.27G** | ~33 min |
| **instruct 交付版** g64+zh对话校准+lmfp+c512 | chat 域 **13.76**（+9.0%）/ FP 12.62 | **1.3G** | ~41 min |
| 英文最优 g64+wiki校准+lmfp+c512 | 英文 **19.66**（+12.4%）/ 中文 15.87（+25.9%） | 1.35G | ~33 min |
| 基线 g128+wiki+lm量化+c128 | 英文 22.31（+27.5%） | 894M | ~15 min |
| 快速验证 s16/c128 | ~21.3（估） | 同基线 | ~3 min |

## 七、遗留项
- **c1024**：GPU 被其他用户进程占用（14.5G）连续 OOM 未测成。空闲时补测：`CUTOFF=1024 bash 02_quant/run_experiment.sh cl1024 64 1024 false 1 --keep-lm-head-fp`，或有再 −0.1~0.2 空间。
- ~~**端侧对话**~~：**已完成**——见 §八 instruct 产线（Qwen2.5-1.5B-Instruct，g64+对话域校准，统一入口 `pipeline.sh`）。
- **手机 NPU 实测**：base omc 已上机验证；instruct omc 已产出（s16s4=4704），待上机。
- **闭源不可见项**：Quant_act_weight_eco 的 "eco" 具体接线、svd_quant（SvdQuantizer，带 LUT）启用场景、`α` 与 `s_w` 的合成公式、`find_params` 是否做 MSE scale 搜索——均在闭源 so 内。

---

## 八、Instruct 产线：Qwen2.5-1.5B-Instruct / int4 g64 / 对话域校准（2026-08-17）

> 目标：端侧**对话** demo（base 版只会续写，§四坑6）。**量化方案与校准数据都按 instruct 重新匹配**，
> 公共结论复用 base 实验矩阵（§一/§三），差异只有三处：模型、group_size、校准语料域。
> 统一入口：`bash pipeline.sh instruct`（base 版为 `bash pipeline.sh base`，两产线共用全部阶段脚本，
> 差异声明在 `profiles/*.env`——演示时可先跑 base 再跑 instruct，互不覆盖）。

### 8.1 配方（与 base 的差异加粗）

| 配置项 | instruct 值 | base 交付版值 | 说明 |
|---|---|---|---|
| 模型 | Qwen2.5-1.5B-**Instruct** | Qwen2.5-1.5B | 同架构同 vocab(151936)，tokenizer.json 三段实测完全一致，设备侧可互换 |
| group_size | **64** | 128 | 精度优先（用户指定）：~1 PPL 换体积 +82M（SubGraph 1.27→1.3G） |
| 校准语料 | **zh 对话（ChatML 渲染）** | zh 维基 | **校准分布=部署分布**（见 8.2），域效应同 §三 2×2 的逻辑 |
| weight/input bit、lm_head、样本数、cutoff、PTQ、quant_param_2 | 4/16、fp、1024、512、KD off、False | 同 | base 实验矩阵的最优公共项，原样继承 |

### 8.2 校准语料构建（`02_quant/data_chat/build_corpus_chat.py`，可复现）

- **来源**（HF 流式）：`BelleGroup/multiturn_chat_0.8M`（解析 instruction 内嵌的 Human:/Assistant: 标记为真多轮 turns，保留 ≥2 问）+ `BelleGroup/train_0.5M_CN`（单轮指令），按对话数 2:1 交错。
- **渲染**：用 **Instruct 模型自带的 chat template**（ChatML，带默认 system "You are Qwen..."）把每条对话渲染成文本——与端侧 App 侧 `apply_chat_template` 后喂引擎的 token 流一致。这正是"匹配对应的校准数据集"的含义：GPTQ 的 Hessian 与 EMA 的 `s_a` 统计什么分布，什么分布就受保护（§三域效应）。
- **格式约束沿用 data_zh_wiki 的坑**：`dataset.json` 为 `[{"text":...}]` 行式 json，**行数 ≥ num_samples**（实测 2401 行）；每行打包若干条**完整**对话至 ~480 token（不跨行拆对话）；stage1 拼接全文（总量 856k token > GPTQ 需要的 524k），stage2 按行索引前 1024 行。
- **留出集**：120 条对话渲染为 `test.txt`（70k 字符），`eval_ppl.py --data` 用，不参与校准。
- 实测构成：多轮 2705 / 单轮 1005 条对话；行宽 min/avg/max = 166/360/640 token（>512 的行被 cutoff 截断，尾部损失可忽略）。

### 8.3 精度：chat 域 PPL（fp32 仿真，8160 token）

| 口径 | FP | QUANT | 劣化 |
|---|---|---|---|
| instruct g64 + zh对话校准（本产线） | 12.62 | 13.76 | **+9.02%** |
| （参照）base g128 + zh维基校准，zh维基测试 | — | — | +17.1% |
| （参照）base g64 + wiki校准，wiki英文测试 | 17.49 | 19.66 | +12.4% |

注意跨模型/跨测试集的 PPL 绝对值不可直接比较（instruct 在对话域天然 PPL 更低），但**相对劣化**可比：
instruct 产线拿到全部交付版中最低的 +9.0%——g64（−1 PPL 级收益）与对话域校准（域对齐收益）叠加的结果。

### 8.4 端侧配置差异（05_device_files_instruct）

- `context.json` 采样参数换为 **Qwen2.5-Instruct 官方推荐值**（generation_config.json）：T=0.7 / top-p=0.8 / top-k=20 / repetition_penalty=1.1（base 版的 T0.6/k16/p0.95/rep1.2 是为 base 抗复读特调，instruct 不需要那么激进）；seed=99 保留（演示可复现）。
- `executor.json`：eos_token_id=**151645**（`<|im_end|>`，base 为 151643）；model_path/embedding 名带 `_Instruct_` 标记。
- 生成停止：引擎 stop_sequence 已含 `<|im_end|>`，与 eos 双保险。
- **App 侧发 prompt 必须套 chat template**（引擎不模板化）：`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`。

### 8.5 产物与验证清单（实测，对照 REPRODUCE.md §验证清单）

| 项 | 实测 | g64 预期 |
|---|---|---|
| quant_params_file | 1.1G | ✅（556M 为 g128 口径，690M 为 quant_param_2 配错） |
| fake_quant_weight.pth | 6.7G | ✅ |
| omg s16s4 字符串计数 | **4704** / don't support=0 | ✅（g64 口径；g128=3136。注意按字符串出现次数统计，按行数统计会得到 3136/其他值） |
| SubGraph_0.weight | **1303977472 字节（1.3G）** | ✅ 传输后按此字节数核对（对照 base g128=1272421888） |
| omc | 3.7M | ✅ |
| 交付目录 | 05_device_files_instruct/（7 文件+脚本） | 命名：`Qwen25_1b5_Instruct_kirin9020.omc`、`model_instruct_64_2048.embedding_*` |

### 8.6 生成质量（chat_compare.py，greedy，chat template，128 token；完整输出 logs/chat_instruct.log）

| 问题 | FP | QUANT(g64) | 评注 |
|---|---|---|---|
| 自我介绍 | 流畅得体 | 流畅得体（措辞不同） | 同档 |
| 长城历史（三句话） | 准确（秦汉明） | 准确（"多个朝代"表述） | **base 版量化曾把长城说成运河；instruct 版量化无此史实错误** |
| 25×37 逐步算 | 175+750=**925** 全对 | 分步正确（175；25×30 拆解），128 token 截断未到最终和（FP 亦被截断） | 同档 |
| Python 质数函数 | 正确实现 | 正确实现且带完整 docstring | 同档 |

- 全部 4 问：**零循环、零复读、零乱码**，与 FP 同质量档——instruct 模型 + g64 + 对话域校准下，
  greedy（0/1 刀锋，§5.3 对量化误差最不利的解码方式）已无可用性退化；端侧还叠加采样阻尼，观感只会更好。
- 对比 base 版结论（§三生成质量）：base 量化在 greedy 下中文有复读/史实混淆，需靠端侧采样参数恢复；
  instruct 版天然对 chat 格式对齐，量化后 greedy 即可用——这是"换对模型"比"调参救模型"的根本差异。

### 8.7 工程注意（本产线新增踩坑）

1. **venv 的 activate 不可用**：本 venv 从别处移入，`bin/activate` 硬编码旧 `VIRTUAL_ENV=/home/chenyipei/test_omc/venv`，source 后 PATH 指向不存在目录 → `python: command not found`。所有脚本一律用 `venv/bin/python` 绝对路径（export.sh 已改）。
2. **评测环境的 deepspeed 检查**：直接跑 eval_ppl.py 需 `CUDA_HOME=01_prepare/cuda_stub` + PYTHONPATH 带 dopt/_autopatch（pipeline.sh 已内置；run.sh 一直有）。
3. **GPU 被占时评测走 CPU**：`EVAL_DEVICE=cpu bash pipeline.sh instruct eval`（fp32 CPU 慢但数值等价；chat_test 由 CHAT_DEVICE 接管）。
4. **c512 量化的显存下限**：外部占用 ~14.7G（剩 ~14.7G 空闲）时 GPTQ c512/s1024 可跑通（本次实测）。
