# Instruct 版端侧交付目录（Qwen2.5-1.5B-Instruct / kirin9020 / int4 g64）

与 `../05_device_files_base/`（base 版）并列的 instruct 交付目录，文件命名全部带 `Instruct`
标记，与 base（`*_Base_*`）一眼可区分：

| 文件 | 说明 |
|---|---|
| `Qwen25_1b5_Instruct_kirin9020.omc` | NPU 离线模型（int4 权重 / int16 激活 / **g64** / lm_head fp） |
| `SubGraph_0.weight` | 外置权重（omg 固定名，与 base 同名——换模型必须整组 7 文件重推） |
| `model_instruct_64_2048.embedding_weights/.embedding_dequant_scale` | 分离 embedding（int8 表 + 逐行 scale） |
| `tokenizer.json` | 词表（与 base 同 vocab 151936，取自 Instruct 模型目录） |
| `context.json` | 采样配置：**对话参数**（T=0.7 / top-p=0.8 / top-k=20 / rep=1.1，官方推荐值；base 版为抗复读参数 T=0.6/k16/p0.95/rep1.2） |
| `executor.json` | 引擎配置（model_path 指向 `Qwen25_1b5_Instruct_kirin9020.omc`，eos=151645 `<|im_end|>`） |

## 演示流程（base → instruct）

1. 先在 `../05_device_files_base/` 跑 push 脚本推 base，App 内演示**续写**；
2. 再回本目录跑 `push_to_device_next.bat`（或 Git Bash 跑 `.sh`）推 instruct，**完全退出并重启 App**，演示**对话**。
3. App 侧发 prompt 时套 Qwen chat template（ChatML）：
   `<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n`
   （demo 工程若已内置 apply_chat_template 则直接输入问题即可；引擎侧已配 `<|im_end|>` 停止符）

## 产线入口（服务器侧）

```bash
bash pipeline.sh instruct          # 全流程：量化→评测→导出→转换→打包到本目录
bash pipeline.sh instruct status   # 查看产物状态
```
配方：int4 per-group 64 + 中文对话校准（Belle 多轮+单轮，ChatML 渲染，2401 行 / 856k token）
+ s1024 + c512 + lm_head 保 fp + PTQ。详见 ../QUANTIZATION.md §八。

## 排障

部署失败/加载报错的排查步骤见 `../06_demo_harmony_next_app/NEXT_端侧测试手册.md`
（hilog 抓取、LoadPrivateWeight 失败等案例；`diagnose_load.bat` 同款用法）。
