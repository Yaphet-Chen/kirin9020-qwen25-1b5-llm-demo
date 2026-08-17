# 版本管理与调试工作流

本目录已用 git 做版本管理。只跟踪**脚本/配置/文档**；所有大产物（venv、tools、models、.pth、.omc、.weight、embedding 等）用 `.gitignore` 排除，靠脚本重建。

## 文档分工

| 文档 | 内容 |
|---|---|
| REPRODUCE.md | 全流程操作手册（怎么跑） |
| QUANTIZATION.md | 量化机制/实验矩阵/最优配方/踩坑（为什么这样配） |
| VERSION_CONTROL.md | 本文档（git 工作流） |

## 仓库结构

```
跟踪的（进 git）:
  .gitignore
  REPRODUCE.md / QUANTIZATION.md / VERSION_CONTROL.md
  pipeline.sh  profiles/{base.env, instruct.env}          # 统一产线入口与双产线差异声明
  01_prepare/{make_venv.sh, prepare.sh}
  02_quant/{config.yaml, instruct_config.yaml, run.sh, edit_dopt_config.py,
            dopt_config.json.example, run_experiment.sh, eval_ppl.py, chat_test.py,
            device_compare.py(含 --emb/--greedy/--probe 模式), _autopatch/sitecustomize.py}
  02_quant/device_compare_report_3way_{sampling,greedy}.md   # 端云三方对比报告
  02_quant/data_zh/build_corpus.py                        # base 校准语料构建脚本
  02_quant/data_chat/build_corpus_chat.py                 # instruct 校准语料构建脚本
  03_onnx_export/{export.sh, model_info_target.yaml, model_info_instruct.yaml}
  04_omc_convert/convert.sh
  05_device_files_base/{context*.json, executor.json, pack.sh, push_to_device_next.*,
                        diagnose_load.bat, collect_phone_replies.sh, NEXT_端侧测试手册.md}
  05_device_files_instruct/{context*.json, executor.json, push_to_device_next.*,
                        diagnose_load.bat, README_DEMO.md}   # pack.sh 复用 base 目录的

不跟踪的（.gitignore 排除，可重建）:
  01_prepare/{venv,tools,models,cuda_stub}
  02_quant/{qwen25_1b5_9020/, qwen25_1b5_instruct_9020/, exp_*/, data_zh/, data_chat/, *_config.yaml}
  03_onnx_export/{output*/,dump/,npu_tuned_export/}    onnx 产物
  04_omc_convert/{model.*,Qwen25_1b5_*/}
  05_device_files*/{*.omc,*.weight,embedding*,tokenizer.json}
  logs/
```

## 日常调试工作流

### 改脚本/配置前：先开分支
```bash
cd qwen25_1b5_run
git checkout -b try-quant_param_2-true    # 例如想试验 quant_param_2=True
```

### 改动后查看差异
```bash
git diff                  # 看未暂存的改动
git diff --stat           # 只看改了哪些文件
```

### 改动有效 → 提交
```bash
git add 02_quant/config.yaml
git commit -m "test: quant_param_2=True 对比试验"
```

### 改动无效/想回到上一个好版本 → 丢弃
```bash
git checkout -- 02_quant/config.yaml     # 丢弃单个文件
git reset --hard HEAD                    # 丢弃所有未提交改动（慎用）
```

### 想对比两个版本的产物效果
```bash
git stash                    # 暂存当前改动
# 跑当前版本的流程，记录产物
# 切到另一个版本
git checkout <other-branch>
# 再跑流程，对比
git stash pop                # 切回来
```

## 常用命令速查

| 场景 | 命令 |
|---|---|
| 看提交历史 | `git log --oneline` |
| 看当前状态 | `git status` |
| 看改动 | `git diff` |
| 提交改动 | `git add <file> && git commit -m "说明"` |
| 丢弃未提交改动 | `git checkout -- <file>` |
| 新建试验分支 | `git checkout -b <分支名>` |
| 切回 main | `git checkout main` |
| 合并分支 | `git checkout main && git merge <分支名>` |
| 删除分支 | `git branch -d <分支名>` |
| 查看某文件历史 | `git log -p <file>` |
| 回到某历史版本看 | `git checkout <commit> -- <file>`（只取该文件） |

## 调试建议的提交粒度

- 每改一个**可独立验证的点**就提交一次（如：改 quant_param_2、改 group_size、改 prefill_len 各一次）
- commit message 写清**改了什么参数、为什么改**（方便日后 `git log` 查阅）
- 试验性改动用**独立分支**，确认有效再 merge 回 main

## 注意事项

1. **产物不进 git**：`.omc/.weight/.pth` 等大文件靠脚本重建。换机器/换分支后，按 REPRODUCE.md 重跑对应阶段即可。
2. **改 .gitignore 后**：`git rm -r --cached <已被跟踪但想忽略的文件>` 再提交。
3. **不要 `git add` 整个产物目录**：虽然 .gitignore 会拦，但养成只 `git add <具体脚本>` 的习惯更安全。
4. **远程备份**（可选）：若要推到远程仓库，先确认 `.gitignore` 生效（`git status` 无大文件），再 `git remote add origin <url> && git push -u origin main`。
