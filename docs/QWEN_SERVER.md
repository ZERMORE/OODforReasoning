# 先执行 Qwen2.5-7B-Instruct 的论文实验

目标是 arXiv:2602.07253v4 Table 4 的 Qwen 三组回答级错误检测。模型为 **Qwen/Qwen2.5-7B-Instruct**，不是 Qwen2-7B、Qwen3 或量化版本。公开模型无需 Llama 审核；模型来源：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct 。

## 1. 本地推送后，在服务器更新

```bash
conda activate ood-repro
cd ~/wdl/data/projects/OODmethod
git pull --ff-only origin codex/ood-reproduction
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
nvidia-smi
```

保留现有代理设置。不要升级 Transformers/Hugging Face 到不兼容版本。7.61B 原始模型的 BF16 权重约 15.2 GB（十进制），还需 KV cache 和评分内存；建议空闲 24 GB 及以上显存。使用已获分配的 GPU；下例 GPU 0 请按实际分配修改。多卡可见时当前引擎仍为单设备，不会自动分片。

## 2. 准备公开权重与固定版本的数据

```bash
bash scripts/run_qwen.sh prepare
```

默认根目录 `/mnt/data/wxh/wdl`，所有新模型/数据缓存和结果都写在数据盘：

```text
/mnt/data/wxh/wdl/
  models/Qwen2.5-7B-Instruct/<model-commit>/
  cache/huggingface/{hub,datasets}/
  tmp/
  results/OODmethod/qwen25-paper-v4/
    recipe.json
    smoke/
    main/
```

prepare 使用公开下载（不需要 Token），将模型 main 解析为 commit SHA，下载 Safetensors、tokenizer 和配置；三份数据也解析到各自 commit，核对评测 split 题数并记录内容 hash。recipe 仅在全部准备完成后写入，重复 prepare 复用原固定版本；改模型 revision 请使用新 `--name`。文件下载中断时重新 prepare 可利用已有文件恢复。

默认评测：CSQA validation=1221、GSM8K test=1319、AQuA raw validation=254。prepare 不下载 Llama，也不进行模型训练。

## 3. 三任务小样本检查

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen.sh smoke
```

每任务前 20 题，greedy、BF16、max_new_tokens=512，先用 **k=100** 做较便宜的评分/格式检查，前三题验证在线与重放，保存逐 token 分数。该阶段没有 All，不作为论文主结果；结果在 smoke/summary.md、summary.csv、summary.json。

检查 summary 中的 scored/total、parse failure 和 truncation，并人工抽查各任务 `prompt-example.txt`、`predictions.jsonl`。如果校验失败，应查精度与提取位置；不要仅为通过而放宽容差。若 BF16 不支持，可显式加 `--dtype float16`，后续 main 也采用相同参数并报告差异。

更改题数可用 `--smoke-limit 5`，但已经存在的同名 smoke 目录会做续跑一致性检查；改设置应换 `--name` 并重新 prepare，不混用结果。

## 4. 主实验

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen.sh main
```

主实验需要之前已完成三个 smoke cell。按顺序运行：

| 任务 | 本次评分 | 论文 PPL | NCI | All | selected-k |
|---|---|---:|---:|---:|---:|
| CSQA | NCI、PPL 排序等价分数、fDBD k100 与 All | 61.94 | 71.60 | 71.50 | 72.47 |
| GSM8K | NCI、PPL 排序等价分数、fDBD All | 71.54 | 75.83 | 77.19 | 77.19 (All) |
| AQuA | NCI、PPL 排序等价分数、fDBD k100 与 All | 71.66 | 78.19 | 77.08 | 78.22 |

表中数字为论文 AUROC×100 参考值，**不是实测值**。本次先固定论文 k，不在评测集重新选 k。当前参考实现的 All 较慢，这是正式论文对照的一部分，不能用 k100 结果替代 Qwen+GSM8K 的 All。

每 cell 默认问题级 2000 次配对 bootstrap；全部完成后自动生成 main/summary.md、summary.csv、summary.json，列出实测、论文、差值（百分点）、样本数、失败率与 JSON 中的 CI。summary 只接受完整 split；不完整预测不标记为主结果。

可重新汇总已有结果：

```bash
bash scripts/run_qwen.sh report
```

summary 的偏差不自动意味着实现错误；需结合 prompt 来源差异、生成准确率、失败率和 CI 解释。模型许可、完整论文协议差异及主表未实现基线仍遵循 README。

## 5. 长任务与续跑

```bash
tmux new -s ood-qwen
# 在 tmux 中
conda activate ood-repro
cd ~/wdl/data/projects/OODmethod
mkdir -p /mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4/logs
set -o pipefail
CUDA_VISIBLE_DEVICES=0 bash scripts/run_qwen.sh main 2>&1 \
  | tee -a /mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4/logs/main.log
```

Ctrl+B 然后 D 脱离会话，`tmux attach -t ood-qwen` 恢复。再次执行相同 main 命令会复用固定模型/数据版本，并续跑未完成样本。不要向该入口传统一 `--output`；三个 cell 已有各自目录。

如需单独的新实验组：每条命令都加相同 `--name my-qwen-run`；修改存储位置则每条都加相同 `--data-root /实际路径`。

当前 suite 不执行独立训练集选 k、TruthfulQA、经验均值消融或多样本基线，不把“代码与 tiny 模型测试通过”写成“7B 实验已完成”。

## BF16 在线/重放校验失败后的更新与恢复

旧版本把完整 prompt prefill 的在线路径与分块 prefill 重放直接比较。BF16 下，不同形状的矩阵乘与 attention 可产生不同舍入，因此序列分数可能超过统一容差。此日志本身不能证明 token 对齐错误，也不能直接视为已验证的数值等价。参见 [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) 与 [Transformers 4.46.3 Cache 文档](https://huggingface.co/docs/transformers/v4.46.3/en/kv_cache)。

修复版默认采用 `--verify-replay-mode matched`：整段 prompt prefill + 单 token 解码的 teacher forcing。每个生成位置与 token ID 逐一检查，原有 `atol=0.001, rtol=0.01` 保留；FP32 独立全序列与分块对齐测试也保留。两条路径使用 `DynamicCache`，避免 tuple-cache 的弃用路径。若再次失败，检查 cell 中新生成的 `verification-failure-*.json`。

更新后实现 hash 与配置字段均变化，**不要向旧 smoke/main 目录强行续写**。建议使用新实验名，保留旧的一题预测作为诊断。以下模型 commit 来自本次用户服务器日志：

```bash
conda activate ood-repro
cd ~/projects/OODmethod  # 或你的实际 checkout
git pull --ff-only origin codex/ood-reproduction
python -m pip install -e . --no-deps

bash scripts/run_qwen.sh prepare \
  --name qwen25-paper-v4-matched \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28

CUDA_VISIBLE_DEVICES=2 bash scripts/run_qwen.sh smoke \
  --name qwen25-paper-v4-matched

# smoke 完成并检查结果后
CUDA_VISIBLE_DEVICES=2 bash scripts/run_qwen.sh main \
  --name qwen25-paper-v4-matched
```

model commit 目录与已有下载一致，prepare 可复用权重；数据仍会解析固定版本并记录在新 recipe。默认存储根目录依旧为 `/mnt/data/wxh/wdl`。不需要删除旧结果、重新申请模型权限或升级 Transformers。恢复后仍需实际服务器 7B smoke 验证，不能用本地 tiny 测试替代。

修复本地验证：38 项测试分两组通过（13 项引擎/校验/CLI + 25 项评分/协议/Qwen suite）。新增 400-token prompt 的 CUDA BF16 Qwen2 回归测试，严格验证 online 与 matched replay 的调用形状一致，并在 rtol=0、atol=0 时逐 token 通过；另测试均值抵消不能掩盖逐 token 错误、错位必须失败、失败诊断完整落盘。

## 本地验证与执行状态

2026-10-22：原有 25 项核心/CLI/CUDA tiny 模型测试通过；新增 7 项 Qwen 准备、固定版本复用、三任务调度、结果比例换算、完整 split 检查和权重分片完整性测试通过。新增准备/调度测试采用明确的合成 fixture，不联网下载完整模型；Bash 入口通过语法检查。

服务器 SSH 当前返回 `Permission denied (publickey,password)`，尚未远程启动 Qwen 7B。只有在服务器生成实际预测与 metrics 后，才可填写实测主结果。
