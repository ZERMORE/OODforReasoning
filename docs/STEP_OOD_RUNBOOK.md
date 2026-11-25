# Step 级 NCI / fDBD 实验

本扩展新增 `scripts/run_step_ood.sh` 和独立包 `step_ood`；所有旧启动入口、评分公式及旧结果保持原样。不新增依赖。新模块放在原 `ood_repro` 包之外，避免改变原实验对整个包计算的实现哈希，影响旧实验断点恢复。

## 要验证什么

保持同一条 Step 格式的 Qwen 推理回答，比较“把全部 token 混合平均”和“先在步骤内平均，再检查局部异常”。直接复用 StepCluster 已生成的实际 prompt/output token IDs 和最终答案标签，用原 OOD GeometryScorer 与 matched replay 计算原始 logits 下 NCI、fDBD_all。无需再次生成，不会把每步改为独立问题，也不会重置上下文；自回归依赖始终保留。

默认来源为 StepCluster 的 `qwen25-pilot-chat-v2`：train 600、validation 200、test 200。既有 StepCluster pilot 完成后可以直接使用。也支持独立 RUN_NAME 的 Step 格式数据，但不能直接拿原 OOD 自由格式回答套 Step 解析器。

对第 j 步的内容 token 集 I_j，计算 a_j=mean_{t in I_j} s_t，其中 s_t 是现有 NCI 或 fDBD 分数，越高越自信。Step N: 字面标记、#### 后最终答案和 EOS 不计入步骤内容；标题冒号后的描述属于内容，正文内的换行也属于内容。跨越标记/内容边界的 BPE token 排除，不强行拆 token。保存每步实际 token 位置和数目，可以审计对齐。严格要求从 Step 1 连续编号，不允许未分配前言或空步骤。

| 规则 | 轨迹置信度 | 作用 |
|---|---|---|
| full_mean | 全部生成 token 分数平均，排除 EOS | 同回答上的旧式全轨迹基线，含步骤标记和最终答案 |
| reasoning_token_mean | 各步骤均值按 token 数加权 | 同样内容范围的对照，隔离排除标记/答案的影响 |
| step_mean | 各步均值等权平均 | 避免长步骤支配轨迹 |
| worst_step | 最低步骤均值 | 任意一步低于阈值则报警；保存首个触发步骤 |
| worst_two_steps | 最低两个步骤均值再平均 | 减弱单步偶发低值影响 |
| worst_adjacent_two | 相邻两步均值的最小值 | 检查持续的局部低分；它不是“两步各自都低于同一个阈值” |
| low_step_fraction | -低分步骤比例 | 检查低分步骤是否广泛出现 |

只有一步时，两步规则退化成单步。low_step_fraction 的“低分”参考阈值只用正确 train 轨迹中所有步骤均值的第 10 百分位估计；它是按步骤池化的经验参考，不保证每步 10% FPR。正确最终答案不保证中间每步正确。

## 校准与评估

每个指标/规则的最终轨迹阈值只使用正确 validation 轨迹，预设目标 FPR=10%。令 n 为正确校准轨迹数，k=floor(0.1*(n+1))，阈值为校准分数第 k 小值，严格用 confidence<threshold 报警；k=0 时禁用报警并记录。这是保守的 split-conformal 下尾规则：正确校准与测试轨迹可交换时，单个固定规则有边际 FPR 控制；不保证每个步数分组、所有规则联合或分布变化后的 FPR。离散分数/并列值可能使报警非常保守。pilot 的前缀子集用于探索，不据此声称泛化保证成立。

test 同时报告错误正类 AUROC、AP（随机基线=错误率）、FPR、TPR、precision、accuracy 和混淆计数。主指标用 AUROC/AP，阈值操作点用 FPR/TPR；accuracy 不是主要证据。保存按总步数 K 的分组结果，检查最差步规则是否因步骤更多而更容易误报。bootstrap 对所有方法共用按题抽样，给出 AUROC/AP 的 95% CI 及相对两个平均基线的配对差值 CI；这些 CI 条件于已固定的校准阈值，不覆盖校准集不确定性。

没有真实步骤对错标签，因此当前评价的是“步骤信号能否预测整题最终答案错误”。不能声称低分步骤就是首次错误，不能计算真实 step-level AUROC，也不自动接入 steering 或提前停止生成。保存的首个触发步骤是事后预警位置。若以后在线触发，需另测延迟、前缀校准及干预收益。

同一批轨迹上的 full_mean / reasoning_token_mean 对照用于比较聚合。原 OOD 的自由格式/不同提示结果只能作为另一生成协议，不能把正确率或检测差异都归因于步骤聚合。

## 1. Git 同步

本地 PowerShell：

```powershell
cd E:\Research\LLMCoT\methods\OODmethod
git status --short
git branch --show-current
git add step_ood scripts/run_step_ood.sh tests/test_step_ood.py docs/STEP_OOD_RUNBOOK.md
git commit -m "Add separate step-wise NCI and fDBD detection study"
git push origin HEAD
```

服务器：

```bash
cd ~/projects/OODmethod
conda activate ood-repro
git pull --ff-only
export DATA_ROOT=/mnt/data/wxh/wdl
export HF_HOME="$DATA_ROOT/cache/huggingface"
export MODEL_PATH="$DATA_ROOT/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28"
export PYTHONUNBUFFERED=1
nvidia-smi
export CUDA_VISIBLE_DEVICES=7  # 改为与 StepCluster 任务不同的空闲 GPU
```

不需要重新安装环境，也不需要再次下载模型或数据。使用哪个远程分支以本地实际当前分支为准；服务器需处于同一个分支。不要把 API key 文件加入提交。

## 2. 先复用已完成的 8 题 smoke

```bash
export STEP_SOURCE_RUN=qwen25-smoke-chat-v2
bash scripts/run_step_ood.sh smoke test
```

默认只回放两题，验证真实模型的对齐、评分和文件输出。没有重新生成。输出为 `$DATA_ROOT/results/OODmethod/step-ood-v1/qwen25-smoke-chat-v2/gsm8k/smoke/test`。仅 test 的 smoke 无法运行三划分校准分析，这是预期的。

## 3. 完成 StepCluster pilot 后启动正式步骤分析

所有三个来源划分的 generation.json 必须 complete=true。可以与 StepCluster 后续 CPU probe/绘图同时运行；若 StepCluster 还在生成，只能先回放它已完成的划分，或等待三划分完成再执行 all。all 会先检查全部来源，未完成会立即退出。

```bash
export STEP_SOURCE_RUN=qwen25-pilot-chat-v2
export ANALYSIS_TAG=main
export BOOTSTRAP=1000
export TARGET_FPR=0.1
export STEP_QUANTILE=0.1
export KS=all
mkdir -p "$DATA_ROOT/logs/OODmethod"
LOG="$DATA_ROOT/logs/OODmethod/step-ood-$STEP_SOURCE_RUN-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash scripts/run_step_ood.sh all > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOG.pid"
echo "PID/PGID: $PID"
echo "LOG: $LOG"
tail -n 30 -f "$LOG"
```

断开 SSH 后继续；Ctrl+C 只退出日志查看。出现 STEP OOD COMPLETE 为整套完成。NCI/fDBD 的全词表距离计算需要 GPU，回放速度不等于原生成速度。选择一张不同的空闲 GPU，避免与 StepCluster 争用显存。默认 all 是精确 fDBD_all，改成 KS=100 是另一种近似设定，必须另设 STEP_OOD_ROOT，不能与 all 混为一组。

若想在 train 生成完成后提前回放：

```bash
bash scripts/run_step_ood.sh replay train
# validation/test 生成各自完成后，分别执行
bash scripts/run_step_ood.sh replay validation
bash scripts/run_step_ood.sh replay test
bash scripts/run_step_ood.sh analyze
```

这些直接命令是前台命令；需要断开 SSH 时也用 nohup setsid 包装。每条来源轨迹原子性由来源工程保证，回放结果逐题追加并 flush，完整记录通过哈希校验。相同来源、评分代码和参数下支持 --resume；输出已完成时直接复用。如果 all 的分析阶段失败，先修复报错，再用新 ANALYSIS_TAG 单独 analyze，不必重新回放 GPU 分数。若已有完整 main 分析而仅想重做报告：

```bash
ANALYSIS_TAG=report-v2 bash scripts/run_step_ood.sh analyze
```

## 4. 输出与解读

默认根目录：`/mnt/data/wxh/wdl/results/OODmethod/step-ood-v1/qwen25-pilot-chat-v2/gsm8k/`。

- replay/train、validation、test：manifest.json 与 predictions.jsonl；保存逐 token 分数、每步平均、原文与来源哈希。格式错误/截断保留排除记录，不悄悄删除。
- analysis/main/metrics.csv 和 metrics.json：2 个指标 × 7 种规则，共 14 行主结果。
- step-scores.csv：所有保留轨迹的逐步均值、token 数、train 参考低分标志及校准最差步阈值标志。
- test-decisions.csv：每条测试轨迹各规则的置信度与报警；worst_step 还有首个触发步骤。
- metrics-by-step-count.csv：按 K 分组的性能与误报率。
- protocol.json：阈值、来源、代码哈希、排除统计与 bootstrap 设置。
- excluded.csv：存在排除样本时生成；没有排除时不生成该文件。

先看 worst_step / worst_two_steps / worst_adjacent_two 相对 reasoning_token_mean 的配对差值与 CI，再看固定 FPR 下的 TPR 和按 K 的 FPR。不要在 test 选最好规则、阈值或分位数后当作最终确认结果。主表仍保留全部预声明规则；如果后续需挑一个部署规则，应另用 validation 选择和独立新测试集确认。

全量 StepCluster 轨迹完成后，仅替换来源名字：

```bash
export STEP_SOURCE_RUN=qwen25-full-chat-v2
# 用上面的新日志/nohup命令启动 all，默认自动进入独立新结果目录。
```

本地取回 pilot 结果：

```powershell
scp -r wxh@rtx6000.yangtan.top:/mnt/data/wxh/wdl/results/OODmethod/step-ood-v1 E:\Research\LLMCoT\methods\OODResults\
```
