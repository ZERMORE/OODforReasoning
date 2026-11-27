# G1 几何分数与聚合交叉实验

G1 复用已生成的 Qwen2.5-7B-Instruct 轨迹。GSM8K 和半量 MATH 读取原始 OOD 生成目录，Step 组读取实验 A 的保留清单。三个实验分别报告；GSM8K 和 Step 组可能有共同题目，不能视为独立复现样本。历史测试轨迹已用于探索，本轮结论亦属探索性。

实际 LM head 输入上的分数包括相对中心范数、top-1/top-2 边界夹角偏离与距离、边界法向、径向和固定随机方向的全词表 Fisher 标准差。保存 top-2 token IDs 和原始位置，以便后续检查候选切换。在 Step 组上另有按步骤序号的无标签参考方向，由 A 的 train 轨迹拟合。每个方向是该步单位径向状态的归一化和，稀少位置回退到总体方向；只在正文汇总此角度。

基线是 NCI、fDBD-All、平均 log-prob、logit margin、全词表负熵和 top-20 负 Shannon 熵。原始结果如已保存完整逐词 fDBD，核对 token IDs、位置和平均值后复用；其他分数用相同的 matched replay 计算，并核验保存的旧分数。Step 组补算全词表 fDBD。当前不做干预或首次错误标注。其它聚合头、跨 checkpoint 及多采样一致性留在后续实验，避免将不同生成协议的得分直接混入 G1。

聚合用全平均、最差 64-token 窗、原仓库 Evidence Drop。短回答的窗口取全段。Step 组另报告正文 token 平均、步骤等权平均和最差步骤。对无预设极性的几何量，在 fit 集用平均分的 AUROC 选一次符号，再固定应用于各聚合。融合用 fit 训练标准化逻辑回归，比较角度＋Fisher、概率模型及概率＋角度＋Fisher。三种融合均使用均值特征。

传统 OOD 来源按规范化题目哈希约 20% fit、20% calibration、60% test。Step 集用原 train 作无标签参考，原 validation 按题哈希一分为二用于 fit 和 calibration，原 test 保持原样。校准用正确轨迹的下尾顺序统计量，默认目标 FPR 10%。测试集报告实际 FPR，不把它写为固定 10%。误答为正类，AURC 越小越好。区间对测试题组做配对 bootstrap，固定拟合结果和阈值，未调整多重比较。

## 本地同步

```powershell
cd E:\Research\LLMCoT\methods\OODmethod
git add geom_diagnostics scripts/run_geom_g1.sh docs/GEOM_G1.md
git commit -m "Add G1 head geometry diagnostics"
git push
```

## 服务器

```bash
cd ~/projects/OODmethod
conda activate ood-repro
git pull --ff-only
export DATA_ROOT=/mnt/data/wxh/wdl
export GEOM_ROOT="$DATA_ROOT/results/OODmethod/geom-diagnostics-v1"
export HF_HOME="$DATA_ROOT/cache/huggingface"
nvidia-smi
export CUDA_VISIBLE_DEVICES=7  # 选择空闲 GPU
```

确认三个来源存在：`results/OODmethod/qwen25-paper-v4-matched/main/qwen_gsm8k`、`results/OODmethod/evidence-math-half-v1/math/test-raw`、`results/StepCluster/qwen25-pilot-chat-v2`。模型默认使用各来源保存的快照路径。如目录不同，单独运行各任务时通过 `GEOM_SOURCE=/实际路径` 覆盖，Step 必须传 run 目录而非其 gsm8k 子目录。移动后的模型目录通过 `MODEL_PATH` 指向同名快照。

先短跑。例如每个划分 2 条：

```bash
SMOKE_LIMIT=2 bash scripts/run_geom_g1.sh smoke gsm8k
SMOKE_LIMIT=2 bash scripts/run_geom_g1.sh smoke math
SMOKE_LIMIT=2 bash scripts/run_geom_g1.sh smoke step
```

smoke 只回放，输出在 `$GEOM_ROOT/smoke/`。默认 30 条/划分。改变 smoke 数量时用新 `GEOM_ROOT`，避免与已有签名冲突。第一个 smoke 成功后可查看 `rows/*.json` 的 `score_audit`。

完整三组在一张 GPU 上顺序运行并写日志：

```bash
mkdir -p "$DATA_ROOT/logs/OODmethod"
LOG="$DATA_ROOT/logs/OODmethod/geom-g1-$(date +%Y%m%d-%H%M%S).log"
nohup setsid bash -c '
  set -euo pipefail
  for task in gsm8k math step; do
    bash scripts/run_geom_g1.sh all "$task"
  done
  echo G1_ALL_COMPLETE
' > "$LOG" 2>&1 < /dev/null &
echo "$!" > "$LOG.pid"
tail -n 30 -f "$LOG"
```

断开 SSH 后继续运行，Ctrl+C 退出 tail。可以用不同空闲 GPU 各起一个任务，如 `CUDA_VISIBLE_DEVICES=6 bash scripts/run_geom_g1.sh all math`。该直接命令在前台运行。同一输出目录由 flock 防止并发写入。全词表 Fisher 额外缓存 FP32 输出头（Qwen7B 约 2 GiB）；全词表 fDBD 回放较耗时。建议以 smoke 观察吞吐量，文件按题原子落盘。

重复同设置的 `replay` 会跳过已保存的问题。若完整串行任务在中间失败，只重启未完成的任务。可独立执行 CPU 分析：

```bash
ANALYSIS_TAG=analysis-v2 BOOTSTRAP=1000 bash scripts/run_geom_g1.sh analyze gsm8k
```

各任务输出为 `$GEOM_ROOT/{gsm8k,math,step}/`。见 `manifest.json`、`rows/*.json`、`analysis-v1/metrics.csv`、`protocol.json` 和 `scores.csv`。metrics 中 `*_delta` 是当前组合减去 reference，AURC 负差值代表改善。先看概率＋几何相对概率模型的配对增益，再看同分数的平均、最差窗和 drop。后续规则可以基于已保存的逐词评分在 CPU 上分析。

另附离线首错诊断入口 `python -m geom_diagnostics.locate --help`。它使用 G1 冻结的分数方向及现有首错标注，核对原行哈希后计算错误位置前后窗口的分数变化、候选切换率及最近 Evidence Drop 命中。该诊断仅使用有定位的测试错误，供后续 G2 对照设计使用。
