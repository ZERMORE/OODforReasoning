# 序列聚合与错误传播：执行指引

本次已实现：逐 token 全指标回放、E1 聚合消融、E2 顺序控制、E3 标注准备/位置分析、E4 人工核验后的三分支续写与行为/置信度曲线。尚未运行服务器完整实验，不提供估计实验成绩。

E3/E4 需要人工确认过程错误/正确步骤和语义编辑；没有标签时程序会拒绝执行依赖标签的分析。本指引保留原入口；新增监督 GRU/CNN/Attention 和长度匹配首错分析见 [EVAL_SECTION2.md](EVAL_SECTION2.md)。自动语义步骤标注、E4 等效恢复检验及基于机制标签的阈值漏检分析尚未实现。完整设计见 [SEQUENCE_AGGREGATION_PLAN.md](SEQUENCE_AGGREGATION_PLAN.md)。

## 1. 同步代码

本地 PowerShell（下列包含本轮及此前尚未提交的 DeepConf/主表文件；只提交检查后确认的项目改动）：

```powershell
cd E:\Research\LLMCoT\methods\OODmethod
git status --short
git add README.md docs ood_repro scripts tests
git commit -m "Add audited sequence aggregation and error propagation experiments"
git push origin codex/ood-reproduction
```

服务器：

```bash
conda activate ood-repro
cd ~/projects/OODmethod
git status --short
git switch codex/ood-reproduction
git pull --ff-only origin codex/ood-reproduction
```

服务器有本地改动或拉取冲突时先保留并处理，避免覆盖。使用原模型和原依赖版本，回放会检查 checkpoint/tokenizer commit、软件包以及所有原评分。不要为本轮随意升级 torch/transformers。CPU 图表使用已有 `requirements-viz.txt` 依赖；若服务器未装 matplotlib，可单独安装 `matplotlib>=3.8,<4`，不要顺手升级模型运行依赖。

## 2. 先做 GSM8K 的 20 题 smoke

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sequence.sh smoke gsm8k
```

成功时输出 `Complete: 20 audited trajectories`。此步骤核对全部旧评分并保存逐 token 新分数，不要求这 20 题同时包含正确和错误样本，因此不在 smoke 上强行计算 AUROC。

默认源目录：`/mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4-matched/main/qwen_gsm8k`。

默认新研究目录：`/mnt/data/wxh/wdl/results/OODmethod/sequence-study-v1`。

如果路径不同，设置 `SOURCE_ROOT`、`STUDY_ROOT`；如果 checkpoint 移动，设置 `MODEL_PATH` 为**同一个 revision**的新本地路径：

```bash
SOURCE_ROOT=/你的/原结果根目录 MODEL_PATH=/你的/同版本模型目录 \
  CUDA_VISIBLE_DEVICES=2 bash scripts/run_sequence.sh smoke gsm8k
```

这些环境变量需在正式执行时继续传入。脚本只从本地加载模型；原始结果只读。新增记录包含原 token/标签/回答、全部原评分的回放误差、新 token 分数、token pieces 和文件/记录哈希。

## 3. 后台执行完整 E1/E2

在同一个 conda 环境和项目目录下：

```bash
mkdir -p /mnt/data/wxh/wdl/logs/OODmethod
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  bash -c 'bash scripts/run_sequence.sh replay gsm8k && bash scripts/run_sequence.sh analyze gsm8k' \
  > /mnt/data/wxh/wdl/logs/OODmethod/sequence-gsm8k.log 2>&1 < /dev/null &
echo "PID: $!"
```

回放使用 GPU；分析仅用 CPU，不加载模型。`&&` 保证回放失败后不运行分析。日志查看：

```bash
tail -f /mnt/data/wxh/wdl/logs/OODmethod/sequence-gsm8k.log
```

Ctrl+C 只退出日志查看；实验后台继续。再次执行相同 replay 命令会按哈希校验并续跑；改变 checkpoint、代码实现或源结果后不能混入旧运行。若最后一行因中断损坏，程序会报错，需要检查该行，不能自动忽略任意坏记录。

E1 主参数：窗口 64、步长 1，同时完整报告 32/128；分数高表示可信。E2 默认对 `fdbd_all` 做 1,000 次顺序控制；其它评分使用同 seed/同排列规则，可扩展但不按结果挑选。

扩展到两个评分的 E2：

```bash
PERMUTATION_METRICS=fdbd_all,nci ANALYSIS_TAG=all-metrics \
  bash scripts/run_sequence.sh analyze gsm8k
```

分析输出目录必须是新目录；重复分析使用新的 `ANALYSIS_TAG`。`BOOTSTRAP` 默认 2000、`PERMUTATIONS` 默认 1000。只检查导出格式时可减小为 20/3 并使用 `ANALYSIS_TAG=debug`，该结果不能充当正式 CI。

## 4. 查看结果

```text
sequence-study-v1/
  smoke/qwen_gsm8k/                     # 与完整运行分开
  replay/qwen_gsm8k/
    manifest.json                      # 源哈希、版本、模式与 complete
    predictions.jsonl                  # 逐 token 分数与所有原评分审计
  analysis/main/qwen_gsm8k/
    report.md                          # 全评分×聚合的主表及 CI
    aggregation.csv                    # 可直接读取的数值表
    analysis.json                      # 长度分层、短轨迹占比、全部 E2 统计
    aggregation_matrix.png/.svg        # E1 矩阵图
    order_controls.png/.svg            # E2 原顺序与置换分布
```

优先查看 `fdbd_all/lowest_w64` 相对同评分 `mean` 的差及配对 CI。若窗口提升，再比较 token_bottom10，判断是否只是低分尾部效应；不要直接归因为时序传播。

E2 的 `distribution95` 是置换 AUROC 的分布区间，不是模型性能 CI。另提供基于最多前 16 个固定置换的配对问题级 bootstrap 区间，字段为 `paired_question_ci95_fixed_null_subset`，其估计对象与 1,000 次置换总平均不同，已在 JSON 注明。`diagnostic_p_two_sided` 是顺序破坏诊断，不是因果证明；多指标 CI 未作多重比较校正。反转应保持最低窗口评分不变。短于窗口的轨迹等价于全段均值，数量在 JSON 单列。

完成 GSM8K 后，按相同冻结参数执行：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sequence.sh replay csqa
bash scripts/run_sequence.sh analyze csqa
CUDA_VISIBLE_DEVICES=2 bash scripts/run_sequence.sh replay aqua
bash scripts/run_sequence.sh analyze aqua
```

## 5. E3：先准备独立 train 校准，再标注首错位置

以下 shell 变量仅是本次研究的路径简写：

```bash
study_root=/mnt/data/wxh/wdl/results/OODmethod/sequence-study-v1
original_run=/mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4-matched/main/qwen_gsm8k

python -m ood_repro.calibration_config \
  --source "$original_run" \
  --output "$study_root/protocol/calibration-gsm8k" \
  --run-output "$study_root/calibration/raw/qwen_gsm8k"

CUDA_VISIBLE_DEVICES=2 python -m ood_repro run \
  --config "$study_root/protocol/calibration-gsm8k/config.json"

CUDA_VISIBLE_DEVICES=2 python -m ood_repro.replay_scores \
  --source "$study_root/calibration/raw/qwen_gsm8k" \
  --output "$study_root/calibration/replay/qwen_gsm8k" --resume

python -m ood_repro.error_positions prepare \
  --replay "$study_root/replay/qwen_gsm8k" \
  --output "$study_root/annotations/gsm8k-template"
```

校准生成配置继承原模型、prompt 和采样设置，只使用 train、固定 seed=1729、200 题，并由现有 loader 排除 prompt 演示问题。E3 再核验题目 ID/文本不与测试重合。若需严格区分 train pilot，先冻结并排除其 ID；当前配置生成器不自动预留独立人工 pilot 清单。

复制 `annotations.jsonl` 为 `annotations-reviewed.jsonl` 进行标注；可仅保留已审核样本。不要改样本 ID 和 source_row_hash。字段说明：

- `reviewed=true` 表示已核验；`process_valid` 为布尔值，不能由 final_correct 自动推断。
- `first_error_position` 是原生成序列中第一处可核验错误跨度的起始 token 索引，从 0 开始；不能定位时留 null，并填写说明。正确过程必须为 null。
- `recovered` 标记后续是否修复错误；不把最终正确自动等同于过程无错。
- `steps` 为 scored-position 的 `[start,end)` 区间，必须按顺序覆盖全部计分 token；只做 E3 位置分析可以不填，做步骤平均或准备 E4 时必须填。
- `token_pieces` 帮助定位，但它们是 tokenizer 的内部片段，不保证逐片拼接就等于可读文本；结合 response/原 token ID 审核。

审核后执行（需覆盖全部关注的最终错误题，并盲审一部分最终正确题；未审核样本会单独报告）：

```bash
python -m ood_repro.error_positions analyze \
  --replay "$study_root/replay/qwen_gsm8k" \
  --calibration "$study_root/calibration/replay/qwen_gsm8k" \
  --annotations "$study_root/annotations/annotations-reviewed.jsonl" \
  --output "$study_root/positions/gsm8k-v1"
```

输出 `positions.json`：独立 calibration 阈值、测试实际 FPR、每桶错误数/TPR/共享正确对照 AUROC/CI，及无法定位错误数。少于 20 个错误的桶标为仅描述。当前没有长度匹配对照，不能据此声称首错位置的因果效应。

若 steps 已覆盖所有回放样本，可用 `python -m ood_repro.sequence_analysis ... --steps annotations-reviewed.jsonl` 增加步骤等权平均；不完整步骤集会拒绝导出，避免不同方法比较不同题集。

## 6. E4：人工核验编辑后运行三分支续写

先在 train pilot 做少量测试，再选测试集正式样本。以下为测试集准备接口，不代替 pilot：

```bash
python -m ood_repro.prepare_interventions \
  --replay "$study_root/replay/qwen_gsm8k" \
  --annotations "$study_root/annotations/annotations-reviewed.jsonl" \
  --output "$study_root/interventions/gsm8k-events-template" \
  --max-questions 100
```

该命令只加载本地 tokenizer，不加载模型权重。从已核验过程正确、最终正确且有步骤标注的轨迹中固定种子随机选题，再为靠近 25%/75% 的不同中间步骤创建候选。输出候选数量和排除后的实际规模，**不自动保证该步骤可作算术编辑**。

复制 events.jsonl 到单独的 events-reviewed.jsonl，逐项确认：

- `original_step_text` 是正确步骤；start/end 是原生成 token 范围，包含该步骤的全部必要空格/标点。
- `error_step_text` 只引入一个明确计算/逻辑错误，保持语法与风格。
- `sham_step_text` 是语义正确的等价编辑；token-identical 的假编辑会被拒绝。
- 删除不可编辑候选；保存审查/排除原因，不按模型结果筛选。审核完成再将 reviewed 改为 true。

审核后执行：

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m ood_repro.interventions \
  --source "$original_run" \
  --events "$study_root/interventions/events-reviewed.jsonl" \
  --output "$study_root/interventions/gsm8k-v1" \
  --seeds 0,1,2 --temperature 0.7 --max-new-tokens 256 --resume
```

可按第 3 节同样的 nohup 方式后台运行。不同分支重新建 cache，prompt 不变，干预后的 token 从完整前缀续写；只对新生成部分计算分数，排除人为注入的 token。输出 token 长度变化以便核查编辑混杂。

产物：`predictions.jsonl`、`behavior.json`、`confidence_curves.json`、`continuation_confidence.png/.svg`。行为差值先在每题内平均多个位置和种子，再按问题 bootstrap。置信曲线只比较同 event/seed 两分支都完整生成了该区间的配对，并记录有效题数；仍存在 EOS 生存选择，不能把差异不显著当作置信度恢复。

`used_injected_error`、`self_corrected` 默认 null；自动报告只回答“错误干预是否改变最终错误率/评分”，不能自动证明逻辑传播。机制核验结果应另存标签文件，不直接编辑带哈希的原始 predictions。正式恢复等效检验、阈值漏检率需在完成对应 calibration 和机制标签后另行实现。

## 7. 本地验证与同步结果

```bash
python -m pytest -q tests/test_sequence_study.py
```

测试使用随机 tiny Qwen2 和合成数据，不下载模型；测试图仅验证布局，不是论文实验结果。完整服务器实验完成后，可将 `sequence-study-v1/analysis` 和 E4 report/curves 同步本地分析；保留 replay JSONL 以便追溯。无需把模型/大数据文件加入 Git。
