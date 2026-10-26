---
title: OODeval2 首错位置与序列聚合实验
created: 2026-10-26
source: https://arxiv.org/abs/2602.07253v4
repository: E:/Research/LLMCoT/methods/OODmethod
tags:
  - llm
  - reasoning-evaluation
  - uncertainty-quantification
---

# OODeval2：首错位置与序列聚合实验

本轮实现用户提出的实验1（首错位置与检测敏感度）和实验2（聚合方式及轻量序列模型）。**本轮不实施实验3的反事实注入。** 编号与旧计划不同：本轮实验1扩展旧 E3；本轮实验2复用旧 E1/E2，并新增监督学习对照。当前仅完成代码与本地测试，正式实验需用户同步至服务器运行。

## 一、理论部分：研究问题及证据边界

正确的命题是：每步几何分数 $s_t=g(h_t,W)$ 的状态 $h_t=f(x,y_{<t})$ 已包含自回归历史；全段均值 $S=T^{-1}\sum_t s_t$ 可能丢失显式顺序、局部异常和错误后的动态信息。**平均不等于独立同分布假设，分类头也不是一组独立模型。** 均值可对相关变量计算，不构成理论矛盾。

既有 Qwen2.5-7B-Instruct/GSM8K 的 1,319 条轨迹已表明：fDBD mean 77.93，lowest-window64 77.76，差 −0.17 [−2.05,+1.83]；顺序打乱未显示真实排列优势。这些结果不能因为不符合原预期而忽略，本轮是在已观察结果之后的进一步分析，不能宣传为全新盲测。

本轮分别检验：

| 问题 | 主要比较 | 结论上限 |
|---|---|---|
| 实验1：检测率是否依赖首错位置？ | fDBD mean 在早期/晚期首错组的 TPR 差；辅以长度匹配 AUROC | 自然错误位置的相关性，不是位置因果效应 |
| 实验2：顺序建模是否提供额外预测信息？ | GRU vs mean、统计特征 Logistic Regression、DeepSets，以及训练/测试打乱对照 | 在固定训练数据、评价集和输入分数下的预测增益 |
| 具体传播机制是否成立？ | 本轮没有干预/传播标签作完整验证 | 不从本轮结果推出级联传播或内部因果链成立 |

不预设早期必然更差：早期异常可能被高置信度后缀稀释，也可能持续低分，或被后续自纠错。Min/分位数不使用顺序，优于均值只能表明尾部分布有用。后部加权变好也不直接等于捕获了传播。

## 二、数据与冻结协议

- 测试：直接复用 `/mnt/data/wxh/wdl/results/OODmethod/sequence-study-v1/replay/qwen_gsm8k`，不重生成、不改标签。
- 模型、revision、prompt、BF16、greedy 和 512-token 上限继承原实验。pool/test 的模型与 tokenizer commit、prompt pack、dtype、采样协议要一致。
- 监督 pool：从 GSM8K **train** 生成 2,400 题，seed=1729，现有 loader 排除 few-shot 演示题；生成时保存在线 token 分数。CPU 导入逐题核验 token 聚合与旧分数，避免额外模型回放。
- 以规范化 question 文本为组，固定 seed=1729，按组分层划分约 70% train / 15% validation / 15% calibration。当前一题一条轨迹时约 1,680 / 360 / 360。
- train 用于拟合及归一化；validation 只用于选择神经模型 epoch；calibration 只用于可信阈值；test 只用于最终报告。splits.json 保存具体 ID 和 pool/test 文件哈希。
- 同题的不同 ID/轨迹不得跨 train/validation/calibration。test 必须每题一条轨迹，以保证当前 bootstrap 的问题单位；多轨迹测试需另做分组评价，不直接套用此入口。
- 默认训练错误至少 20 个，其他分区每类至少 2 个；这些只是程序可运行下限，不是统计功效保证。错误太少应增加训练池规模，不能尝试不同 split seed 来挑结果。
- 新实验仍沿用已观察过的 GSM8K 测试集。后续应在未用于设计的任务/模型复核；不按本测试集结果调隐藏维度、窗口或学习率。

## 三、实验1：首错位置与检测敏感度

### 3.1 标注与定义

`eval2 prepare-annotations` 自动收集全部最终错误轨迹，并固定种子抽取等量最终正确轨迹，随机打散顺序。当前预计 138 个错误 + 138 个正确；输出中隐藏几何评分和显式最终正确标志，但包含 gold 与回答，所以不是完全盲审。

每题核验以下字段，保留 sample_id/source_row_hash：

| 字段 | 定义 |
|---|---|
| `reviewed` | 人工核验完成后改为 true |
| `process_valid` | 推理过程是否正确；最终正确不自动意味着过程正确 |
| `first_error_position` | 第一处可核验错误命题的起始 **生成 token 索引，从0开始**；不是 prompt token 位置 |
| `first_error_span_end` | 可选，错误跨度的结束索引，右端不包含；用于审查记录 |
| `first_error_step` | 可选，语义步骤编号，从1开始；当前主要统计按 token 相对位置 |
| `recovered` | 后续是否明确修复该错误；不从最终答案自动推断 |
| `notes` / `reviewer` | 依据、争议及核验者 |

配套 tokens.tsv 只给 token 索引/ID/piece，不给评分。piece 不等于完整可读字符，应结合 response 审核。第一错误以语义步骤/命题为准，再映射到 token；不要凭最低分位置生成“真值”。

全部最终错误题都必须有已审核记录，避免只挑可支持假设的样本。无法判定时允许 `process_valid=null, first_error_position=null, reviewed=true`，但必须在 notes 写明原因；过程确定错误却无法精确定位，也可 `process_valid=false, first_error_position=null` 并说明。程序分别报告不可判定和未定位数，不强行分桶。

过程正确但抽取失败导致最终错误者，不应编造推理首错。最终正确但曾出错/自纠错者单列，不混入最终错误桶。先用 train 内少量样本固定标注规则，双人独立核验一部分正式样本，保留分歧/裁决。

### 3.2 指标与控制

相对首错位置 $r=t_{err}/T$ 分为 [0,.25)、[.25,.5)、[.5,.75)、[.75,1] 四桶。T 是完整生成回答长度；它自身受生成影响，所以这里不是随机处理变量。

1. 以 calibration 正确回答分数的下 5% 分位数设置信任阈值，低于阈值记为检测到错误；报告实际 calibration/test FPR，不声称测试 FPR 恰为5%。
2. 每桶报告错误数、TPR、95% CI。**只有错误样本不能计算 AUROC**。
3. AUROC 同时提供：与全部最终正确样本的共享对照；与按 log(1+长度) 最近邻、桶内不放回 1:1 匹配的正确对照。匹配只用长度，不看分数。
4. 同一正确题可能被不同桶复用，所有桶使用同一组问题级 bootstrap 抽样权重，避免误把桶当独立样本；匹配在 bootstrap 前固定，CI 条件于该匹配，不含匹配选择不确定性。
5. 主要终点：fDBD mean 的早期桶−晚期桶 TPR 差及双侧区间；同时报告长度匹配 AUROC 差。每桶错误少于20时标为仅描述，不按显著性临时合并桶。

默认评分 fDBD-All/NCI；聚合包括 mean、min、第10百分位数 quantile10、最低10% token 平均、head/tail weighting、lowest-window64。quantile10 与底部10%均值是两个不同统计量。可直接在 CLI 扩展 margin 等，但不得只报告有利组合。

产物：`experiment1.json`、`positions.csv`、`report.md`、`experiment1.png/.svg`。结果不包含未经审核的自动首错标签。

## 四、实验2：无监督聚合与有监督序列模型

### 4.1 对照矩阵

默认分别输入 fDBD-All 和 NCI 的单通道 token 分数序列，避免多通道信息量变化混淆聚合效果。所有监督模型均获得相同的训练集归一化分数和 log(1+轨迹长度) 特征。

| 模型/汇总 | 输入与作用 | 额外监督 |
|---|---|---|
| mean/min/quantile10/token-bottom10/head/tail/lowest-window64 | 无监督公式；分位数与底部均值分列；长度对照另列 | 无 |
| length-only | 仅以回答长度排序的对照 | 无 |
| Logistic Regression | mean/std/min/max/分位数/bottom10/长度等无序统计 | 有 |
| DeepSets | 逐 token MLP 后 masked mean，无位置编码 | 有，学习无序分布 |
| GRU | 单层循环编码，使用最后有效状态 | 有，使用顺序 |
| 1D-CNN | 两层 kernel=3，masked mean，局部感受野为5 | 有，使用局部邻接 |
| Tiny Attention | 一层2头编码器＋固定正弦位置编码＋masked mean | 有，使用位置 |
| Attention-no-position | 相同 Attention 结构，移除位置编码 | 有，容量匹配无序对照 |
| GRU-shuffled | 每次训练将分数顺序打乱，验证/测试也用固定种子打乱 | 有，无自然排列 |

额外对 GRU/CNN/Attention 做固定权重的 test-shuffle 和 test-reverse，打乱的是分数，不是文本。每个训练种子采用一个固定测试置换，三个种子都报告；这是分布破坏诊断，不是内部因果检验。

### 4.2 训练设置与验收

- 小型聚合器使用 FP32；hidden=16，epochs=30，batch=32，AdamW lr=1e-3、weight_decay=1e-4，梯度裁剪1；训练 seeds=0/1/2。原 LLM 仍保持原 BF16 配置。
- BCE 对错误类按 train 类别比例加权；Logistic Regression 使用 balanced class weight、固定 C=1，不在测试集搜索。
- token 和长度归一化均只从 train 拟合。使用长度 mask、GRU packed sequences，防止 padding 引入伪顺序/长度信号。所有模型均是**离线完整轨迹检测**，不能据此称为在线早停算法。
- 神经模型仅根据 validation AUROC 选择 epoch；保留全部预先指定种子，不挑测试最佳种子。LR 为确定性模型，只拟合一次。
- 输出实际参数数目；DeepSets 与 GRU 不声称参数完全相同。Attention vs no-position 是更接近的架构容量控制。
- 保存每个模型 checkpoint、哈希、归一化、最佳 epoch、逐题预测、calibration 阈值及测试 FPR/TPR。完成的模型 job 支持匹配配置下 --resume；中断在某个 epoch 时，该 job 会从头训练，已完成 job 不重训。
- AUROC 主表使用训练种子 AUROC 的平均；CI 是同步重采样 test 问题后，对各种子 AUROC 取平均，不是先集成预测，也不把种子当独立测试题。另报 seed 标准差，CI 不涵盖全部训练随机性。

解释顺序：先检验 GRU 相对 mean 是否提升；再看相对 LR/DeepSets 是否仍提升、打乱训练是否削弱收益，以及有/无位置 Attention 的配对差。如果只赢无监督 mean，不赢无序监督对照，更可能是监督或分布统计收益，不能归因为顺序。

所有报告 CI 为未校正探索性95%区间，不能将多个区间中偶然排除0者写成全局确认。正式方法优越性需要预声明主要比较、控制多重比较并在独立任务/模型复核。本轮即使有增益，也不证明错误传播因果机制。

产物：`experiment2.json`、`report.md`、`experiment2.png/.svg`、每模型 `.pt/.json`、`test_predictions.jsonl`、`manifest.json`。JSON 同时提供相对 mean/LR/DeepSets/Attention-no-position/GRU 的配对区间。

## 五、服务器操作指引

### 5.1 本地同步与服务器准备

本地 PowerShell：

```powershell
cd E:\Research\LLMCoT\methods\OODmethod
git status --short
git add README.md docs ood_repro scripts tests
git commit -m "Implement eval section 2 position and supervised aggregation experiments"
git push origin codex/ood-reproduction
```

服务器：

```bash
conda activate ood-repro
cd ~/projects/OODmethod
git status --short
git switch codex/ood-reproduction
git pull --ff-only origin codex/ood-reproduction
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
mkdir -p /mnt/data/wxh/wdl/logs/OODmethod
```

继续使用原依赖，不升级 torch/transformers。无需额外机器学习包，图表依赖 matplotlib（前一阶段已安装）。示例 GPU 3，执行前确认空闲。下列默认结果根目录为 `/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1`。

### 5.2 立即准备标注，可与训练池生成并行开展

```bash
bash scripts/run_eval2.sh annotations
```

得到 `eval-section2-v1/annotations-template/annotations.jsonl` 和 `tokens.tsv`。复制 JSONL 为根目录下 `annotations-reviewed.jsonl`，按第三部分规则审核；不要改 ID/hash。实验2不依赖此人工标注，可以先运行。

### 5.3 只生成一次 train 数据池，保留在线分数

```bash
bash scripts/run_eval2.sh pool-config

nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
  bash scripts/run_eval2.sh pool \
  >> /mnt/data/wxh/wdl/logs/OODmethod/eval2-pool.log 2>&1 < /dev/null &
echo "PID: $!"
```

```bash
tail -n 30 -f /mnt/data/wxh/wdl/logs/OODmethod/eval2-pool.log
```

原 LLM 需要生成2,400条 train 回答，是本轮主要GPU成本；不要误以为轻量聚合器训练需要重新加载7B模型。pool 完成后 CPU 导入保存的在线分数，不再做第二遍模型回放。原 GSM8K 测试回放完全复用。

`pool-config` 只运行一次；已有配置时不要重复创建。pool 可重新执行续跑，在线导入也可续跑。若已有未保存 token_scores 的旧训练数据，可用 `ood_repro.replay_scores` 单独回放到新的 pool/replay，不能凭最终平均分恢复 token 序列。默认在线数据池只有原几何/概率分数，不含 DeepConf；如需 DeepConf 训练输入，必须做相应额外评分回放。

### 5.4 冻结 train/validation/calibration

```bash
bash scripts/run_eval2.sh split
cat /mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/protocol/splits.json
```

检查 counts 和具体 ID；正式训练前冻结该文件，不反复换 seed。pool 错误不足时程序会提示，需要扩大采样，而不是降低标准凑显著性。

### 5.5 运行实验2：先连通性检查，再正式训练

可选的小 epoch 检查，使用完整 pool 和固定 split，但仅跑一种输入/一个训练种子：

```bash
RESULT_TAG=smoke EPOCHS=2 SEEDS=0 METRICS=fdbd_all FIT_DEVICE=cpu BOOTSTRAP=20 \
  bash scripts/run_eval2.sh experiment2
```

这只检查模型/导出，不是正式成绩，不据此改测试导向参数。正式运行：

```bash
nohup env CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 FIT_DEVICE=cuda \
  bash scripts/run_eval2.sh experiment2 \
  >> /mnt/data/wxh/wdl/logs/OODmethod/eval2-experiment2.log 2>&1 < /dev/null &
echo "PID: $!"
```

默认训练 fDBD-All 和 NCI，全部7类监督对照、3个神经训练种子、30个epoch、2000次bootstrap。仅加载小型分类器，不加载Qwen权重；也可令 FIT_DEVICE=cpu。训练日志逐epoch报告 validation AUROC。

```bash
tail -n 30 -f /mnt/data/wxh/wdl/logs/OODmethod/eval2-experiment2.log
```

输出在 `eval-section2-v1/experiment2/main/`。训练中断可以相同命令续跑；改变模型配置或设备等参数时用新 RESULT_TAG，不混写原运行。不要挑测试最好的模型/种子作唯一结果。

### 5.6 标注完成后运行实验1

```bash
nohup env PYTHONUNBUFFERED=1 \
  bash scripts/run_eval2.sh experiment1 \
  >> /mnt/data/wxh/wdl/logs/OODmethod/eval2-experiment1.log 2>&1 < /dev/null &
echo "PID: $!"
```

默认读取 `eval-section2-v1/annotations-reviewed.jsonl`，也可设置 ANNOTATIONS 指定其他审核文件。此阶段只用CPU，输出在 `eval-section2-v1/experiment1/main/`。未审核完全部最终错误样本时会停止；已审核但无法判断者按明确未知规则记录。

图表/表格重新生成用新 RESULT_TAG。Ctrl+C 退出 tail 不会停止 nohup 任务；停止/更换GPU按 [[OOD实验服务器命令速查]] 的进程核查与 TERM 顺序操作。

## 六、实施与验收边界

代码入口：`eval2.py`（split/annotation/import）、`eval2_data.py`（隔离）、`eval2_positions.py`（实验1）、`eval2_models.py`/`eval2_fit.py`（实验2），统一 Bash 入口 `scripts/run_eval2.sh`。全部输入原结果只读；所有输出放数据盘的新目录。

本地测试覆盖：同题隔离、缺失标注拒绝、明确未知标注、长度匹配、padding 不变性、无序模型置换不变性、所有模型的训练/导出、job 续跑、在线分数审计导入。合成测试只验证实现，不是论文实验数据。

解释结果时至少报告：位置桶错误数/未知率/实际FPR；mean与各聚合的差；监督与无监督的区别；GRU vs 无序监督对照；attention有无位置对照；打乱/反转效果；训练种子差异。任何单一正结果都不等于“独立假设被推翻”。
