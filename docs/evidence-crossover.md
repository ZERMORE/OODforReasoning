# Evidence Drop × OOD 实验方案

比较三个逐 token 信号（top-20 负 Shannon 熵、fDBD All、NCI）与五种聚合（Mean、Evidence Drop、最低 10% token 均值、最低 64-token 窗口均值、GRU）。GSM8K 和 Hendrycks MATH 各有 15 个主表组合。每个数据集的组合共享生成轨迹、答案标签和 token 边界。

| 数据 | 测试轨迹 | GRU 与阈值的数据 |
|---|---:|---|
| GSM8K | 复用 1,319 条 Qwen2.5-7B-Instruct 轨迹 | 复用 2,400 条原训练池轨迹，冻结 1,680 / 360 / 360 划分 |
| MATH | 七个题型完整 test，计划生成 5,000 条 | 从原始 train 固定种子取 2,400 题，生成后按题目组划分 70% / 15% / 15% |

每题生成一次。GSM8K 回放补算负熵并审计旧分数。MATH 使用同一 Qwen2.5 checkpoint，zero-shot、greedy、上限 2,048 token。生成时同时保存三个信号。题目自带参考答案，轨迹标签来自最终答案比较。GSM8K 沿用既有判分。MATH 抽取最后一个盒中答案，先精确比较，再用 Math-Verify 检查符号等价。抽取失败与无法判分按错答计，另报原因和截断数。正式 Linux 判分每次解析或比较的超时为 5 秒。已有 GSM8K 参考标签争议应在结果解释中保留。

## 分数与主要比较

- Shannon 在原始 logits 的 top-20 内做 softmax，计算 $\sum p\log p$。Drop 对置信度做 EMA（span 5），求相邻下降的严重度，取最大五次的平均。少于五次按实际下降数计，无下降风险为零。保存下降的右端 token。公开 `LN_E_drop` 先做累计平均，作为另一种实现记录。主实验采用论文 Eq. 9–12 的原始证据序列。[论文](https://proceedings.mlr.press/v306/chen26n.html) · [公开实现](https://github.com/QJ0114/evidence-drop/blob/main/uncertainty_estimation/utils/metrics.py)
- GRU 对每种信号分别训练。复用已有单层 GRU 和长度特征，hidden 16、AdamW、30 epoch、三个种子，以验证 AUROC 选 checkpoint，标准化只拟合训练数据。汇总各 seed 的 AUROC，置信区间按测试题配对重采样。单独保存 seed 波动。
- AUROC 为主，答错为正类。AP 与 AURC 补充。比较同信号 Drop 对 Mean、固定聚合下 Shannon 对几何信号，以及两种信号的 Drop 增益差。附加对照包括长度基线、长度分层、Drop 和 GRU 的测试乱序，以及 GRU 反序。乱序检验分数对顺序的依赖。
- 固定轨迹的生成 accuracy 是数据背景。拒答实验报告 selective accuracy、coverage、错答接受率与正确答案误拒率。阈值只从校准集取得。采用错答风险低尾 cutoff，边界并列一律拒绝。论文文字的 $(1-\alpha)$ 分位数与高风险拒绝约定存在方向歧义，代码记录实际规则与实测比例。经验分位数只描述本次校准，额外的有限样本保证需另行实现。

## 运行顺序

在服务器项目目录同步代码并激活 `ood-repro`。配置已有 Hugging Face 缓存路径；MATH 另需判分依赖。

```bash
python -m pip install -r requirements-viz.txt -r requirements-math.txt

# 复用原始 GSM8K 测试和训练池。源目录可用 GSM_TEST_RAW、GSM_POOL_RAW、GSM_SPLIT 覆盖。
bash scripts/run_evidence.sh replay gsm8k
bash scripts/run_evidence.sh split gsm8k
bash scripts/run_evidence.sh fit gsm8k
```

下方使用已核对的 MATH revision。导出保留全部 5,000 道测试题，原训练集排除 1 道与测试集重叠的题和 2 道参考答案盒为空的题。训练池从剩余 7,497 题取样。导出记录排除原因和文件哈希。指定同一份 Qwen2.5 checkpoint 和 tokenizer revision。 [MATH 数据集仓库](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3)

```bash
export MATH_REVISION=21a5633873b6a120296cce3e2df9d5550074f4a3
export MATH_DATA=/mnt/data/wxh/wdl/datasets/hendrycks-math-v1
export MODEL_PATH=/mnt/data/wxh/wdl/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28
export MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28
python -m ood_repro.prepare_math --revision "$MATH_REVISION" --output "$MATH_DATA"

# 检查格式、判分和评分链路。正式生成使用独立的默认目录。
EVIDENCE_ROOT=/mnt/data/wxh/wdl/results/OODmethod/evidence-smoke \
  TEST_LIMIT=20 POOL_LIMIT=20 MAX_TOKENS=256 bash scripts/run_evidence.sh generate math
bash scripts/run_evidence.sh generate math
bash scripts/run_evidence.sh split math
bash scripts/run_evidence.sh fit math
```

GPU 阶段为 `replay` 和 `generate`。长任务可用 `nohup` 和独立日志。`FIT_DEVICE=cuda` 可加速长序列 GRU。每个输出目录同一时间只运行一个进程。继续运行时保留相同配置、代码和结果目录。默认结果根目录 `/mnt/data/wxh/wdl/results/OODmethod/evidence-crossover-v1`。其中 `<task>/fit/` 保存 AUROC 与配对区间、AP、AURC、逐题分数、阈值结果、ECDF 和风险—覆盖率图。

## 论文对照

另用 Qwen3-8B 跑 greedy 的 GSM8K 与 MATH。指定 checkpoint、数据 revision、prompt 和生成上限，保留偏离论文的配置。用 `python -m ood_repro run --shannon --save-token-scores` 生成 test 输出，再用 `python -m ood_repro.eval2 import-online --test <raw> --output <replay>` 转换。MATH 传入准备好的 `--data-jsonl`。最后运行下列命令，按固定种子折半校准和评估，对比 Shannon Mean 与 Drop 的 selective accuracy、coverage 和 AURC。

```bash
python -m ood_repro.evidence_paper --replay "$PAPER_REPLAY" --output "$PAPER_OUTPUT"
```

论文对照使用明确记录的经验阈值规则。作者的精确测试划分与生成细节若无法取得，将结果标为复现协议下的对照值。
