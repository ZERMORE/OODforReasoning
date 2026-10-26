# 已保存结果的审计、统计与可视化

入口：`python -m ood_repro.visualize`。CPU 即可，不下载模型，不启动实验，也不修改输入结果。

## 安装与运行

在已有 ood-repro 环境安装可视化依赖：

```bash
python -m pip install -r requirements-viz.txt
```

当前 Windows 数据：

```powershell
cd E:\Research\LLMCoT\methods\OODmethod
python -m ood_repro.visualize `
  --results E:\Research\LLMCoT\methods\OODResults `
  --output analysis_outputs\qwen25-main `
  --bootstrap 2000 --seed 0
```

服务器同样可运行：

```bash
python -m ood_repro.visualize \
  --results /mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4-matched \
  --output /mnt/data/wxh/wdl/results/OODmethod/qwen25-analysis \
  --bootstrap 2000 --seed 0
```

`--results` 可指定 OODResults、单个实验目录或 main 目录；同时找到多个实验则拒绝，须显式指定。输出目录必须与输入 main 分开且为空，避免覆盖历史分析。重跑使用新的输出目录。

## 输出

- `report.md`：中文审计、论文对照、配对比较、格式/截断敏感性与研究边界。
- `analysis.json`：完整数值、配对 CI、配置、环境、输入文件 SHA256 与长度关联。
- `paper_comparison.csv`：论文与复现的主表数值。
- `failure_cases.jsonl`：全部抽取失败/截断案例；标签与原结果保持一致。
- 六组 PNG + 可编辑 SVG：论文对照 CI、相对 PPL 的配对增益、ROC、正确/错误分布、风险覆盖、生成长度。

图表使用英文标签，避免跨平台中文字体缺失；报告为中文。置信分布对每个正确性类别分别归一化，不混淆类别频数。风险覆盖曲线在同分时只按完整阈值组保留，避免题目顺序制造优势；同时提供完整范围和 coverage≥20% 的明确放大视图，不裁掉极低覆盖率下的高风险。

## 论文式完整主表

```bash
python -m ood_repro.main_table --results <结果根目录> --output <新的表格目录>
```

输出 `main_results.png/svg/md/csv/tex/json`，以及含95% CI 的 `main_results_with_ci.png/svg/md`。主表覆盖全部实际运行评分；灰色突出几何方法，粗体标最高点值，下划线标第二个不同点值，等值并列（按未舍入值排名）。selected-k 行按论文的 100/All/100 复用已测结果；NCI decoded 在 greedy 下复用同分，不是独立实验。GSM8K 的 k100 未运行，保留缺失。导出前会重新核验全部诊断分数的 AUROC。LaTeX 片段需要 booktabs 与 multirow 宏包。排名不代表配对统计显著性。

## 方法与约束

1. 核验三个官方 split 题数、唯一问题 ID、输入内容 hash、严格答案抽取、标签、评分有限性、保存的准确率/失败率及 AUROC。核验同一实现 hash、recipe 模型和数据 revision、已执行的重放检查。
2. 问题级 paired percentile bootstrap，对所有方法共享相同重采样索引，AUROC 始终以错误为正类并使用负置信分数。使用分数并列组与重采样 multiplicity 的秩公式加速；独立测试对照 sklearn 的实际重采样 AUROC。
3. 相同 seed / draw 数时核对新计算的 CI 与服务器保存值。提供 selected fDBD 相对 PPL、NCI、logit margin、All 的配对差值 CI。
4. 置信区间反映当前问题样本的不确定性，不覆盖模型、prompt、解码设置的变化。探索性比较未做多重校正。
5. 不对本次与论文做配对检验：缺少论文逐题预测和 CI。只报告点差及本次自身 CI。
6. 排除格式失败/截断后的子集仅作敏感性分析，不能替换完整 split 主结果或证明因果关系。
7. 风险覆盖曲线在本次评测集上描述，不是独立校准后的部署风险保证。

## 本次结果的初步评价（2026-10-23）

三组共有 2,794 题，均完整评分，环境为 Qwen2.5-7B-Instruct、BF16、greedy、512-token 上限，服务器实现 commit 556934c。重新计算的点指标与 2000 次 bootstrap CI 与服务器保存值一致（CI 最大浮点差约 1.1×10⁻¹⁶）。

- CSQA：selected fDBD 71.02（论文72.47），比 PPL +11.47 点，配对 CI [+8.20,+14.87]。
- GSM8K：selected fDBD / All 77.93（论文77.19），比 PPL +4.85 点，CI [+0.78,+8.97]；NCI 77.97，logit margin 78.75，尚不能证明 fDBD 优于这些廉价分数。
- AQuA：selected fDBD 71.67（论文78.22），比 PPL +2.45 点，CI [−2.55,+7.52]；有7个抽取失败，其中6个截断。排除这7题的描述性分数为74.73，仍未达到论文点值，且不能替代完整集成绩。

因此应描述为：**CSQA/GSM8K 复现主要趋势，逐项数值未完全复现；AQuA 的差距需要优先检查生成协议，几何分数超越 margin 的独立价值仍需验证。** 不应把本次结果称为“所有任务都成功复现”或“几何分数总是最优”。

## DeepConf 追加对照

新增 top-20 token confidence 的离线回放脚本及主表支持。补算、日志与导出命令见 [DEEPCONF_COMPARISON.md](DEEPCONF_COMPARISON.md)。原结果不包含 top-20 log 概率，需要服务器读取相同 checkpoint 对既有 token 序列补算。
