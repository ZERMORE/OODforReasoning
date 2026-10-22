# 工程协议与验收

## 论文设置与本实现选择

| 项目 | 原文依据 | 当前实现/处理 |
|---|---|---|
| 论文版本 | v4，2026-06-04 | 固定引用版本，不与早期版本混用 |
| 模型与 split | §5.1 | 两模型、三数据集，官方评测 split 大小强校验 |
| few-shot | 附录 G.2，Wei canonical examples | 原作者公开归档的 7/8/4 示例；hash 和已知差异入 manifest |
| system prompt | 附录 G.3 | 按任务保存明确字符串，普通 ASCII 引号 |
| chat 序列化 | 原文不完整披露 | 独立 system + 单 user 包含 few-shot 与问题；支持 plain 对照 |
| max_new_tokens | 未明确 | 默认 512，可覆盖；报告截断率 |
| 结束 token | 未完整披露 | 使用模型 generation_config EOS 列表，缺失则 tokenizer EOS |
| EOS 是否计分 | 未明确 | 默认排除，可 --include-eos；两分支分开输出 |
| 字符串停止符 | 未完整披露 | 不额外引入；避免隐藏截断规则 |
| 随机解码 | §4.4，四温度、五 seed | raw logits 评分，采样支持温度/top-p；默认 top-p=1 |
| 选 k | §5.1 training subset，与部分 validation 表述不一致 | 强制 train 子集选 k；保存 ID；CLI 检查与评测协议及 ID 不重叠 |
| 标签 | 附录 G.2 | 严格末尾标记提取，失败计错并单报；Decimal 比较数值 |
| PPL | §4.1 | 平均 logprob，排序等价于负 PPL；不指数化 |
| Max P / LN PP | §5.1 细节不足 | mean max-prob 仅诊断；LN PP 暂不声称复刻 |
| 延迟 | Table 4b，A100、CSQA | 另有固定轨迹受控 benchmark；不是完全相同 prompt/采样集合 |

## 阶段验收

1. 数学：解析三类几何、偏置中心一阶条件、k=V−1 对照、退化范数、tie、方向性；全部测试通过。
2. 引擎：随机 tiny Llama 与 Qwen2，在线、分块重放、单次全序列独立位置校验一致；hook 清理、EOS、随机种子测试通过。
3. 端到端：本地 tiny checkpoint、tokenizer、JSONL 数据，不联网执行 run/evaluate/resume；配置覆盖和不一致续跑拒绝测试通过。
4. 服务器小样本：20–50 题，无 NaN/inf；审查 prompt、长度、抽取失败与 EOS；BF16 容差经检查；此阶段不看小样本 AUROC 是否贴近论文。
5. Llama+CSQA 全集：对照 PPL 63.23、NCI 66.07、fDBD All 68.15、k1000 69.24（AUROC×100）；以结果偏差、配对 CI 和协议差异报告，不预设 ±1 点就是成功。
6. 六 cell 主结果：每个 cell 保留准确率、parse/truncation/unscored 率和 CI。固定论文 k 与独立调参结果分开。单次主表不以多个 seed 中最好一次替代。
7. 消融：k 扫描、四温度×五 seed、EOS/prompt/长度敏感性；必要时再实现经验均值和多样本基线。
8. 性能：同 prompt、同轨迹、同 token 数、同 batch 和 dtype，预热+同步；记录 hardware/kernel/check overhead；比较各检测器专用路径。

## 当前实现限制

- batch=1、单设备 dense head；没有分布式、vLLM、量化或 MoE 扩展。
- All 分块算法额外计算仍为 O(Vd)，不是论文低增量的优化实现。
- GPU 退化检查会同步；优化需重新验证并单独标记。
- 未实现 TruthfulQA judge、SelfCheckGPT、Semantic Entropy、CoE 等完整 Table 4 基线。
- 非零偏置解析中心为 float64 O(d²) 内存 / O(d³) 解法，主模型零偏置直接返回零；低配 GPU 不应拿任意带偏置大 head 做默认主实验。
- manifest 不含本地权重完整 SHA256。训练子集在抽样前排除与 few-shot 问题文字匹配的样本；评测检查与校准问题的归一化文本 hash 不重叠。大小写/空白归一化无法识别语义改写，仍需按来源审核。
- 实测与论文数值差异应先检查生成行为，再检查检测器；小幅领先须配对统计，不能自动称显著。
