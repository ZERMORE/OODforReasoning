# DeepConf top-20 置信度对照

来源：[Deep Think with Confidence](https://arxiv.org/abs/2508.15260)，式 (2)–(6)。

本扩展评价单条已生成回答的错误检测能力，不实现多轨迹置信度投票、warmup、共识阈值或提前终止；因此表中标为单次生成。不能据此声称复现了完整 DeepConf 的推理准确率或 token 节省。

## 固定的评分协议

- 每步从**全词表 softmax** 得到概率，取 top-20：`C_t = -mean(log p_top20)`。不在 top-20 内重新归一化，也不改变生成采样参数。
- 保存整段平均 `deepconf_mean`、最低窗口均值 `deepconf_lowest`、最低 10% 窗口均值的平均 `deepconf_bottom10`。三者越高越可信；主要局部评分对照采用 bottom-10%。
- 预先固定窗口 64、步长 1；只使用完整窗口，回答不足 64 token 时使用一个覆盖全回答的窗口。bottom-10% 的窗口数向上取整，至少一个。EOS 是否计分继承原实验。
- 原论文主要使用 1024/2048 窗口；本项目现有回答上限 512，直接用长窗口会让大多数回答退化为全段平均，因此 64 是**适配当前短回答的设置**，不是原论文窗口复现。不要基于测试 AUROC 选择窗口。可预先声明 32/64/128 的敏感性实验，分别输出到不同目录。
- top-k 的负平均 log 概率不是 Shannon entropy，也不是正确概率；“越大越确定”是原方法的评分方向，不能保证对所有概率分布严格成立。

## 为什么需要服务器补算

已有主实验保存了 prompt/response token ID，没有保存 top-20 log 概率，不能用现有 PPL、margin 或 NCI 反推出该指标。
通过 teacher-forced matched replay 读取**相同模型、相同原回答**的 logits；不重新生成、不改变答案或正确性标签。逐样本核对计分 token 数以及原 mean_logprob（沿用原数值容差）。模型与 tokenizer revision、dtype、attention 从原 manifest 继承，并检查模型/tokenizer commit、chat template 与关键软件包版本。

新结果保存在独立目录，原评分和标签完整保留。新增 token 置信度、AUROC、2000 次问题级 bootstrap CI，以及相对 PPL、NCI、logit margin、fDBD 的配对差值 CI。每个完成样本立即写入，`--resume` 可续跑；中断恰好写坏最后一行时会明确报错，需要先检查并修复该行。

## 服务器执行

先同步本地代码到服务器；继续使用原 `ood-repro` conda 环境和原 checkpoint，无需新增依赖。

```bash
conda activate ood-repro
cd ~/projects/OODmethod
mkdir -p /mnt/data/wxh/wdl/logs/OODmethod
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
  bash scripts/run_deepconf.sh \
  > /mnt/data/wxh/wdl/logs/OODmethod/deepconf-w64.log 2>&1 < /dev/null &
echo $!
```

```bash
tail -f /mnt/data/wxh/wdl/logs/OODmethod/deepconf-w64.log
```

如果原实验目录不同，使用环境变量覆盖 `SOURCE_ROOT`；`DEEPCONF_ROOT` 指定新目录。改变窗口必须同时换结果目录，例如：

```bash
WINDOW=128 DEEPCONF_ROOT=/mnt/data/wxh/wdl/results/OODmethod/qwen25-deepconf-w128 \
  CUDA_VISIBLE_DEVICES=2 bash scripts/run_deepconf.sh
```

如 checkpoint 已移动，可在脚本命令末尾追加 `--model /新的/checkpoint/目录`，仍需是同一 revision。脚本只回放轨迹，不下载数据集。

## 更新完整主表

三项任务全部完成后，在服务器安装已有可视化依赖（若尚未安装），然后运行：

```bash
pip install -r requirements-viz.txt
python -m ood_repro.main_table \
  --results /mnt/data/wxh/wdl/results/OODmethod/qwen25-deepconf-w64 \
  --output /mnt/data/wxh/wdl/results/OODmethod/tables-deepconf-w64
```

输出 PNG/SVG/CSV/Markdown/LaTeX，以及带 95% CI 的主表。新增三个 DeepConf 汇总行，并注明窗口设置。每次导出使用新空目录。补算前不填入任何估计结果。

重点查看 `deepconf_vs_logit_margin.json` 和 `deepconf_vs_fdbd_*.json` 中 `paired_delta_vs_reference`：差值是 DeepConf AUROC 减对照 AUROC，区间包含 0 时不能确认优劣。
