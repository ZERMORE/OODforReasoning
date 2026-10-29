# OODmethod：几何幻觉检测独立复现

追加实验：[DeepConf top-20 置信度对照：服务器回放、后台日志与完整主表](docs/DEEPCONF_COMPARISON.md)。

序列实验：[E1/E2 聚合与顺序对照、E3 首错标注、E4 前缀干预的执行步骤](docs/SEQUENCE_STUDY_RUNBOOK.md)。

Eval section 2：[首错位置实验＋GRU/CNN/Attention 与无序监督聚合对照](docs/EVAL_SECTION2.md)。

复现 Liu et al., **From Out-of-Distribution Detection to Hallucination Detection: A Geometric View**, arXiv:2602.07253v4（2026-06-04，ICML 2026）。

- [原论文](https://arxiv.org/abs/2602.07253v4)；公式与实现边界见 [理论说明](docs/theory.md)。
- 本项目是独立实现，不是作者官方代码。已提供主实验执行代码，尚未声称复现论文数值。
- 当前覆盖 Llama-3.2-3B-Instruct、Qwen2.5-7B-Instruct；CSQA、GSM8K、AQuA；NCI、fDBD top-k/All、PPL 的排序等价分数。
- 包含 online / replay、在线重放校验、严格答案抽取、断点续跑、独立训练子集选 k、配对 bootstrap 和在线延迟比较。
- TruthfulQA judge、多样本基线、经验均值消融和步骤级检测是后续工作，不包含在当前主结果实现中。

**等待 Llama 审核时，可先执行 [Qwen 7B 服务器流程](docs/QWEN_SERVER.md)**：`bash scripts/run_qwen.sh prepare` → `smoke` → `main`。入口只运行 Qwen 三任务，模型与数据固定到 commit，结果与缓存默认位于 `/mnt/data/wxh/wdl`。

实验完成后可使用 [结果审计与可视化](docs/RESULTS_ANALYSIS.md)：安装 `requirements-viz.txt`，运行 `python -m ood_repro.visualize --results <结果目录> --output <新的图表目录>`，自动生成中文报告、配对统计及六组 PNG/SVG，无需 GPU。

## 1. 服务器安装

建议 Linux、Python 3.10/3.11、单张 24 GB GPU，batch size 固定为 1。24 GB 是工程预算，实际用量由上下文和评分配置决定。先运行小样本，再扩大任务。默认不量化，不进行模型训练。

```bash
cd OODmethod
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 以下为 CUDA 12.1 示例；按服务器驱动选择 PyTorch 官方支持的 CUDA wheel。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

requirements 固定 Transformers、Datasets、NumPy、PyArrow 等关键版本；Torch 允许 2.3–2.6，以适配服务器 CUDA。它不是跨平台完整 lockfile，实际运行版本写入 manifest。Windows 可用 `.venv/Scripts/python` 运行同样的 Python 命令。

Llama 权重需要先在 Hugging Face 获得访问权限，再通过 `hf auth login` 登录。令牌不要写进配置或 Git。模型下载、数据下载均在服务器执行；本仓库不携带模型权重或评测数据。

## 2. 先做 20 题 smoke test

```bash
python -m ood_repro run \
  --config configs/llama_csqa.json \
  --limit 20 --ks 1000 \
  --verify-replay 3 --save-token-scores \
  --bootstrap 0 --output runs/smoke_llama_csqa
```

JSON 配置提供默认值，显式 CLI 参数优先。`max_new_tokens=512`、默认排除 EOS、chat 格式和不额外使用字符串停止符都是**公开记录的本实现选择**，不是论文已明确披露的设置。

`--verify-replay 3` 会对前三个未完成样本核对在线与重放的 token 位置、token ID、逐 token 分数和序列分数。默认 `--verify-replay-mode matched`：重放整个 prompt 的 prefill，再逐 token teacher forcing，与在线路径保持相同计算形状；默认容差仍为 `atol=0.001, rtol=0.01`，没有放宽。失败时保存 `verification-failure-*.json`（两条轨迹、具体失败指标和位置），且不把失败样本写成成功预测。

`--verify-replay-mode chunked` 可显式检查不同分块形状的独立重放，但 BF16 下不同 GEMM/attention 形状的舍入差异可能超出相同容差；它不再是默认数值一致性门槛。FP32 单元测试仍保留分块重放与独立全序列位置校验，避免同形状校验掩盖对齐错误。

输出目录包含：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 完整配置、数据指纹与 ID、prompt hash、来源、依赖、模型 commit、硬件、EOS、head 校验 |
| `prompt-example.txt` | 首题完整序列化 prompt，便于人工核对 |
| `predictions.jsonl` | 每题 token IDs、回答、标签、分数、长度、停止原因、校验差异 |
| `metrics.json` | 准确率、抽取失败率、截断率、AUROC、配对 bootstrap CI |

样本完成后逐行 flush。`--resume` 仅接受一致配置、数据和 prompt，已完成的题不会重复生成。若进程在写一行的中途被强制终止导致尾行 JSON 损坏，会显式报错；备份后仅移除损坏尾行再续跑，不会自动丢弃数据。修改配置应使用新目录。每题随机种子由全局种子和固定样本顺序确定。

## 3. 主实验与 k 选择

固定论文 k 对照：

```bash
python -m ood_repro run --config configs/llama_csqa.json --resume
bash scripts/run_matrix.sh
```

六个配置均包含论文给定 selected-k 和 All（Qwen GSM8K 为 All）。All 是按词表分块的参考实现，计算较慢；首次检查可用 `--ks 1000` 或 `--ks 100`。不要将 All / 多 k 扫描的时间当成单个 top-k 检测器的延迟。

数据默认来自：

| 任务 | HF 数据 | split | 题数 |
|---|---|---|---:|
| CSQA | `tau/commonsense_qa` | validation | 1221 |
| GSM8K | `openai/gsm8k`, main | test | 1319 |
| AQuA | `deepmind/aqua_rat`, raw | validation | 254 |

支持 `--dataset-revision <commit>`、`--model-revision <commit>`、`--tokenizer-revision <commit>`。正式实验应固定 commit；本地 checkpoint 用 `--model /path/to/model --local-files-only`。离线数据可通过 `--data-jsonl /path/to/rows.jsonl` 指定，行格式需与相应 HF 数据集原始字段一致。manifest 保存指纹，但本地模型权重不做全量 SHA256；请自行维护不可变 checkpoint 和校验清单。

独立训练子集调参示例（512 题与 seed=0 是本实现选择）：

```bash
python -m ood_repro run --config configs/llama_csqa.json \
  --split train --limit 512 --seed 0 \
  --ks 1,10,100,1000,10000,100000,all \
  --bootstrap 0 --output runs/calibration_llama_csqa
python -m ood_repro select-k runs/calibration_llama_csqa \
  --output runs/llama_csqa_k.json
python -m ood_repro run --config configs/llama_csqa.json \
  --k-selection runs/llama_csqa_k.json \
  --output runs/eval_llama_csqa_selected
```

只有完成的 `split=train` 运行能选 k；训练子集在抽样前排除与 few-shot 问题匹配的文本。评测检查任务、模型、生成协议、prompt pack、样本 ID 和归一化问题文本 hash；语义改写重复仍需结合原始数据检查。该调参集只用于选 k，不训练模型。不要在 validation/test 上挑最优 k 后当作独立测试成绩。

## 4. 在线、重放与随机解码

```bash
# 生成后，基于相同 token IDs 做分块 teacher forcing 评分。
python -m ood_repro run --config configs/llama_csqa.json \
  --score-mode replay --replay-chunk 16 --limit 20 \
  --output runs/replay_debug --bootstrap 0

# 对应论文温度实验的一个 cell；分别运行 0.2/0.5/0.8/1.0 和五个 seed。
python -m ood_repro run --config configs/llama_csqa.json \
  --temperature 0.5 --top-p 1.0 --seed 1 --ks all \
  --output runs/llama_csqa_t05_seed1
```

Online 在每次预测时 hook LM head 的实际输入。默认数值校验为同形状 matched replay；独立的 `--score-mode replay` 仍以 `prompt[:-1]` 建立 cache，再分块输入 `[prompt[-1], y1, ..., y(T-1)]`，保证第一个生成 token 不漏计。两者均排除 prompt 与 padding；当前 batch=1，不引入 padding。缓存采用显式 `DynamicCache`。随机解码的 NCI/fDBD 使用原始 logits 的 argmax，PPL 使用实际生成 token。`nci_decoded` 是附录 K 的额外变体。

**只有 online 的耗时包含评分。** replay 输出的 `elapsed_seconds` 只代表生成，不包含重放，记录中有明确 timing_scope。不要将 replay 结果解释为低开销在线检测。

## 5. 指标与统计

所有存储的分数均为“越大越有信心”。AUROC 使用 `is_error=1` 和 `-score`，不做强行翻转或将小于 0.5 的结果改成 1-AUROC。

- `nci`、`fdbd_kN`、`fdbd_all`：论文几何分数。
- `mean_logprob`：生成 token 的平均 log 概率，与负 Perplexity 的排序完全相同，避免指数溢出。
- `log_predictive_probability`：序列 log 概率，避免概率乘积下溢。
- `nci_decoded`：按实际生成 token 权重的 NCI。
- `negative_entropy`、`logit_margin`、`max_probability`：额外诊断基线，其中 `max_probability` 为逐 token 最大概率的均值；**不声称其聚合与论文 Max P 完全一致**。

不单独伪造 LN Predictive Probability：若定义为概率几何平均，它与 PPL 应排序等价，而论文两者报告值不同，其具体实现需进一步确认。

```bash
python -m ood_repro evaluate runs/llama_csqa/predictions.jsonl \
  --bootstrap 2000 --seed 0 --output runs/llama_csqa/metrics_rechecked.json
```

CI 以问题为单位做 percentile bootstrap，对所有检测器使用同一重采样索引；报告相对 mean_logprob 的 AUROC 差及 CI。单类重采样跳过并报告有效次数；单类原始样本的 AUROC 显式未定义。EOS-only 且排除 EOS 的回答会保留为 unscored，报告其数量，不能静默丢弃。不同随机种子分别报告 AUROC，之后再汇总均值与标准差，不把同一问题的五个回答当作独立问题 bootstrap。

严格抽取要求答案标记出现在结尾。数值使用 Decimal 归一化，支持负数、逗号、十进制与科学计数法；不猜测未提供结尾标记的答案。抽取失败按错误回答计入，另报失败率。

## 6. 延迟

```bash
python -m ood_repro benchmark --model meta-llama/Llama-3.2-3B-Instruct \
  --tokens 128 --k 1000 --warmup 2 --repeats 5 \
  --output runs/latency_llama.json
```

先生成固定 token 轨迹，再对 standard、NCI-only、fDBD-only 执行相同的在线 forward/评分；固定 token 数，预热、CUDA 同步、轮换顺序，保存各次 ms/token 与相对增量。包含 prefill、batch=1、同一短 prompt，属于本实现的受控性能检查。论文 Table 4b 使用 CSQA 和 A100；当前 benchmark 不是其完全一致协议。

参考实现包含 Python 循环和退化几何检查的 GPU 同步，不能预期直接达到论文 2.4% 的增量。应先验证数学与检测效果，再优化缓存/融合，并以相同协议重新测量；不能关闭检查后混用旧测试结果。

## 7. Prompt 来源与已知协议差异

`ood_repro/prompts/*.json` 已包含可运行的 7-shot CSQA、8-shot GSM8K、4-shot AQuA，来源是 [Wei et al. 作者公开归档](https://github.com/jasonwei20/chain-of-thought-prompting)。文件记录 ZIP SHA256、归档成员、示例顺序与差异说明；不需要在服务器重新下载该归档。

可重建：下载原作者 `chain-of-thought-zip.zip` 后执行 `python scripts/import_wei_prompts.py --archive /path/to/archive.zip`。导入器只读取每个输入文件第一行的 few-shot 前缀，丢弃评测问题与模型输出，不执行归档内容。

- CSQA 的 fox 示例推理选 natural habitat，但源归档结尾为 (b)，而该选项是 (c)；保留并标注。
- AQuA 源归档用 `So the answer is`，Liu 附录要求 `The answer is`；示例保留来源内容，system prompt 采用 Liu 附录规则。
- 当前 chat 序列化把 few-shot 与测试问题放进一个 user message，system prompt 独立。Liu 未公布完整序列化，不能声称逐字复原。可复制 JSON pack，用 `--prompt-file` 建立显式修订对照；不同 pack 必须用新输出目录。

论文 §4.1 CSQA 题数与 §5.1 不一致，取主实验的 1221；调参集也有 training subset/validation 表述差异，当前强制训练子集调参。完整协议表见 [protocol.md](docs/protocol.md)。

## 8. 主结果参考与代码结构

数值是论文 Table 4 的 **AUROC × 100**，不是本项目实测：

| 模型 | 任务 | PPL | NCI | fDBD All | selected-k | k |
|---|---|---:|---:|---:|---:|---|
| Llama 3B | CSQA | 63.23 | 66.07 | 68.15 | 69.24 | 1000 |
| Llama 3B | GSM8K | 69.63 | 76.32 | 75.59 | 76.36 | 100 |
| Llama 3B | AQuA | 72.85 | 74.41 | 75.80 | 76.20 | 100 |
| Qwen 7B | CSQA | 61.94 | 71.60 | 71.50 | 72.47 | 100 |
| Qwen 7B | GSM8K | 71.54 | 75.83 | 77.19 | 77.19 | All |
| Qwen 7B | AQuA | 71.66 | 78.19 | 77.08 | 78.22 | 100 |

```text
ood_repro/scoring.py      # NCI/fDBD、解析中心、分块候选距离
ood_repro/engine.py       # KV-cache 在线生成、分块重放、head 校验
ood_repro/data.py         # 三数据集、严格答案抽取
ood_repro/prompting.py    # prompt packs 与显式序列化
ood_repro/evaluation.py   # AUROC、配对 bootstrap、独立选 k
ood_repro/cli.py          # run/evaluate/select-k/benchmark
configs/                 # 六个主实验 cell
tests/                   # 几何解析测试、真实 tiny 模型及 CLI 集成
docs/                    # 理论与协议边界
```

Git 忽略 `.venv/`、缓存、模型、数据、运行结果及 ZIP，跨服务器同步时只需代码与配置。运行结果应单独备份，不要仅依赖 Git。代码本身不启动下载完整模型的实验。

实验1可使用 `deepseek-v4-pro` API 自动定位首次错误，复用已有轨迹，无需 GPU。支持精确原文到 token 映射、断点续跑、原始响应与来源记录；结果明确标注为自动判定。服务器操作见 [DeepSeek 自动首次错误定位](docs/experiment1-deepseek.md)。

## 9. 本地验证记录

2026-10-22：**25 项测试通过**。环境为 Windows、Python 3.11、PyTorch 2.3.0+cu121、Transformers 4.46.3；包括随机 tiny Llama/Qwen2 的 CPU 对齐测试、RTX 3060 Laptop GPU 上的 BF16/SDPA 对齐测试，以及本地 checkpoint 的 CLI/续跑/benchmark 测试。

完整 3B/7B 模型、三个完整数据集、A100 延迟和论文 AUROC 尚未验证。Windows 沙箱的系统临时目录可能无法由 Safetensors 原生扩展写入；本次将 pytest 临时目录显式设在工作区 `.cache/pytest-release`，标准 Linux 可直接使用 `python -m pytest -q`。
