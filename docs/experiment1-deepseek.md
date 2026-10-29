# 实验1：DeepSeek 自动首次错误定位

本流程复用已完成的 GSM8K 轨迹和 calibration 划分，无需重新运行 Qwen、无需 GPU。API 判定为自动标注，不是人工真值。默认官方端点 `https://api.deepseek.com`，模型 `deepseek-v4-pro`，thinking enabled，JSON 输出。接口依据：[Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)、[Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)。

本地验证（2026-10-29）：全套 79 项测试通过；真实 GSM8K 的 276 条选定轨迹通过离线精确位置映射预检。API 调用在测试中使用模拟响应，尚未使用真实 key 发起付费请求。

## 实验做什么

要检验的是：平均置信度的检测能力是否与首次错误出现的位置有关，早期错误是否更容易漏检。平均聚合不要求 token 独立；这里研究其可能丢失的局部与顺序信息。结果只能支持位置关联，不能独立证明错误传播或因果关系。

1. 固定抽取全部 138 条最终答错轨迹，加随机抽取的 138 条最终答对轨迹，打乱次序。选择 seed=1729。正确轨迹也可能存在已修正的过程错误。
2. 每条轨迹独立请求 DeepSeek。输入只有题目、参考答案、候选全文和机械切分的句段；不提供 NCI/fDBD 分数、最终正误标签或位置分桶。
3. 判定过程是否正确，返回首次错误所在句段和最小原文错误片段，包括后来修正的错误。参考答案冲突本身不构成过程错误证据。
4. 本地严格校验原文片段，再通过 Qwen byte-BPE 的字节区间映射为原始、从 0 起算的生成 token 位置。不能直接把 Unicode 字符位置当作 token 位置。
5. 以错误断言片段的起点作为操作性位置；这是模型选择片段的边界，不保证就是错误的真实因果起点。句段是机械切分，不等于经验证的推理步骤。`first_error_span_end` 是右开 token 边界，`char_start/char_end` 是 Python 字符区间。
6. 对不能判断或不能定位的案例保留空位置，统计未定位数量，不强行归桶。传输失败、JSON 不合法、原文片段不匹配不会被伪装成“未知”：重试耗尽后退出，待恢复续跑。
7. 复用实验1分析：四个相对位置桶；NCI/fDBD × mean/min/quantile10/token_bottom10/head_weighted/tail_weighted/lowest_w64。输出 calibration 阈值下 TPR、实际 test FPR、共享正确对照和长度匹配对照的 AUROC、问题级 bootstrap 区间、早期减晚期差值。

位置分母沿用已有分析协议的 `len(generated_token_ids)`，含存储的终止 token（若存在）；错误位置必须落在已评分 token 上。桶 0/1/2/3 对应 [0%,25%)/[25%,50%)/[50%,75%)/[75%,100%]。

正确对照仍按最终答案标签定义，不能称为全部推理无误的真值对照。自动判定的过程错误和恢复信息另行保留。

## 同步与服务器准备

先把本地修改提交并推送到当前 GitHub 分支，再在服务器执行：

```bash
cd ~/projects/OODmethod
git pull --ff-only origin codex/ood-reproduction
conda activate ood-repro
python -m pip install -r requirements-viz.txt
mkdir -p /mnt/data/wxh/wdl/logs/OODmethod
```

API 客户端使用 Python 标准库，不需要安装 OpenAI SDK，不改现有模型环境。可先执行相关测试：

```bash
python -m pytest tests/test_auto_positions.py tests/test_eval2.py -q
```

如果轨迹不在默认的 `sequence-study-v1/replay/qwen_gsm8k`，运行时显式设置 `TEST_REPLAY=/实际/replay/路径`。所有步骤必须使用同一份 test replay，不能混用旧轨迹。

## 先预检，再配置 key

```bash
bash scripts/run_eval2.sh auto-annotations --dry-run
```

预期：`Selected 276 (138 final errors + 138 correct controls)` 和 `Dry run OK`。预检不联网调用 API、不收费。默认标注输出为：

```text
/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/annotations-deepseek-v4-pro
```

默认从仓库的 `configs/deepseek.local.json` 读取 key。该文件已加入 Git 忽略规则，本地已创建空文件。填入：

```json
{
  "api_key": "你的DeepSeek API key"
}
```

代码通过 Git 同步时不会上传这个文件，须单独传到服务器同一路径；也可直接在服务器创建并编辑：

```bash
cp -n configs/deepseek.example.json configs/deepseek.local.json
chmod 600 configs/deepseek.local.json
nano configs/deepseek.local.json
```

保存后直接执行试跑/后台命令即可，不再需要每次输入 key。`nano` 用 `Ctrl+O`、回车保存，`Ctrl+X` 退出。程序读取 JSON，不执行文件中的任何 shell 命令。

环境变量 `DEEPSEEK_API_KEY` 优先于文件；如需确保使用文件，先执行 `unset DEEPSEEK_API_KEY`。自定义文件可传 `bash scripts/run_eval2.sh auto-annotations --api-key-file /绝对路径/key.json`。不要将带 key 的文件提交到 Git。

也可选择只使用当前会话环境变量：

```bash
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

不能只设置 Hugging Face token；这里需要 DeepSeek 平台 API key。采用文件方式时新 SSH 会话会自动读取文件；采用环境变量方式时需要重新输入。已经由 nohup 启动的子进程会继承 key，断开 SSH 后仍可运行。密钥不进入实验 manifest、请求示例或日志。

## 试跑 2 条

```bash
bash scripts/run_eval2.sh auto-annotations --limit 2
```

查看这两条的判定与费用统计。下面只读取本地 JSON，不会再调用 API：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/annotations-deepseek-v4-pro')
m = json.loads((p / 'judge-manifest.json').read_text())
print({k: m[k] for k in ('complete', 'completed_n', 'selected_n', 'usage')})
for line in (p / 'annotations.jsonl').read_text().splitlines():
    r = json.loads(line)
    print({k: r[k] for k in ('sample_id', 'process_valid', 'segment_id', 'error_quote', 'first_error_position', 'notes')})
PY
```

此时 `complete=false` 正常，因为只完成了 2/276。可以自行检查原始结果，但不要求手工标注。程序校验只保证格式和定位一致，不保证 DeepSeek 的数学判断正确。

## 后台完成全部标注

确保前一次前台进程已结束；同一输出目录只运行一个标注进程。

```bash
LOG=/mnt/data/wxh/wdl/logs/OODmethod/eval1-deepseek-$(date +%Y%m%d-%H%M%S).log
nohup env PYTHONUNBUFFERED=1 \
  bash scripts/run_eval2.sh auto-annotations \
  > "$LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: $LOG"
tail -n 20 -f "$LOG"
```

程序跳过已有成功结果，试跑的两条不会重复计费。`Ctrl+C` 只停止 tail 查看；nohup 任务继续。中断或网络错误后，用同一命令重启即可。极少数 API 已处理请求但本地尚未保存时的中断，仍可能造成重复计费。

默认顺序请求，每条最多 3 次重试（合计最多 4 次尝试），指数退避；401/402/403 等账户、权限错误立即停止，避免重复请求。总耗时与费用取决于 DeepSeek 响应、思考 token 数和重试；没有固定耗时或费用承诺。`usage` 汇总成功接受的调用，重试原始响应中的用量也可能计费。没有把 key 放进请求文件、结果或日志。

若持续返回长度耗尽，可使用新输出目录提高上限，例如：

```bash
JUDGE_OUTPUT=/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/annotations-deepseek-v4-pro-16k \
JUDGE_MAX_TOKENS=16384 bash scripts/run_eval2.sh auto-annotations --limit 2
```

这会创建新的标注协议，重新请求相应样本；不要混合不同设置的结果。模型、prompt、源文件、种子或输出上限变化时，续跑校验会拒绝混用。代码本身变化也要求新目录。

## 完成后运行实验1分析

日志应显示 `complete=True`；也可以查看 `judge-manifest.json` 的 `complete`。只有全部选定案例完成后才能执行分析：

```bash
LOG=/mnt/data/wxh/wdl/logs/OODmethod/eval1-analysis-$(date +%Y%m%d-%H%M%S).log
nohup env PYTHONUNBUFFERED=1 \
  ANNOTATIONS=/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/annotations-deepseek-v4-pro/annotations.jsonl \
  RESULT_TAG=deepseek-v4-pro BOOTSTRAP=2000 \
  bash scripts/run_eval2.sh experiment1 \
  > "$LOG" 2>&1 < /dev/null &
echo "PID: $!"
echo "Log: $LOG"
tail -n 20 -f "$LOG"
```

若标注使用了自定义目录，要修改 `ANNOTATIONS`。分析输出目录必须为空；重新分析时使用新的 `RESULT_TAG`，不要删除原结果来覆盖。

结果位于：

```text
/mnt/data/wxh/wdl/results/OODmethod/eval-section2-v1/experiment1/deepseek-v4-pro/
  experiment1.json      # 完整统计、阈值、FPR、未知数量、标注来源
  positions.csv         # 各 metric/aggregation/位置桶结果
  report.md             # 可读数据表与限制
  experiment1.png/.svg  # mean 的位置 TPR / 长度匹配 AUROC 图
```

标注目录中 `records/` 保存逐样本成功结果，`raw/` 保存 API 原始返回，`annotations.jsonl` 是分析输入，`judge-manifest.json` 保存模型、协议、源文件哈希、完成度和用量。`reviewed=true` 仅为兼容既有标注格式，自动结果明确带 `annotation_kind=llm_auto`、`human_reviewed=false`。不要只复制 annotations 文件而丢掉 manifest；分析会拒绝缺少 provenance 或未完成的自动标注。

将标注目录和 experiment1 目录整体同步回本地后，即可继续分析。

## 如何解读

重点看未定位比例、每桶错误数、mean 的 early-minus-late 差值及区间，再看替代聚合是否改善。若早期检测显著更弱，支持“位置相关的检测不足”；若没有差异，应如实报告。少于 20 条的桶只做描述。

之前实验2已显示 calibration 阈值迁移存在偏移；这里 TPR 必须同时看实际 test FPR，不能标成 test TPR@5%FPR。长度匹配 AUROC 是补充证据。95% CI 不含自动 judge 的误判、边界选择或 API 非确定性；显著位置差异也可能来自题目难度、错误类型和答案阶段。后续若需要更强证据，可增加第二轮独立 judge 和位置边界敏感性分析，并在新协议下报告一致率，不能把本次结果直接称作人工验证的错误传播证据。
