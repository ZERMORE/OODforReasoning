# G1 扩展实验服务器操作指引

四种方法共用一次逐 token 前向。CSQA 和 AQuA 回放已有输出，SuperGPQA 生成并在线打分，ProcessBench 回放公开解答。窗口扫宽、融合与评估都在 CPU 上完成。模型、数据、日志和结果使用原数据盘。

## 1. 同步并激活环境

本地提交推送到 GitHub 后，在服务器执行以下命令。现有分支为 codex/ood-reproduction。

```bash
cd ~/projects/OODmethod
git fetch origin
git switch codex/ood-reproduction
git pull --ff-only origin codex/ood-reproduction
conda activate ood-repro
export DATA_ROOT=/mnt/data/wxh/wdl
export LOG_DIR="$DATA_ROOT/logs/OODmethod/geom-extension-v1"
mkdir -p "$LOG_DIR"

python -c 'import torch, sklearn, huggingface_hub; print(torch.__version__, sklearn.__version__, torch.cuda.is_available())'
bash -n scripts/run_geom_extension.sh
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv
```

可选安装 matplotlib 后生成扫窗图。缺少 matplotlib 仍会生成全部 CSV。

```bash
python -m pip install matplotlib
```

如果前一轮 G1 的结果在 geom-diagnostics-v1 下，执行下面的覆盖，再运行冻结。路径需同时有 gsm8k/manifest.json、math/manifest.json 和各自的 rows/ 逐题回放文件。

```bash
export G1_SOURCE_ROOT="$DATA_ROOT/results/OODmethod/geom-diagnostics-v1"
```

默认查找 geom-diagnostics-v1-fixed。已有 CSQA、AQuA 默认从 qwen25-paper-v4-matched/main 读取。如原模型快照挂载路径不同，可设 MODEL_PATH 指向同一个模型 revision 的目录。

## 2. 准备数据、冻结策略

只有数据准备需要联网。先运行 CSQA/AQuA 队列也可以，这两个任务复用已有数据。数据准备保存 HF 提交 SHA 和本地 SHA-256。

```bash
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE bash scripts/run_geom_extension.sh prepare supergpqa
env -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE bash scripts/run_geom_extension.sh prepare processbench
bash scripts/run_geom_extension.sh freeze
```

原 G1 的 fit 部分确定 Fisher 符号并拟合两个融合器。calibration 部分按 GSM8K、MATH 的宏平均 AP 选出一个窗口宽度。扩展集不参与冻结策略。ProcessBench 的步骤融合器和首报阈值单独在题目级开发划分上训练。

## 3. 三张 GPU 并行

将下面三个物理编号换成 nvidia-smi 中自己能用的三张卡。每张卡一个队列，CSQA 和 AQuA 串行。样本逐题完成后立即落盘。重启同一命令可续跑，脚本用输出目录锁避免两个任务写入同一组结果。

```bash
GPU_OOD=2
GPU_SUPER=6
GPU_PROCESS=7
TAG=$(date +%Y%m%d-%H%M%S)

nohup env CUDA_VISIBLE_DEVICES="$GPU_OOD" bash -c '
  set -e
  bash scripts/run_geom_extension.sh smoke csqa
  bash scripts/run_geom_extension.sh smoke aqua
  bash scripts/run_geom_extension.sh score csqa
  bash scripts/run_geom_extension.sh score aqua
' > "$LOG_DIR/legacy-$TAG.log" 2>&1 < /dev/null &
echo "CSQA+AQuA PID=$!"

nohup env CUDA_VISIBLE_DEVICES="$GPU_SUPER" bash -c '
  set -e
  bash scripts/run_geom_extension.sh smoke supergpqa
  bash scripts/run_geom_extension.sh score supergpqa
' > "$LOG_DIR/supergpqa-$TAG.log" 2>&1 < /dev/null &
echo "SuperGPQA PID=$!"

nohup env CUDA_VISIBLE_DEVICES="$GPU_PROCESS" bash -c '
  set -e
  bash scripts/run_geom_extension.sh smoke processbench
  bash scripts/run_geom_extension.sh score processbench
' > "$LOG_DIR/processbench-$TAG.log" 2>&1 < /dev/null &
echo "ProcessBench PID=$!"
```

Smoke 与正式结果使用各自的目录。上面的命令在 smoke 成功后自动进入正式任务。如果想先看显存和速度，首次只运行 smoke 一行，检查日志后再运行 score。SuperGPQA 默认正式抽 1,200 题，每题贪心生成一条最多 1,024 token 的回答。ProcessBench 默认回放四个子集全部 3,400 条解答。新任务 smoke 默认各 100 条，旧任务 smoke 默认每个旧划分选最多 8 条。

若调整 LIMIT、MAX_NEW_TOKENS、数据版本或模型，设置新的 G1_EXTENSION_ROOT，避免改写已有结果。换同型号 GPU 可以使用原参数续跑。tail -f 用 Ctrl+C 退出，后台实验会继续。

```bash
tail -n 30 -f "$LOG_DIR/processbench-$TAG.log"
pgrep -afu "$USER" 'run_geom_extension|geom_diagnostics'
```

## 4. CPU 分析与结果

GPU 阶段完成后，可以在登录节点或普通 CPU 会话中运行分析。也可以在某组完成后立即单独分析。正式 bootstrap 默认 1,000 次。需要对 smoke 测试分析时，使用 SMOKE_ANALYSIS=1 BOOTSTRAP=20，并保证 ProcessBench smoke 中开发集和测试集都有可评分样本。

```bash
TAG=$(date +%Y%m%d-%H%M%S)
nohup env CUDA_VISIBLE_DEVICES="" bash -c '
  set -e
  for task in csqa aqua supergpqa processbench; do
    bash scripts/run_geom_extension.sh analyze "$task"
  done
' > "$LOG_DIR/analysis-$TAG.log" 2>&1 < /dev/null &
echo "CPU analysis PID=$!"
```

默认结果根目录为 /mnt/data/wxh/wdl/results/OODmethod/geom-extension-v1。每组 scored/rows/ 存放逐轨迹分数，analysis/ 存放实验结果。

| 文件 | 内容 |
|---|---|
| final-answer-metrics.csv | 最终答案错误检测。AUROC、AP、梯形 PR-AUC，数值为 0–1 |
| final-answer-paired.csv / process-error-paired.csv | 同题配对增益和区间，包括 Fisher 窗口相对均值、融合增益和窗口与信号的交互增益 |
| window-fallback.csv / window-budget-8.csv / window-ap.png | 短轨迹回退率、最多八个均匀候选窗口的对照和宽度曲线。图需要 matplotlib |
| process-error-metrics.csv | ProcessBench 的过程错误检测，含最终答对子组 |
| localization-summary.csv | 离线首错覆盖率、同宽随机窗口的解析期望及首错前后关系 |
| step-metrics.csv / step-alarms.csv | Step 级排序、首报步骤、精确命中、±1 步和无错轨迹误报。与 trajectory-scores.csv 按 row_index 对应 |
| summary.json | 数据量、排除情况、生成与解析情况、版本依据 |

ProcessBench 步骤判分使用当前步骤或截至该步末尾的 token。首错后的步骤保留给首报诊断，步骤级 AP 只用首错及其之前、或者无错轨迹的步骤。公开标签绑定原解答，重生成的轨迹不能沿用。首错命中与无错识别率的调和平均按官方定义计算，阈值使用本实验按题切分的开发集。
