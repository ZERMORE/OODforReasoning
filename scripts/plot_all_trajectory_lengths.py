"""Inventory response lengths across saved OOD experiments and remove replay copies."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def statistics(rows):
    a = np.array([r["tokens"] for r in rows])
    return dict(n=len(a), mean=float(a.mean()), q25=float(np.quantile(a,.25)),
                median=float(np.median(a)), q75=float(np.quantile(a,.75)),
                p90=float(np.quantile(a,.9)), max=int(a.max()),
                correct=sum(r["correct"] for r in rows),
                truncated=sum(r["stop_reason"]=="max_new_tokens" for r in rows),
                at_most_256=int((a<=256).sum()), at_most_512=int((a<=512).sum()))


def save(fig, output, name):
    for ext in ("png","pdf"):
        fig.savefig(output / (name+"."+ext), dpi=170, facecolor="white")
    plt.close(fig)
    with Image.open(output / (name+".png")) as image:
        image.verify()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    unique={}
    experiments=[]
    for f in sorted(args.root.rglob("predictions.jsonl")):
        rel=f.relative_to(args.root).as_posix()
        entries=[]
        with f.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row=json.loads(line)
                ids=row["generated_token_ids"]
                key=hashlib.sha256(json.dumps([row["sample_id"],row["prompt_hash"],ids],
                                              separators=(",",":")).encode()).hexdigest()
                item=dict(sample_id=row["sample_id"], tokens=row["scored_tokens"],
                          correct=bool(row["is_correct"]),stop_reason=row["stop_reason"],
                          task=next(task for task in ("gsm8k","math","csqa","aqua") if task in (rel + "/" + row["sample_id"]).lower()),
                          role="pool" if "/pool/" in rel or "/pool-raw/" in rel else
                               "smoke" if "/smoke/" in rel else "test", sources=[rel])
                assert 0 < item["tokens"] <= len(ids)
                if key in unique:
                    old=unique[key]
                    assert all(old[k]==item[k] for k in ("tokens","correct","stop_reason"))
                    old["sources"].append(rel)
                    if old["role"] == "smoke" and item["role"] != "smoke":
                        old["role"] = item["role"]
                else:
                    unique[key]=item
                entries.append(item)
        experiments.append(dict(path=rel,stats=statistics(entries)))
        print(rel,len(entries),flush=True)
    derived=[]
    for f in sorted(args.root.rglob("test_predictions.jsonl")):
        with f.open(encoding="utf-8") as stream:
            entries=[json.loads(line) for line in stream if line.strip()]
        derived.append(dict(path=f.relative_to(args.root).as_posix(),n=len(entries),
                            note="Downstream aggregation output; not an additional generation run."))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    args.output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in unique.values():
        groups[row["task"],row["role"]].append(row)
    colors=("#0072B2","#A34000")
    fig,axes=plt.subplots(len(groups),2,figsize=(12,2.65*len(groups)),squeeze=False,layout="constrained")
    fig.suptitle("All saved OOD trajectories | replay copies removed",fontsize=17)
    for rowaxes,((task,role),records) in zip(axes,sorted(groups.items())):
        a=np.array([r["tokens"] for r in records])
        y=np.array([r["correct"] for r in records])
        step = 8 if a.max() <= 128 else 32 if a.max() <= 512 else 128
        edges=np.arange(0,int(a.max())+step+1,step)
        for correct,name,color,style in zip((True,False),("Correct","Incorrect"),colors,("-","--")):
            v=a[y==correct]
            if not len(v):
                continue
            count,_=np.histogram(v,bins=edges)
            rowaxes[0].stairs(count/len(v)*100,edges,color=color,linestyle=style,label=f"{name} n={len(v):,}")
            rowaxes[1].step(np.sort(v),np.arange(1,len(v)+1)/len(v)*100,
                            color=color,linestyle=style,label=name)
        rowaxes[0].set(title=f"{task.upper()} / {role} | n={len(a):,}, median={np.median(a):g}, bin={step}",
                       ylabel="Within-group percentage",xlabel="Scored response tokens")
        rowaxes[1].set(title="Cumulative distribution",ylabel="Cumulative percentage",
                       xlabel="Scored response tokens",ylim=(0,100))
        for ax in rowaxes:
            ax.axvline(256,color="#444444",ls=":",lw=1)
            ax.set_xlim(0,edges[-1])
            ax.grid(axis="y",alpha=.2)
            ax.legend(fontsize=8)
            ax.spines[["top","right"]].set_visible(False)
    save(fig,args.output,"unique-length-distributions")
    fig,axes=plt.subplots(int(np.ceil(len(experiments)/3)),3,
                          figsize=(15,3*int(np.ceil(len(experiments)/3))),layout="constrained")
    fig.suptitle("Every saved trajectory file | replay copies shown in their own panels",fontsize=16)
    for ax,exp in zip(axes.flat,experiments):
        records=[r for r in unique.values() if exp["path"] in r["sources"]]
        a=np.array([r["tokens"] for r in records])
        # Files may include repeated trajectories; use source multiplicity to retain all file rows.
        values=[]
        for r in records:
            values.extend([r["tokens"]]*r["sources"].count(exp["path"]))
        step = 8 if max(values) <= 128 else 32 if max(values) <= 512 else 128
        count,edge=np.histogram(values,bins=np.arange(0,max(values)+step+1,step))
        ax.stairs(count,edge,color="#0072B2",lw=1.4)
        ax.axvline(256,color="#444444",ls=":",lw=1)
        ax.set(title=exp["path"].replace("/predictions.jsonl","").replace("/", " /\n",1),
               xlabel="Scored response tokens",ylabel="Count")
        ax.text(.98,.95,f"n={exp['stats']['n']:,}\nmedian={exp['stats']['median']:g}",
                transform=ax.transAxes,ha="right",va="top",fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    for ax in list(axes.flat)[len(experiments):]:
        ax.set_visible(False)
    save(fig,args.output,"all-experiment-files")
    grouped={f"{task}/{role}":statistics(rows) for (task,role),rows in sorted(groups.items())}
    total=statistics(list(unique.values()))
    report=dict(root=str(args.root),total_unique=total,groups=grouped,experiments=experiments,
                downstream_files=derived,identity="sample_id + prompt_hash + complete generated_token_ids",
                definition="Scored response tokens, excludes prompt and unscored EOS.",
                bin_width="8 for maximum <=128; 32 for maximum <=512; otherwise 128",trajectories=list(unique.values()))
    (args.output/"length-statistics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# 所有实验轨迹长度统计","",
           "按 sample_id、prompt_hash 和完整生成 token IDs 去除 replay 等副本。smoke 单列，聚合结果文件单独登记。",
           "长度为 scored_tokens，不包含提示和未评分的 EOS。直方图区间宽度按最长轨迹取 8、32 或 128 token，竖虚线为 256 token。",
           "",f"去重后共 {total['n']} 条轨迹。","",
           "| 数据集 / 用途 | 条数 | 长度均值 | P25 | 中位数 | P75 | P90 | 最大 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group,s in grouped.items():
        lines.append(f"| {group} | {s['n']} | {s['mean']:.1f} | {s['q25']:g} | {s['median']:g} | {s['q75']:g} | {s['p90']:g} | {s['max']} |")
    lines+=["","![去重后长度分布](unique-length-distributions.png)","",
            "![逐实验文件长度分布](all-experiment-files.png)","","## 各实验文件","",
            "| 文件 | 轨迹数 | 中位数 | 最大长度 |","|---|---:|---:|---:|"]
    for e in experiments:
        s=e["stats"]
        lines.append(f"| {e['path']} | {s['n']} | {s['median']:g} | {s['max']} |")
    lines+=["","## 下游聚合文件",""]
    for d in derived:
        lines.append(f"- {d['path']}，{d['n']} 条，登记但不作为新增生成轨迹计数。")
    (args.output/"index.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(total=total,groups=grouped),indent=2),flush=True)


if __name__=="__main__":
    main()

