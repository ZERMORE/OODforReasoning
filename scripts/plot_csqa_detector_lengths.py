"""Draw length distributions of detector decisions for a replay dataset."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-shannon", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate derived plots in an existing directory")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Choose a new output directory")
    with (args.source/"predictions.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    n = len(rows)
    ids = np.array([r["sample_id"] for r in rows])
    lengths = np.array([r["scored_tokens"] for r in rows])
    errors = np.array([not r["is_correct"] for r in rows])
    assert len(set(ids)) == n and (lengths>0).all()
    methods = [("nci","NCI + mean","#0072B2","///"),
               ("fdbd_all","fDBD + mean","#A34000","xxx")]
    has_shannon = all(all("shannon_top20" in t["scores"] for t in r["sequence_tokens"]) for r in rows)
    if args.require_shannon and not has_shannon:
        parser.error("Top-20 Shannon scores are missing. Replay the original source with --shannon into a new directory.")
    if has_shannon:
        methods.append(("shannon_top20","Shannon Drop","#007C62","..."))
    edges = np.arange(0, lengths.max()+5, 4)
    k = int(np.ceil(.25*n))
    decisions = {}
    for key, name, color, hatch in methods:
        values=[]
        for row in rows:
            tokens=row["sequence_tokens"]
            assert len(tokens)==row["scored_tokens"]
            x=np.array([t["scores"][key] for t in tokens])
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite token scores")
            if key=="shannon_top20":
                smooth=x.copy()
                for t in range(1,len(x)):
                    smooth[t]=x[t]/3+smooth[t-1]*2/3
                delta=np.diff(smooth)
                negative=np.sort(delta[delta<0])
                values.append(-negative[:5].mean() if len(negative) else 0.)
            else:
                values.append(-x.mean())
        order=np.lexsort((ids,-np.array(values)))
        flags=np.zeros(n,dtype=bool)
        flags[order[:k]]=True
        decisions[key] = {}
        for category,mask in (("flagged",flags),("false_error",flags&~errors),("false_correct",~flags&errors)):
            hist,_=np.histogram(lengths[mask],edges)
            assert hist.sum()==mask.sum()
            decisions[key][category]=hist.tolist()
    totals=np.histogram(lengths,edges)[0]
    fig,axes=plt.subplots(2,2,figsize=(12,8.2))
    fig.subplots_adjust(top=.82,bottom=.13,left=.075,right=.98,hspace=.42,wspace=.19)
    fig.suptitle(f"CSQA | response length and detector decisions | n={n:,}",fontsize=17,y=.97)
    categories=[("all","All test trajectories"),("flagged",f"Predicted incorrect | {k} per detector"),
                ("false_error","False error | correct answer flagged incorrect"),
                ("false_correct","False correct | incorrect answer accepted")]
    for ax,(category,title) in zip(axes.flat,categories):
        if category=="all":
            ax.stairs(totals,edges,baseline=0,fill=True,facecolor="#667D8C",edgecolor="#3D5361",alpha=.5)
        else:
            for key,name,color,hatch in methods:
                ax.stairs(decisions[key][category],edges,baseline=0,fill=True,
                          facecolor=color,edgecolor=color,alpha=.3,hatch=hatch)
        ax.set(title=title,xlabel="Response length (scored tokens)",ylabel="Number of trajectories",
               xlim=(0,edges[-1]),ylim=(0,None))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    handles=[Patch(facecolor=c,edgecolor=c,alpha=.3,hatch=h,label=name) for _,name,c,h in methods]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.925),ncol=len(methods),frameon=False)
    note="Top-20 Shannon scores are unavailable in this replay." if not has_shannon else "Shannon Drop uses EMA span 5 and the five largest declines."
    fig.text(.5,.052,"Each detector flags its highest-risk 25%. Bin width = 4 tokens. Independent axis ranges per panel.",
             ha="center",fontsize=9)
    fig.text(.5,.025,note,ha="center",fontsize=9,color="#465562")
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    for ext in ("png","pdf"):
        fig.savefig(args.output/f"csqa-decisions-by-length.{ext}",dpi=200,facecolor="white")
    plt.close(fig)
    with Image.open(args.output/"csqa-decisions-by-length.png") as im:
        im.verify()
    report=dict(source=str(args.source),n=n,errors=int(errors.sum()),screened=k,
                top20_shannon_available=has_shannon,bin_edges=edges.tolist(),
                all_trajectories=totals.tolist(),decisions=decisions,
                tie_break="sample_id lexicographic")
    (args.output/"plot-data.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("n/errors/flagged",n,errors.sum(),k)
    for key,counts in decisions.items():
        print(key,{name:sum(values) for name,values in counts.items()})


if __name__=="__main__":
    main()

