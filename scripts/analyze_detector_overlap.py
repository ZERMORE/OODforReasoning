"""Compare calibrated and equal-budget errors of geometry mean and Shannon Drop."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


def overlap(a,b,mask):
    a,b=a&mask,b&mask
    n=int(mask.sum()); na=int(a.sum()); nb=int(b.sum())
    inter=int((a&b).sum()); union=int((a|b).sum())
    expected=na*nb/n if n else 0.
    return dict(a=na,b=nb,intersection=inter,a_only=na-inter,b_only=nb-inter,
                jaccard=inter/union if union else None,
                expected_intersection=expected,
                enrichment=inter/expected if expected else None)


def budget(risk,fraction,ids):
    order=np.lexsort((ids,-risk))
    mask=np.zeros(len(risk),dtype=bool)
    mask[order[:int(np.ceil(len(risk)*fraction))]]=True
    return mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    report={}
    sources={"math":"evidence-math-half-v1/math/fit-no-shuffle",
             "gsm8k":"evidence-crossover-v1/gsm8k/fit"}
    for task,rel in sources.items():
        p=args.root/rel
        with (p/"test_predictions.jsonl").open(encoding="utf-8") as f:
            rows=[json.loads(l) for l in f if l.strip()]
        metrics=json.loads((p/"evidence_metrics.json").read_text(encoding="utf-8"))
        y=np.array([not r["is_correct"] for r in rows])
        length=np.array([r["scored_tokens"] for r in rows])
        ids=np.array([r["sample_id"] for r in rows])
        assert len(set(ids))==len(ids)
        confidence={k:np.array([r["confidence"][k]["fixed"] for r in rows])
                    for k in ("fdbd_all/mean","nci/mean","shannon_top20/evidence_drop")}
        item=dict(n=len(y),errors=int(y.sum()),source=str(p),comparisons={})
        for geometry in ("fdbd_all/mean","nci/mean"):
            a,b=confidence[geometry],confidence["shannon_top20/evidence_drop"]
            ra,rb=-a,-b
            lr=rankdata(length)
            residual=[]
            design=np.column_stack([np.ones(len(y)),lr])
            for risk in (ra,rb):
                ranks=rankdata(risk)
                residual.append(ranks-design@np.linalg.lstsq(design,ranks,rcond=None)[0])
            comp=dict(spearman=float(spearmanr(ra,rb).statistic),
                      partial_rank_correlation_length=float(np.corrcoef(residual)[0,1]),settings={})
            for setting in ("0.05","0.1","0.5","budget25"):
                if setting=="budget25":
                    pa,pb=budget(ra,.25,ids),budget(rb,.25,ids)
                    thresholds=None
                else:
                    thresholds=[metrics["combinations"][k]["fixed"]["operating_points"][setting]["confidence_threshold"]
                                for k in (geometry,"shannon_top20/evidence_drop")]
                    pa,pb=a<=thresholds[0],b<=thresholds[1]
                result=dict(thresholds=thresholds,
                            false_correct=overlap(~pa,~pb,y),
                            false_error=overlap(pa,pb,~y),
                            accepted_a=int((~pa).sum()),accepted_b=int((~pb).sum()),
                            groups={})
                for name,mask in (("both_reject",pa&pb),("geometry_only_reject",pa&~pb),
                                  ("drop_only_reject",~pa&pb),("both_accept",~pa&~pb)):
                    result["groups"][name]=dict(n=int(mask.sum()),errors=int(y[mask].sum()),
                                                error_rate=float(y[mask].mean()) if mask.any() else None,
                                                median_tokens=float(np.median(length[mask])) if mask.any() else None,
                                                sample_ids=ids[mask].tolist())
                comp["settings"][setting]=result
            item["comparisons"][geometry]=comp
            print(task,geometry,"correlations",round(comp["spearman"],3),round(comp["partial_rank_correlation_length"],3))
            for setting in ("0.1","budget25"):
                r=comp["settings"][setting]
                print(setting,"false_correct",r["false_correct"],"false_error",r["false_error"])
                print("groups",[(k,{j:v for j,v in g.items() if j!="sample_ids"}) for k,g in r["groups"].items()])
        report[task]=item
    args.output.mkdir(parents=True)
    (args.output/"overlap-results.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

