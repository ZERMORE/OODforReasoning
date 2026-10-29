"""Export every saved eval2 aggregate result into readable Markdown tables."""
import argparse
import json
from pathlib import Path

import numpy as np


def interval(point, bounds, signed=False):
    return f"{100*point:+.2f}" + f" [{100*bounds[0]:+.2f}, {100*bounds[1]:+.2f}]" if signed else f"{100*point:.2f} [{100*bounds[0]:.2f}, {100*bounds[1]:.2f}]"


def table(headers, rows):
    return "|" + "|".join(headers) + "|\n|" + "|".join(["---"]*len(headers)) + "|\n" + "\n".join("|"+"|".join(str(v) for v in row)+"|" for row in rows) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="experiment2/main")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents or (output.exists() and any(output.iterdir())):
        p.error("Use a new empty output directory independent of source")
    d = json.loads((source/'experiment2.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['complete'], 'Formal run incomplete'
    output.mkdir(parents=True, exist_ok=True)
    records = d['records']
    predictions = [json.loads(line) for line in (source/'test_predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    parts = []
    def save(name, title, headers, rows, note=''):
        text = '# '+title+'\n\n'+note+'\n\n'+table(headers, rows)
        (output/name).write_text(text, encoding='utf-8')
        parts.append(text)
    methods = list(dict.fromkeys(r['method'] for r in records))
    bykey = {(r['metric'], r['method']): r for r in records}
    main_rows=[]
    for method in methods:
        main_rows.append([method]+[interval(bykey[(metric,method)]['auroc'],bykey[(metric,method)]['ci95']) for metric in ('fdbd_all','nci')])
    save('01-main.md','表1 全部方法主结果（42项）',['方法','fDBD-All AUROC [95% CI]','NCI AUROC [95% CI]'],main_rows,
         '数值均为 AUROC×100。神经模型点值是训练种子AUROC平均，不是预测集成。所有区间未作多重比较校正。')
    save('02-delta-vs-mean.md','表2 相对同评分均值的完整配对差值',['评分','方法','差值百分点 [95% CI]'],
         [[r['metric'],r['method'],interval(r['paired_deltas']['mean']['delta'],r['paired_deltas']['mean']['ci95'],True)] for r in records])
    save('03-seeds.md','表3 全部测试种子成绩与种子波动',['评分','方法','fixed/seed0','seed1','seed2','种子标准差（点）'],
         [[r['metric'],r['method'],f"{100*r['replica_aurocs'].get('0',r['replica_aurocs'].get('fixed',0)):.2f}",
           f"{100*r['replica_aurocs']['1']:.2f}" if '1' in r['replica_aurocs'] else '—',
           f"{100*r['replica_aurocs']['2']:.2f}" if '2' in r['replica_aurocs'] else '—',f"{100*r['seed_std']:.2f}"] for r in records],
         'fixed代表无需训练的固定汇总；LR只拟合一次。固定汇总/单次拟合的标准差0不表示不存在统计不确定性。')
    save('04-all-paired.md','表4 所有已保存配对比较（210项）',['评分','前者','减去的对照','差值百分点 [95% CI]'],
         [[r['metric'],r['method'],reference,interval(v['delta'],v['ci95'],True)] for r in records for reference,v in r['paired_deltas'].items()],
         '差值统一为前者减对照；GRU控制行减gru是负值时表示破坏顺序后下降。自身比较的0是定义所致。')
    save('05-training.md','表5 所有模型训练与选择记录（38项）',['模型/种子','参数量','最佳epoch','最佳验证AUROC×100'],
         [[name,v['parameters'],v.get('best_epoch','不适用'),f"{100*v['best_validation_auroc']:.2f}" if 'best_validation_auroc' in v else '不适用'] for name,v in d['jobs'].items()],
         '验证集选择epoch；LR没有神经训练epoch。参数量不同，不能把GRU与DeepSets当成完全等容量对照。')
    save('06-thresholds.md','表6 各模型种子的阈值与实际检测表现（38项）',['模型/种子','原始分数阈值','校准FPR %','测试FPR %','测试TPR %'],
         [[name,f"{v['threshold']:.8f}",f"{100*v['calibration_actual_fpr']:.2f}",f"{100*v['test_actual_fpr']:.2f}",f"{100*v['test_tpr']:.2f}"] for name,v in d['thresholds'].items()],
         '阈值的原始分数单位不可跨模型比较。这里的TPR不是TPR@5%测试FPR；必须同时看实际测试FPR。')
    macro=[]
    for metric in ('fdbd_all','nci'):
        for model in ('logreg','deepsets','gru','cnn','attention','attention_nopos','gru_shuffled'):
            items=[v for k,v in d['thresholds'].items() if k.startswith(metric+'/'+model+'::')]
            macro.append([metric,model]+[f"{100*np.mean([v[key] for v in items]):.2f}" for key in ('calibration_actual_fpr','test_actual_fpr','test_tpr')])
    save('07-threshold-summary.md','表7 阈值表现按训练种子平均',['评分','模型','校准FPR %','测试FPR %','测试TPR %'],macro,
         '均值是各模型种子实际比例的平均；不是一个集成模型的新阈值。')
    save('08-data.md','表8 数据与运行设置',['项目','数值'],
         [[k,f"{v['n']}题 / {v['errors']}错"] for k,v in manifest['counts'].items()]+[
             ['测试',f"{len(predictions)}题 / {sum(not r['is_correct'] for r in predictions)}错"],['epochs',manifest['config']['epochs']],['seeds',manifest['config']['seeds']],
             ['hidden',manifest['config']['hidden']],['batch_size',manifest['config']['batch_size']],['lr',manifest['config']['lr']],
             ['bootstrap',manifest['config']['bootstrap']],['complete',manifest['complete']]])
    (output/'全部结果数据表.md').write_text('# Eval section2 全部汇总结果\n\n来源：'+str(source)+'\n\n'+ '\n\n'.join(parts),encoding='utf-8')
    (output/'full_precision_results.json').write_text(json.dumps({'source':str(source),'results':d,'manifest':manifest},ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,12))
    ax.axis('off')
    ax.set_title('Eval section 2: complete test results\nAUROC x 100 [unadjusted 95% CI]',fontsize=15,pad=20)
    t=ax.table(cellText=main_rows,colLabels=['Method','fDBD-All','NCI'],cellLoc='center',loc='center',colWidths=[.38,.31,.31])
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.05)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#d0d5dc')
        if i==0:
            cell.set_facecolor('#16324f');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
        elif main_rows[i-1][0]=='gru':
            cell.set_facecolor('#e3eff9');cell.get_text().set_weight('bold')
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(output/f'complete_main_table.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(f'Exported 8 result tables, exact JSON and main-table image: {output}')


if __name__=='__main__':
    main()
