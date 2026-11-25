"""Separate step-wise OOD study; reuses immutable StepCluster responses and OOD scoring."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ood_repro.data import digest
from ood_repro.study_io import (check_record, file_hash, seal_record, start_output, write_json)

RULES = ('full_mean', 'reasoning_token_mean', 'step_mean', 'worst_step',
         'worst_two_steps', 'worst_adjacent_two', 'low_step_fraction')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def implementation():
    root = Path(__file__).resolve().parents[1]/'ood_repro'
    return digest({'step_ood/study.py': file_hash(__file__), **{n: file_hash(root/n) for n in
                   ('scoring.py', 'engine.py', 'cli.py', 'study_io.py', 'data.py')}})


def source_records(root, task, split):
    cell = root/task/split
    manifest = load_json(cell/'generation.json')
    if manifest.get('complete') is not True:
        raise ValueError(f'StepCluster generation not complete: {cell}')
    files = list((cell/'merged').glob('gsm8k_*.json'))
    rows = sorted([(p, load_json(p)) for p in files], key=lambda x: int(x[1]['id']))
    if len(rows) != manifest['n'] or len({r['id'] for _, r in rows}) != len(rows):
        raise ValueError(f'Incomplete or duplicate source files: {cell}')
    if not rows:
        raise ValueError('Empty source')
    for _, row in rows:
        if not isinstance(row.get('is_correct'), bool):
            raise ValueError('Source must contain boolean final-answer labels')
        prompt, full = row['input_ids'], row['full_seq_ids']
        if not prompt or full[:len(prompt)] != prompt or len(full) <= len(prompt):
            raise ValueError('Source token prefix mismatch/empty continuation')
    return cell, manifest, rows


def content_spans(text):
    """Exclude literal Step N: markers and #### answer, retaining each step's prose."""
    end = text.find('####')
    if end < 0:
        raise ValueError('missing_answer_marker')
    matches = list(re.finditer(r'(?m)^[ \t]*(?:\*\*)?Step[ \t]+(\d+)[ \t]*:(?:\*\*)?[ \t]*', text[:end]))
    if not matches or [int(m[1]) for m in matches] != list(range(1, len(matches)+1)):
        raise ValueError('missing_or_nonsequential_steps')
    if text[:matches[0].start()].strip():
        raise ValueError('unassigned_preamble')
    spans = []
    for i, m in enumerate(matches):
        stop = matches[i+1].start() if i+1 < len(matches) else end
        if not text[m.end():stop].strip():
            raise ValueError('empty_step')
        spans.append((m.end(), stop))
    return spans


def reduce_steps(text, offsets, trace):
    spans = content_spans(text)
    metrics = sorted(trace[0]['scores']) if trace else []
    if not metrics:
        raise ValueError('empty_token_scores')
    result = []
    for j, (start, stop) in enumerate(spans, 1):
        selected = [t for t in trace if offsets[t['position']][1] > offsets[t['position']][0]
                    and start <= offsets[t['position']][0] and offsets[t['position']][1] <= stop]
        if not selected:
            raise ValueError('step_without_fully_contained_tokens')
        result.append({'step': j, 'char_span': [start, stop], 'n_tokens': len(selected),
                       'token_positions': [t['position'] for t in selected],
                       'scores': {k: float(np.mean([t['scores'][k] for t in selected])) for k in metrics}})
    return result


def score_response(model, tokenizer, scorer, row):
    from ood_repro.engine import eos_ids, replay
    generated = row['full_seq_ids'][len(row['input_ids']):]
    raw = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    clean = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if raw != row['decoded_with_specials'] or clean != row['produced_text']:
        raise ValueError('Source tokenizer decoding changed')
    encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != generated:
        raise ValueError('Tokenizer roundtrip differs; cannot silently realign steps')
    stops = eos_ids(model, tokenizer)
    if any(t in stops for t in generated[:-1]):
        raise ValueError('EOS occurs before continuation ends')
    result = replay(model, row['input_ids'], generated, scorer, mode='matched',
                    stop_ids=stops, include_eos=False, save_tokens=True)
    steps = reduce_steps(raw, encoded['offset_mapping'], result['token_scores'])
    return {'step_scores': steps, 'full_scores': result['scores'],
            'token_scores': result['token_scores'], 'scored_tokens': result['scored_tokens']}


def replay_source(args):
    from ood_repro.cli import environment, load_model, parse_ks
    from ood_repro.scoring import GeometryScorer
    from tqdm import tqdm
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Output must be independent from the StepCluster source')
    cell, manifest, rows = source_records(source, args.task, args.split)
    rows = rows[:args.limit] if args.limit else rows
    config = manifest['config']
    args.model = args.model or config['model']
    if file_hash(Path(args.model)/'config.json') != config['model_config']:
        raise ValueError('Model config differs from source checkpoint')
    # A relocated snapshot must retain its pinned snapshot-directory name.
    if Path(args.model).resolve() != Path(config['model']).resolve() and Path(args.model).name != Path(config['model']).name:
        raise ValueError('Relocated model must retain original snapshot identity')
    settings = {'kind': 'step-ood-replay-v1', 'source': str(cell),
                'source_manifest': file_hash(cell/'generation.json'),
                'source_rows': [[r['id'], file_hash(p)] for p, r in rows],
                'implementation': implementation(), 'ks': args.ks, 'dtype': args.dtype,
                'attention': args.attention, 'limit': args.limit, 'model': str(Path(args.model).resolve()),
                'token_policy': 'fully-contained content tokens; raw-logit scores; exclude EOS; matched replay'}
    signature, previous, saved = start_output(output, settings, args.resume)
    if len(previous) > len(rows):
        raise ValueError('Resume contains extra rows')
    for old, (_, row) in zip(previous, rows):
        if old['source_hash'] != digest(row):
            raise ValueError('Resume/source row mismatch')
    if len(previous) == len(rows) and saved and saved.get('complete'):
        print(f'Already complete: {output}', flush=True)
        return
    model = tokenizer = scorer = None
    state = {'signature': signature, 'settings': settings, 'source_generation': manifest,
             'complete': False, 'environment': saved.get('environment') if saved else None}
    write_json(output/'manifest.json', state)
    with (output/'predictions.jsonl').open('a', encoding='utf-8') as stream:
        for _, row in tqdm(rows[len(previous):], initial=len(previous), total=len(rows), desc=f'step OOD/{args.split}'):
            record = {'sample_id': f'{args.task}/{args.split}/{row["id"]}', 'source_id': row['id'],
                      'question_hash': digest(' '.join(row['question'].split())),
                      'question': row['question'], 'response': row['produced_text'],
                      'source_hash': digest(row), 'is_correct': row['is_correct'],
                      'produced_answer': row.get('produced_answer'), 'gold_answer': row.get('gold_answer')}
            reason = 'truncated' if row['truncated'] else None
            if reason is None:
                try:
                    content_spans(row['produced_text'])
                except ValueError as error:
                    reason = str(error)
            if reason is None:
                if model is None:
                    model, tokenizer = load_model(args)
                    env = environment(model, tokenizer)
                    if state['environment'] is not None:
                        for key in ('packages', 'model_commit', 'tokenizer_commit', 'chat_template_hash'):
                            if state['environment'][key] != env[key]:
                                raise ValueError(f'Replay environment changed: {key}')
                    state['environment'] = env
                    write_json(output/'manifest.json', state)
                    head = model.get_output_embeddings()
                    scorer = GeometryScorer(head.weight, getattr(head, 'bias', None),
                                            ks=parse_ks(args.ks), metrics={'nci'})
                record.update(score_response(model, tokenizer, scorer, row))
            record['excluded_reason'] = reason
            stream.write(json.dumps(seal_record(record), ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
    state['complete'] = True
    write_json(output/'manifest.json', state)
    print(f'Complete: {len(rows)} source trajectories -> {output}', flush=True)


def confidence(row, metric, rule, step_threshold):
    steps = row['step_scores']
    x = np.asarray([s['scores'][metric] for s in steps], dtype=float)
    if rule == 'full_mean':
        return row['full_scores'][metric]
    if rule == 'reasoning_token_mean':
        return float(np.average(x, weights=[s['n_tokens'] for s in steps]))
    if rule == 'step_mean':
        return float(x.mean())
    if rule == 'worst_step':
        return float(x.min())
    if rule == 'worst_two_steps':
        return float(np.sort(x)[:2].mean())
    if rule == 'worst_adjacent_two':
        return float(((x[:-1]+x[1:])/2).min()) if len(x)>1 else float(x[0])
    if rule == 'low_step_fraction':
        return -float(np.mean(x < step_threshold))
    raise ValueError(rule)


def lower_threshold(values, alpha):
    """Strict '<' lower-tail split-conformal rule, conservative with ties."""
    x = np.sort(np.asarray(values, dtype=float))
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Need finite correct calibration scores')
    rank = math.floor(alpha*(len(x)+1))
    return float(x[rank-1]) if rank >= 1 else None


def detection(y, uncertainty, alarm):
    y, alarm = np.asarray(y, dtype=bool), np.asarray(alarm, dtype=bool)
    tp, fp = int((y & alarm).sum()), int((~y & alarm).sum())
    fn, tn = int((y & ~alarm).sum()), int((~y & ~alarm).sum())
    return {'n': len(y), 'errors': int(y.sum()), 'error_prevalence': float(y.mean()),
            'auroc': float(roc_auc_score(y, uncertainty)) if len(np.unique(y)) == 2 else None,
            'ap_error': float(average_precision_score(y, uncertainty)) if y.any() else None,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'fpr': fp/(fp+tn) if fp+tn else None, 'tpr': tp/(tp+fn) if tp+fn else None,
            'precision': tp/(tp+fp) if tp+fp else None, 'accuracy': (tp+tn)/len(y)}


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyzed_run(root, split):
    path = root/split
    manifest = load_json(path/'manifest.json')
    if not manifest.get('complete'):
        raise ValueError(f'Incomplete replay: {split}')
    rows = jsonl(path/'predictions.jsonl')
    if len(rows) != len(manifest['settings']['source_rows']):
        raise ValueError(f'Missing replay rows: {split}')
    for r in rows:
        check_record(r)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate replay sample IDs')
    return manifest, rows


def analyze(args):
    root, output = args.replays.resolve(), args.output.resolve()
    if output == root or output in root.parents:
        raise ValueError('Use a dedicated analysis output directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Analysis output nonempty; choose a new --output')
    manifests, all_rows, data = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        manifests[split], all_rows[split] = analyzed_run(root, split)
        data[split] = [r for r in all_rows[split] if r['excluded_reason'] is None]
        if not data[split]:
            raise ValueError(f'No retained rows in {split}')
    for i, a in enumerate(data):
        for b in list(data)[i+1:]:
            if {r['question_hash'] for r in all_rows[a]} & {r['question_hash'] for r in all_rows[b]}:
                raise ValueError(f'Question leakage: {a}/{b}')
    for split in ('validation', 'test'):
        for key in ('ks', 'dtype', 'attention', 'implementation', 'model'):
            if manifests[split]['settings'][key] != manifests['train']['settings'][key]:
                raise ValueError(f'Inconsistent replay settings: {split}/{key}')
        for key in ('model_config', 'prompt', 'stop_mode', 'max_new_tokens', 'adapter'):
            if manifests[split]['source_generation']['config'].get(key) != manifests['train']['source_generation']['config'].get(key):
                raise ValueError(f'Inconsistent source generation: {split}/{key}')
    correct_train = [r for r in data['train'] if r['is_correct']]
    correct_cal = [r for r in data['validation'] if r['is_correct']]
    if not correct_train or not correct_cal:
        raise ValueError('Train and validation both need correct trajectories')
    metrics = sorted(data['train'][0]['full_scores'])
    for split in data:
        for r in data[split]:
            if sorted(r['full_scores']) != metrics or any(sorted(s['scores']) != metrics for s in r['step_scores']):
                raise ValueError('Inconsistent score keys')
    output.mkdir(parents=True, exist_ok=True)
    reference = {m: float(np.quantile([s['scores'][m] for r in correct_train for s in r['step_scores']], args.step_quantile)) for m in metrics}
    calibration, reports, test_columns = {}, [], {}
    y = np.asarray([not r['is_correct'] for r in data['test']], dtype=int)
    decisions = [{'sample_id':r['sample_id'], 'is_error':int(not r['is_correct']), 'n_steps':len(r['step_scores'])} for r in data['test']]
    groups = []
    for metric in metrics:
        for rule in RULES:
            key = f'{metric}/{rule}'
            cal_values = [confidence(r, metric, rule, reference[metric]) for r in correct_cal]
            threshold = lower_threshold(cal_values, args.fpr)
            values = np.asarray([confidence(r, metric, rule, reference[metric]) for r in data['test']])
            alarms = values < threshold if threshold is not None else np.zeros(len(values), dtype=bool)
            cal_alarm = np.asarray(cal_values) < threshold if threshold is not None else np.zeros(len(cal_values), dtype=bool)
            calibration[key] = {'threshold': threshold, 'n_correct':len(cal_values),
                                'correct_validation_fpr':float(cal_alarm.mean()), 'disabled_small_calibration':threshold is None}
            report = {'metric':metric, 'rule':rule, 'threshold':threshold, **detection(y, -values, alarms)}
            reports.append(report)
            test_columns[key] = -values
            for i, d in enumerate(decisions):
                d[key+'/confidence'] = float(values[i])
                d[key+'/alarm'] = bool(alarms[i])
                if rule == 'worst_step':
                    d[key+'/first_alarm_step'] = next((s['step'] for s in data['test'][i]['step_scores']
                                                       if threshold is not None and s['scores'][metric] < threshold), None)
            for k in sorted({len(r['step_scores']) for r in data['test']}):
                mask = np.asarray([len(r['step_scores']) == k for r in data['test']])
                groups.append({'metric':metric, 'rule':rule, 'n_steps':k, **detection(y[mask], -values[mask], alarms[mask])})
    # Resample the same test questions for every method, with thresholds frozen.
    draws = {key:{'auroc':[], 'ap_error':[]} for key in test_columns}
    rng = np.random.default_rng(args.seed)
    for _ in range(args.bootstrap):
        ix = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) < 2:
            continue
        for key, values in test_columns.items():
            draws[key]['auroc'].append(float(roc_auc_score(y[ix], values[ix])))
            draws[key]['ap_error'].append(float(average_precision_score(y[ix], values[ix])))
    for r in reports:
        key = r['metric']+'/'+r['rule']
        for stat in ('auroc', 'ap_error'):
            samples = draws[key][stat]
            if samples:
                lo, hi = np.quantile(samples, [.025, .975])
                r[stat+'_ci_low'], r[stat+'_ci_high'] = float(lo), float(hi)
                for control in ('full_mean', 'reasoning_token_mean'):
                    delta = np.asarray(samples)-np.asarray(draws[r['metric']+'/'+control][stat])
                    base = next(x[stat] for x in reports if x['metric']==r['metric'] and x['rule']==control)
                    r[stat+'_delta_vs_'+control] = r[stat]-base
                    lo, hi = np.quantile(delta, [.025, .975])
                    r[stat+'_delta_vs_'+control+'_ci_low'] = float(lo)
                    r[stat+'_delta_vs_'+control+'_ci_high'] = float(hi)
    step_rows = []
    for split in data:
        for r in data[split]:
            for s in r['step_scores']:
                item = {'split':split, 'sample_id':r['sample_id'], 'trajectory_is_correct':r['is_correct'],
                        'step':s['step'], 'n_steps':len(r['step_scores']), 'n_tokens':s['n_tokens']}
                for m in metrics:
                    item[m] = s['scores'][m]
                    item[m+'_below_train_reference'] = s['scores'][m] < reference[m]
                    threshold = calibration[m+'/worst_step']['threshold']
                    item[m+'_below_worst_step_threshold'] = threshold is not None and s['scores'][m] < threshold
                step_rows.append(item)
    audit = {split:{'n_source':len(all_rows[split]), 'n_retained':len(data[split]),
                    'n_correct_source':sum(r['is_correct'] for r in all_rows[split]),
                    'n_correct_retained':sum(r['is_correct'] for r in data[split]),
                    'n_excluded_correct':sum(r['is_correct'] for r in all_rows[split] if r['excluded_reason']),
                    'n_excluded_error':sum(not r['is_correct'] for r in all_rows[split] if r['excluded_reason'])} for split in data}
    protocol = {'target_correct_trajectory_fpr':args.fpr, 'train_step_quantile':args.step_quantile,
                'train_step_thresholds':reference, 'calibration':calibration,
                'threshold_rule':'confidence < kth correct-validation score, k=floor(alpha*(n+1)); k=0 disables alarm',
                'guarantee':'Per-rule marginal FPR bound requires exchangeable correct calibration/test trajectories; not simultaneous across rules or step counts.',
                'step_labels':'No true step correctness labels; below-reference is a heuristic flag, not proof of a wrong step.',
                'bootstrap':{'requested':args.bootstrap, 'accepted':len(next(iter(draws.values()))['auroc']),
                             'seed':args.seed, 'unit':'paired test question; calibration thresholds fixed'},
                'audit':audit, 'source_manifests':manifests}
    protocol['analysis_implementation'] = implementation()
    protocol['source_predictions_sha256'] = {s:file_hash(root/s/'predictions.jsonl') for s in data}
    write_json(output/'protocol.json', protocol)
    write_json(output/'metrics.json', reports)
    table(output/'metrics.csv', reports)
    table(output/'step-scores.csv', step_rows)
    table(output/'test-decisions.csv', decisions)
    table(output/'metrics-by-step-count.csv', groups)
    table(output/'excluded.csv', [{'split':s, 'sample_id':r['sample_id'], 'is_correct':r['is_correct'],
                                  'reason':r['excluded_reason']} for s in all_rows for r in all_rows[s] if r['excluded_reason']])
    print(json.dumps({'audit':audit, 'test_error_prevalence':float(y.mean()), 'results':str(output)}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    p = sub.add_parser('replay')
    p.add_argument('--source', type=Path, required=True, help='StepCluster root/RUN_NAME')
    p.add_argument('--task', default='gsm8k', choices=['gsm8k', 'math500', 'math-full'])
    p.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    from ood_repro.cli import model_args
    model_args(p)
    p.add_argument('--ks', default='all')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(local_files_only=True, seed=42)
    p = sub.add_parser('analyze')
    p.add_argument('--replays', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fpr', type=float, default=.1)
    p.add_argument('--step-quantile', type=float, default=.1)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.phase == 'replay':
        if args.limit < 0:
            parser.error('--limit must be nonnegative')
        replay_source(args)
    else:
        if not 0 < args.fpr < 1 or not 0 < args.step_quantile < 1 or args.bootstrap < 0:
            parser.error('Invalid threshold quantile, FPR or bootstrap count')
        analyze(args)


if __name__ == '__main__':
    main()
