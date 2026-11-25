"""Step alignment, causal scoring integration, calibrated rules and held-out audit."""
from argparse import Namespace
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from step_ood import study as s


def test_content_alignment_excludes_markers_and_answer():
    text = 'Step 1: alpha\nStep 2: beta\n#### 999'
    trace = [{'position':i, 'scores':{'nci':float(i), 'fdbd_all':float(i)/2}} for i in range(len(text))]
    steps = s.reduce_steps(text, [(i,i+1) for i in range(len(text))], trace)
    assert len(steps) == 2
    assert ''.join(text[i] for i in steps[0]['token_positions']) == 'alpha\n'
    assert ''.join(text[i] for i in steps[1]['token_positions']) == 'beta\n'
    for text in ('Step 2: x\n#### 1', 'Step 1: \n#### 1', 'Step 1: x', 'Preamble\nStep 1: x\n#### 1'):
        with pytest.raises(ValueError):
            s.content_spans(text)
    # Token crossing the header/content boundary is excluded, not misattributed.
    result = s.reduce_steps('Step 1: abc\n#### 1', [(6,9),(9,10)],
                            [{'position':0,'scores':{'nci':100}}, {'position':1,'scores':{'nci':2}}])
    assert result[0]['token_positions'] == [1]


def row(values=(1.,3.,2.), counts=(10,1,1)):
    return {'full_scores':{'nci':9., 'fdbd_all':9.},
            'step_scores':[{'step':i+1, 'n_tokens':n, 'scores':{'nci':v,'fdbd_all':v}} for i,(v,n) in enumerate(zip(values,counts))]}


def test_rules_threshold_ties_and_weighting():
    r = row()
    assert s.confidence(r,'nci','step_mean',2)==2
    assert s.confidence(r,'nci','reasoning_token_mean',2)==pytest.approx(15/12)
    assert s.confidence(r,'nci','worst_step',2)==1
    assert s.confidence(r,'nci','worst_two_steps',2)==1.5
    assert s.confidence(r,'nci','worst_adjacent_two',2)==2
    assert s.confidence(r,'nci','low_step_fraction',2)==pytest.approx(-1/3)
    assert s.lower_threshold([1]*19,.1)==1
    assert not np.any(np.array([1]*19)<s.lower_threshold([1]*19,.1))
    assert s.lower_threshold([1,2],.1) is None
    assert s.lower_threshold(range(19),.1)==1
    for rule in s.RULES:
        assert np.isfinite(s.confidence(row((2.,),(1,)),'nci',rule,1))


def test_score_response_uses_preceding_causal_state(monkeypatch):
    from ood_repro import engine
    from ood_repro.scoring import GeometryScorer
    # Actual shared replay and scorer, with an identity-backbone toy causal model.
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(4)
            self.embed=torch.nn.Embedding(128,4)
            self.head=torch.nn.Linear(4,128,bias=False)
            self.generation_config=SimpleNamespace(eos_token_id=0)
        def get_input_embeddings(self): return self.embed
        def get_output_embeddings(self): return self.head
        def forward(self,input_ids,**kwargs):
            return SimpleNamespace(logits=self.head(self.embed(input_ids)),past_key_values=None)
    class Tokenizer:
        eos_token_id=0
        def decode(self,ids,skip_special_tokens=False,**kwargs):
            return ''.join(chr(i) for i in ids if not(skip_special_tokens and i==0))
        def __call__(self,text,**kwargs):
            return {'input_ids':[ord(c) for c in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
    model, tokenizer=Model(),Tokenizer()
    monkeypatch.setattr(engine,'new_cache',lambda:None)
    text='Step 1: a\nStep 2: b\n#### 1'
    tokens=[ord(c) for c in text]+[0]
    source={'input_ids':[1,2], 'full_seq_ids':[1,2]+tokens,
            'produced_text':text,'decoded_with_specials':text+chr(0)}
    scorer=GeometryScorer(model.head.weight,ks=('all',),metrics={'nci'},vocab_chunk=31)
    output=s.score_response(model,tokenizer,scorer,source)
    assert output['scored_tokens']==len(text)
    assert len(output['step_scores'])==2
    assert all(t['token_id']!=0 for t in output['token_scores'])
    for i,t in enumerate(output['token_scores']):
        preceding=2 if i==0 else tokens[i-1]
        z=model.embed.weight[preceding]
        expected=scorer.score(z,model.head(z),tokens[i])
        for k in expected:
            assert t['scores'][k]==pytest.approx(expected[k].item())
    source['produced_text']='changed'
    with pytest.raises(ValueError,match='decoding'):
        s.score_response(model,tokenizer,scorer,source)


def fixture(root):
    config={'model_config':'same','prompt':'chat','stop_mode':'final-line','max_new_tokens':2048,'adapter':'same'}
    for split in ('train','validation','test'):
        p=root/split;p.mkdir(parents=True)
        rows=[]
        for i in range(30):
            r=row((.2,.3) if i%3==0 else (2.,3.),(2,3))
            r.update(sample_id=f'{split}/{i}',question_hash=f'{split}/{i}',is_correct=i%3!=0,excluded_reason=None)
            r['full_scores']={k:float(i%3) for k in ('nci','fdbd_all')}
            rows.append(s.seal_record(r))
        settings=dict(ks='all',dtype='float32',attention='sdpa',implementation='x',model='same',source_rows=list(range(30)))
        (p/'manifest.json').write_text(json.dumps(dict(complete=True,settings=settings,source_generation={'config':config})))
        (p/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))


def test_analysis_no_leakage_and_frozen_thresholds(tmp_path):
    root=tmp_path/'replay';fixture(root)
    args=Namespace(replays=root,output=tmp_path/'analysis',fpr=.1,step_quantile=.1,bootstrap=12,seed=42)
    s.analyze(args)
    reports=s.load_json(args.output/'metrics.json')
    assert len(reports)==14
    assert all(r['auroc']==1 for r in reports if r['rule']=='worst_step')
    protocol=s.load_json(args.output/'protocol.json')
    for c in protocol['calibration'].values(): assert c['correct_validation_fpr']<=.1
    # Test perturbation cannot change train/validation thresholds.
    path=root/'test/predictions.jsonl';rows=s.jsonl(path)
    for r in rows:
        r.pop('record_hash')
        for step in r['step_scores']:
            step['scores']={k:v+100 for k,v in step['scores'].items()}
    path.write_text(''.join(json.dumps(s.seal_record(r))+'\n' for r in rows))
    args.output=tmp_path/'other';s.analyze(args)
    other=s.load_json(args.output/'protocol.json')
    assert other['calibration']==protocol['calibration']
    assert other['train_step_thresholds']==protocol['train_step_thresholds']
    # Leakage is checked before fitting or saving results.
    rows[0]['question_hash']='train/0';rows[0].pop('record_hash',None)
    path.write_text(''.join(json.dumps(s.seal_record(r))+'\n' for r in rows))
    args.output=tmp_path/'leak'
    with pytest.raises(ValueError,match='leakage'):
        s.analyze(args)


def test_replay_manifest_resume_and_source_mutation(tmp_path,monkeypatch):
    from ood_repro import cli
    source=tmp_path/'source';cell=source/'gsm8k/test';(cell/'merged').mkdir(parents=True)
    modelpath=tmp_path/'checkpoint';modelpath.mkdir();(modelpath/'config.json').write_text('{}')
    config={'model':str(modelpath),'model_config':s.file_hash(modelpath/'config.json')}
    manifest={'complete':False,'n':2,'config':config}
    (cell/'generation.json').write_text(json.dumps(manifest))
    args=Namespace(source=source,output=tmp_path/'output',task='gsm8k',split='test',limit=0,
                   model=str(modelpath),ks='all',dtype='float32',attention='eager',resume=True)
    with pytest.raises(ValueError,match='not complete'):
        s.replay_source(args)
    manifest['complete']=True;(cell/'generation.json').write_text(json.dumps(manifest))
    for i in range(2):
        original={'id':i,'question':f'Question {i}','input_ids':[1,2],'full_seq_ids':[1,2,3],
                  'produced_text':'Step 1: x\n#### 1','is_correct':True,'truncated':False}
        (cell/f'merged/gsm8k_{i}.json').write_text(json.dumps(original))
    head=SimpleNamespace(weight=torch.randn(5,3),bias=None)
    model=SimpleNamespace(get_output_embeddings=lambda:head)
    calls=[]
    monkeypatch.setattr(cli,'load_model',lambda _: (model,None))
    monkeypatch.setattr(cli,'environment',lambda *_: {'packages':{},'model_commit':None,'tokenizer_commit':None,'chat_template_hash':'same'})
    def fake_score(*a):
        calls.append(1)
        return row()
    monkeypatch.setattr(s,'score_response',fake_score)
    s.replay_source(args)
    assert len(calls)==2 and s.load_json(args.output/'manifest.json')['complete']
    s.replay_source(args)
    assert len(calls)==2
    path=cell/'merged/gsm8k_0.json';r=s.load_json(path);r['question']='changed';path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='mismatch'):
        s.replay_source(args)
