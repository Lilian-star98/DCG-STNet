"""Held-out event assessment with pre-period-only synthetic-control fitting.

Estimated counterfactuals are assumption-dependent references, never observed
ground truth. Event-level resampling, not timestamp-level pseudo replication.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .common import digest,save


def event_effect(treated,controls,pre,post):
    treated=np.asarray(treated,float);controls=np.asarray(controls,float)
    pre=np.asarray(pre,bool);post=np.asarray(post,bool)
    if pre.shape!=treated.shape or post.shape!=treated.shape or (pre&post).any():raise ValueError('Invalid pre/post windows')
    if controls.ndim!=2 or len(controls)!=len(treated):raise ValueError('Invalid controls')
    if not np.isfinite(treated).all() or not np.isfinite(controls).all():raise ValueError('Missing outcome data')
    if min(pre.sum(),post.sum())<4:raise ValueError('Insufficient pre/post observations')
    K=controls.shape[1]
    if K<2:raise ValueError('At least two pre-selected uncontaminated donors required')
    # Convex weights use pre-period only; no post-period donor tuning.
    result=minimize(lambda w:np.mean((controls[pre]@w-treated[pre])**2),np.ones(K)/K,
                    bounds=[(0,1)]*K,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],method='SLSQP')
    if not result.success:raise RuntimeError(result.message)
    reference=controls@result.x;gap=treated-reference
    baseline=float(gap[pre].mean());effect=float(gap[post].mean()-baseline)
    first,second=np.array_split(np.flatnonzero(pre),2)
    return dict(estimated_ATT=effect,pre_RMSE=float(np.sqrt(np.mean(gap[pre]**2))),
                pre_drift=float(gap[second].mean()-gap[first].mean()),donor_weights=result.x.tolist(),
                counterfactual=reference+baseline,estimated_effect_series=treated-reference-baseline)


def evaluate(npz,metadata,out):
    meta=json.loads(Path(metadata).read_text(encoding='utf-8'))
    if set(meta['training_event_ids'])&set(meta['test_event_ids']):raise ValueError('Event identity leakage')
    events=meta['events']
    if [e['event_id'] for e in events]!=meta['test_event_ids']:raise ValueError('Test order must match arrays')
    for e in events:
        for key in ['source','announcement_time','start_time','end_time','treated_road_ids','donor_road_ids','donor_exclusion_rationale','independent_cluster_id']:
            if not e.get(key):raise ValueError('Missing event evidence: '+key)
        if set(e['treated_road_ids'])&set(e['donor_road_ids']):raise ValueError('Treatment/control overlap')
    with np.load(npz,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    rows=[];refs=[]
    for i,e in enumerate(events):
        time=d['timestamps'][i].astype('datetime64[ns]')
        start=np.datetime64(e['start_time']);end=np.datetime64(e['end_time'])
        if end<=start or np.any(time[d['pre'][i].astype(bool)]>=start) or np.any(time[d['post'][i].astype(bool)]<start):
            raise ValueError('Pre/post windows disagree with documented intervention time')
        r=event_effect(d['treated'][i],d['controls'][i],d['pre'][i],d['post'][i])
        refs.append(r.pop('counterfactual'));r.pop('estimated_effect_series')
        r.update(event_id=e['event_id'],evidence='observational_estimate_under_identification_assumptions')
        if 'model_counterfactual' in d:
            post=d['post'][i].astype(bool)
            r['model_ATT']=float(np.mean((d['treated'][i]-d['model_counterfactual'][i])[post]))
            r['ATT_disagreement']=abs(r['model_ATT']-r['estimated_ATT'])
        rows.append(r)
    vals=np.array([r['estimated_ATT'] for r in rows]);rng=np.random.default_rng(0)
    labels=[e['independent_cluster_id'] for e in events];clusters=sorted(set(labels))
    grouped=[vals[np.array(labels)==c] for c in clusters]
    sampled=[float(np.concatenate([grouped[j] for j in rng.integers(0,len(grouped),len(grouped))]).mean()) for _ in range(2000)] if len(clusters)>=5 else []
    ci=np.quantile(sampled,[.025,.975]).tolist() if sampled else None
    save(Path(out)/'events.json',dict(events=rows,mean_estimated_ATT=float(vals.mean()),cluster_bootstrap_95=ci,independent_clusters=len(clusters),
         data_sha256=digest(npz),identification_verified=False,
         remaining=['Review parallel trends, anticipation, spillovers and concurrent events',
                    'Pre-fit quality thresholds and placebo tests must be specified before inspecting test effects']))
    np.savez_compressed(Path(out)/'reference_counterfactuals.npz',reference=np.stack(refs))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();evaluate(a.data,a.metadata,a.out)

if __name__=='__main__':main()
