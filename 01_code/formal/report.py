"""Regenerate every comparison from computed runs; never fill missing results."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel
from .common import ROOT,save


def bh(pvalues):
    p=np.asarray(pvalues,float);order=np.argsort(p);out=np.empty_like(p)
    q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out[order]=np.minimum(q,1);return out


def report(study):
    study=Path(study)
    rows=json.loads((study/'final_results.json').read_text(encoding='utf-8')) if (study/'final_results.json').exists() else []
    original=json.loads((ROOT/'src/paper/tables/paper_reported.json').read_text(encoding='utf-8'))
    summary=[];comparisons=[];mappings={'metr-la':'METR-LA','pems-bay':'PEMS-BAY'}
    for dataset in mappings:
        ds=[r for r in rows if r['dataset']==dataset and r.get('test_evaluated')]
        models=sorted({r['model'] for r in ds})
        for model in models:
            mr=[r for r in ds if r['model']==model]
            for minute in ['15','30','60']:
                for metric in ['MAE','RMSE','MAPE_percent']:
                    vals=[r['test'][minute][metric] for r in mr]
                    summary.append(dict(dataset=dataset,model=model,horizon_minutes=int(minute),metric=metric,
                         mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)) if len(vals)>1 else None,
                         seeds=[r['seed'] for r in mr],runs=[r['path'] for r in mr]))
        competitors=[m for m in models if not m.startswith('STCausalNet')]
        if 'STCausalNet' in models and competitors:
            ref=min(competitors,key=lambda m:np.mean([r['validation_mae'] for r in ds if r['model']==m]))
            a={r['seed']:r for r in ds if r['model']=='STCausalNet'};b={r['seed']:r for r in ds if r['model']==ref}
            seeds=sorted(a.keys()&b.keys())
            for minute in ['15','30','60']:
                if len(seeds)<5:continue
                x=np.array([a[s]['test'][minute]['MAE'] for s in seeds]);y=np.array([b[s]['test'][minute]['MAE'] for s in seeds])
                p=float(ttest_rel(x,y).pvalue)
                if not np.isfinite(p):p=1. if np.allclose(x,y) else 0.
                comparisons.append(dict(dataset=dataset,horizon_minutes=int(minute),reference=ref,seeds=seeds,
                     improvement_percent=float(100*(y.mean()-x.mean())/y.mean()),p_value=p))
    if comparisons:
        for row,q in zip(comparisons,bh([r['p_value'] for r in comparisons])):
            row['BH_q']=float(q);row['supports_lower_MAE']=bool(q<.05 and row['improvement_percent']>0)
    cells=[]
    for row in original['table1_in_distribution']['rows']:
        for ds,label in mappings.items():
            for i,key in enumerate(['MAE','RMSE','MAPE_percent']):
                found=[r for r in summary if r['dataset']==ds and r['model']==row['model'] and r['horizon_minutes']==60 and r['metric']==key]
                cells.append(dict(table=1,dataset=ds,model=row['model'],metric=key,reported=row[label][i],
                     computed=found[0]['mean'] if found else None,
                     delta=found[0]['mean']-row[label][i] if found else None,
                     status='computed_under_documented_unified_protocol' if found else 'not_run',
                     protocol_equivalence_confirmed=False))
    # Preserve all original values, explicitly blocked instead of substituting smoke data.
    for key,table in original.items():
        if key=='table1_in_distribution':continue
        if key=='table3_ablation_pred':
            for row in table['rows']:
                model='STCausalNet' if row['variant']=='DCG-STNet (full)' else 'STCausalNet:'+row['variant']
                for label,ds in [('METR-LA','metr-la'),('Q-Traffic','q')]:
                    for i,metric in enumerate(['MAE','RMSE']):
                        found=[r for r in summary if r['dataset']==ds and r['model']==model and r['horizon_minutes']==60 and r['metric']==metric]
                        cells.append(dict(table=3,dataset=ds,model=model,metric=metric,reported=row[label][i],
                             computed=found[0]['mean'] if found else None,
                             status='computed_forecast_variant_without_physics' if found else 'not_run',protocol_equivalence_confirmed=False))
            continue
        cells.append(dict(table=key,reported_rows=table['rows'],computed=None,
                          status='pending_real_event_inputs_or_compatible_ablation_and_counterfactual_runs'))
    save(study/'computed_summary.json',summary);save(study/'significance.json',comparisons)
    save(study/'paper_table_audit.json',dict(cells=cells,findings=[
        'Original Table 1 METR-LA full-model MAE=2.72 vs Table 3 full-model MAE=3.45; protocol difference unexplained',
        'A forecast adapter without calibrated physics cannot substantiate physical or causal superiority',
        'No claim of superiority is made before complete comparator runs and corrected tests']))
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',type=Path,required=True);a=ap.parse_args();report(a.study)

if __name__=='__main__':main()
