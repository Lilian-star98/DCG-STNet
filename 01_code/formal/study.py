"""Sequential GPU search and five-seed final evaluation, with a frozen protocol."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import shutil
from .common import ROOT,digest,save,source_snapshot


def launch(data,out,model,settings,seed,phase):
    if shutil.disk_usage(out.parent if out.parent.exists() else ROOT).free < 3*1024**3:
        raise RuntimeError('Less than 3 GiB free; preserve checkpoints and free space before resuming')
    out.mkdir(parents=True,exist_ok=True);save(out/'settings.json',settings)
    command=[sys.executable,'-m','paper.formal.train','--data',str(data),'--out',str(out),
             '--model',model,'--settings',str(out/'settings.json'),'--seed',str(seed),'--phase',phase]
    # Each run gets a fresh process: official packages use conflicting module names.
    with (out/'console.log').open('a',encoding='utf-8') as log:
        p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Run failed ({p.returncode}); see {out}/console.log')
    return json.loads((out/'result.json').read_text(encoding='utf-8'))


def study(config,out):
    config=Path(config);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(config.read_text(encoding='utf-8'))
    frozen=dict(plan=plan,source=source_snapshot(),config_sha256=digest(config))
    if (out/'frozen_protocol.json').exists():
        if json.loads((out/'frozen_protocol.json').read_text(encoding='utf-8'))!=frozen:
            raise ValueError('Protocol/source changed: use a new study version')
    else:save(out/'frozen_protocol.json',frozen)
    trials=[];final=[];selections=[];started=time.time()
    for dataset in plan['datasets']:
        data=ROOT/'data/formal_v3'/dataset
        for model,grid in plan['models'].items():
            group=[]
            for i,parameters in enumerate(grid):
                settings=dict(plan['training'],**parameters)
                target=out/dataset/model/'search'/f'trial_{i}'
                save(out/'status.json',dict(state='running',phase='search',dataset=dataset,model=model,trial=i,started=started))
                r=launch(data,target,model,settings,plan['search_seed'],'search')
                trials.append(dict(dataset=dataset,path=str(target),**r));group.append((r['validation_mae'],i,settings))
                save(out/'search_results.json',trials)
            score,index,best=min(group,key=lambda x:(x[0],x[1]))
            selection=dict(dataset=dataset,model=model,settings=best,validation_mae=score,trial=index,
                           scope='best validation MAE among prespecified completed configurations, not a global optimum')
            selections.append(selection);save(out/'selected_parameters.json',selections)
            for seed in plan['final_seeds']:
                target=out/dataset/model/'final'/f'seed_{seed}'
                save(out/'status.json',dict(state='running',phase='final',dataset=dataset,model=model,seed=seed,started=started))
                r=launch(data,target,model,best,seed,'final')
                final.append(dict(dataset=dataset,path=str(target),**r));save(out/'final_results.json',final)
        full=next(s for s in selections if s['dataset']==dataset and s['model']=='STCausalNet')
        for variant in plan.get('prediction_ablations',[]):
            name='STCausalNet:'+variant;safe=variant.replace('/','_').replace(' ','_')
            for seed in plan['final_seeds']:
                target=out/dataset/'ablations'/safe/f'seed_{seed}'
                save(out/'status.json',dict(state='running',phase='ablation',dataset=dataset,model=name,seed=seed,started=started))
                r=launch(data,target,name,full['settings'],seed,'final')
                final.append(dict(dataset=dataset,path=str(target),**r));save(out/'final_results.json',final)
    from .report import report
    report(out)
    save(out/'status.json',dict(state='completed_available_benchmark_subset',started=started,finished=time.time(),
                               paper_complete=False,real_Q_and_interventions='blocked_by_missing_evidence'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    try:study(a.config,a.out)
    except Exception as e:
        save(a.out/'failure.json',dict(error=str(e),time=time.time()));raise

if __name__=='__main__':main()
