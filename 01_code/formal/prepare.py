"""Prepare measured speed benchmarks without using validation/test statistics.

Targets retain missingness. Zero is the published benchmark missing-value
convention, not a general assertion that stationary traffic cannot occur.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, digest, save


def causal_fill(raw, train_end):
    valid = np.isfinite(raw) & (raw > 0)
    count = valid[:train_end].sum(0)
    if (count == 0).any():
        raise ValueError('A sensor has no observed training samples')
    mean = np.where(valid[:train_end], raw[:train_end], 0).sum(0)/count
    var = np.where(valid[:train_end], (raw[:train_end]-mean)**2, 0).sum(0)/count
    std = np.maximum(np.sqrt(var), 1e-3)
    filled = np.empty_like(raw, dtype=np.float32)
    last = mean.copy()
    for t in range(len(raw)):
        last = np.where(valid[t], raw[t], last)
        filled[t] = last
    return filled, valid, mean.astype('float32'), std.astype('float32')


def prepare(name, out):
    label = {'metr-la':'METR-LA','pems-bay':'PEMS-BAY'}[name]
    directory = ROOT/'raw_data'/label
    h5 = directory/(label+'.h5'); graph = directory/('adj_'+label+'.pkl')
    df = pd.read_hdf(h5)
    if not isinstance(df.index,pd.DatetimeIndex) or not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError('Unique chronological datetime index required')
    # Never compress missing dates into adjacent time steps.
    grid = pd.date_range(df.index[0],df.index[-1],freq='5min')
    missing_times = len(grid.difference(df.index))
    df = df.reindex(grid)
    raw = df.to_numpy(dtype=np.float32)[...,None]
    with graph.open('rb') as f:
        ids, idmap, A = pickle.load(f,encoding='latin1')
    labels = [str(x) for x in df.columns]; ids = [str(x) for x in ids]
    if set(labels)!=set(ids) or len(labels)!=len(ids):
        raise ValueError('Graph/sensor IDs disagree')
    order = [ids.index(x) for x in labels]
    A = np.asarray(A,dtype=np.float32)[np.ix_(order,order)]
    if not np.isfinite(A).all() or (A<0).any():raise ValueError('Invalid adjacency')
    T,N,F = raw.shape; train_end = int(T*.6); val_end=train_end+int(T*.2)
    filled,valid,mean,std = causal_fill(raw,train_end)
    tod = (grid.hour*60+grid.minute).to_numpy()/1440
    dow = grid.dayofweek.to_numpy()
    zg = np.stack([np.sin(2*np.pi*tod),np.cos(2*np.pi*tod),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7),dow>=5],-1).astype('float32')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    arrays=dict(X=filled,truth=raw,observed=valid,A=A,mean=mean,std=std,Z_global=zg,
                timestamps=grid.to_numpy(dtype='datetime64[ns]').astype('int64'),sensor_ids=np.array(labels),
                split=np.array([0,train_end,val_end,T]),tod=tod.astype('float32'),dow=dow)
    for key,a in arrays.items():np.save(out/(key+'.npy'),a,allow_pickle=False)
    manifest=dict(dataset=name,evidence='measured_speed_benchmark',sample_minutes=5,
                  shape=list(raw.shape),units='native benchmark speed; verify mph/kmh before physical use',
                  flow_available=False,events_available=False,physics_calibrated=False,
                  split=[0,train_end,val_end,T],missing_fraction=float(1-valid.mean()),missing_timestamps=missing_times,
                  missing_policy='nonfinite/zero targets masked; history forward-fill, initial training mean',
                  normalization='observed training samples only, per sensor',
                  timestamp_policy='source local clock retained; no unverified UTC conversion',
                  adjacency='published sensor distance weights; NOT CTM turning fractions',
                  provenance_url='https://github.com/liyaguang/DCRNN',
                  raw_files={str(h5.resolve()):digest(h5),str(graph.resolve()):digest(graph)},
                  files={p.name:digest(p) for p in out.glob('*.npy')})
    save(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['dataset','shape','missing_fraction','missing_timestamps']},ensure_ascii=False))
    return manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['metr-la','pems-bay'],required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();prepare(a.dataset,a.out)

if __name__=='__main__':main()
