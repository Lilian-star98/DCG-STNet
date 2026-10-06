"""Write factual readiness states, preserving unresolved scientific evidence."""
import json
from pathlib import Path
from paper.formal.common import ROOT,digest,save,repository

out=ROOT/'results/formal_v3';out.mkdir(parents=True,exist_ok=True)
weather=ROOT/'data/evidence_v3/weather/manifest.json'
qscript=ROOT/'Q/around_traffic_mv_avg_15min_1km_completion.py'
result={
 'measured_benchmarks':{name:json.loads((ROOT/'data/formal_v3'/name/'manifest.json').read_text(encoding='utf-8')) for name in ['metr-la','pems-bay']},
 'weather':json.loads(weather.read_text(encoding='utf-8')) if weather.exists() else {'status':'unavailable'},
 'weather_quality':json.loads((weather.parent/'quality_audit.json').read_text(encoding='utf-8')) if (weather.parent/'quality_audit.json').exists() else {'status':'pending'},
 'official_sources':{name:repository(ROOT/'third_party/official'/name) for name in ['Graph-WaveNet','D2STGNN','CaST','PDFormer','STAEformer']},
 'gates':{
  'Q_raw_missingness':{'status':'blocked','reason':'Available completion script computes weekly statistics across all 61 days; raw pre-imputation speed and mask not found','script':str(qscript),'sha256':digest(qscript)},
  'Q_measured_flow':{'status':'blocked','reason':'No matched vehicle-count source or sampling/units provenance identified; speed proxy forbidden'},
  'real_event_labels':{'status':'blocked','reason':'Existing link/time lists lack verified accident/weather/school category, announcement and source lineage'},
  'historic_school_POI':{'status':'blocked','reason':'No verified 2017 school entrances, opening calendar and road mapping identified'},
  'network_calibration':{'status':'blocked','reason':'Missing aligned density/flow, lane/length metadata, turns, boundary arrivals and initial state'},
  'independent_intervention':{'status':'blocked','reason':'No documented held-out treatment episodes and uncontaminated donor outcomes'},
  'paper_tables':{'status':'pending_training','reason':'Existing manuscript values are reference text, never computed evidence'}
 },
 'search_scope':'Existing authorized project folders; absence here does not imply the records do not exist elsewhere',
 'paper_complete':False
}
save(out/'evidence_readiness.json',result)
print(json.dumps({'blocked_gates':list(result['gates']),'weather_available':weather.exists()},ensure_ascii=False))
