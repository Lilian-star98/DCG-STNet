"""Decode precipitation intervals using the NOAA ISD AA1-AA4 specification."""
import json
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
import hashlib

root=Path(__file__).resolve().parents[1]/'data/evidence_v3/weather'
source=root/'54511099999_Q_period.json'
rows=json.loads(source.read_text(encoding='utf-8'));output=[];flags=Counter()
for row in rows:
    for key in ['AA1','AA2','AA3','AA4']:
        raw=row.get(key,'')
        if not raw:continue
        values=raw.split(',')
        if len(values)!=4:raise ValueError('Unexpected NOAA precipitation format')
        hours,amount,condition,quality=values;hours=int(hours);amount=int(amount)
        valid=0<hours<99 and amount<9999 and condition not in {'1','3','5','6','7','8','E','I','J'}
        qc='all_checks_passed' if quality in {'1','5'} else 'gross_limits_only' if quality in {'0','4','9'} else 'requires_review'
        end=datetime.fromisoformat(row['time_Asia_Shanghai'])
        value=dict(source_field=key,source_value=raw,interval_end=end.isoformat(),
                   interval_start=(end-timedelta(hours=hours)).isoformat() if hours<99 else None,
                   duration_hours=hours if hours<99 else None,depth_mm=amount/10 if amount<9999 else None,
                   trace=condition=='2',condition=condition,quality=quality,quality_tier=qc,
                   usable_under_strict_policy=valid and qc=='all_checks_passed')
        output.append(value);flags[qc]+=1
summary=dict(observation_records=len(rows),precipitation_fields=len(output),quality_counts=dict(flags),
             strict_usable_fields=sum(r['usable_under_strict_policy'] for r in output),
             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             specification='https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf',
             policy='strict quality codes 1/5; keep gross-limit observations separately; do not disaggregate accumulations into observed 15-minute rainfall',
             spatial_match_complete=False,complete_rain_labels=False,
             limitations=['AA fields include liquid equivalent and do not alone distinguish rain from snow',
                          'Missing AA fields are unknown precipitation, not evidence of dry weather',
                          'Overlapping accumulation intervals are not summed'])
(root/'precipitation_intervals.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'quality_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
