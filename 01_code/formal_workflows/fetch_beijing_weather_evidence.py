"""Archive official station metadata and 2017 observations; never invent weather labels."""
import csv
from datetime import datetime,timezone,timedelta
import io
import json
from pathlib import Path
import urllib.request
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/evidence_v3/weather';OUT.mkdir(parents=True,exist_ok=True)

def download(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=45) as r:
            data=r.read()
        if data.lstrip().startswith(b'<'):raise ValueError('Expected CSV, received HTML')
        path.write_bytes(data)
    data=path.read_bytes()
    return dict(url=url,path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)),data

history_url='https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
record,raw=download(history_url,OUT/'isd-history.csv')
stations=[]
for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))):
    try:
        if 116.10<=float(r['LON'])<=116.71 and 39.69<=float(r['LAT'])<=40.18 and r['BEGIN']<='20170401' and r['END']>='20170531':
            stations.append(r)
    except (ValueError,KeyError):continue
records=[]
for s in sorted(stations,key=lambda r:r['USAF'])[:3]:
    code=s['USAF']+s['WBAN'];url=f'https://www.ncei.noaa.gov/data/global-hourly/access/2017/{code}.csv'
    try:
        provenance,data=download(url,OUT/(code+'_2017.csv'))
        rows=[]
        for row in csv.DictReader(io.StringIO(data.decode('utf-8-sig'))):
            t=datetime.fromisoformat(row['DATE']).replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
            if '2017-04-01'<=t.date().isoformat()<'2017-06-01':
                row['time_Asia_Shanghai']=t.isoformat();rows.append(row)
        (OUT/(code+'_Q_period.json')).write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
        records.append(dict(station=s,provenance=provenance,matched_observations=len(rows),
                            precipitation_records=sum(bool(r.get('AA1')) for r in rows),
                            status='downloaded_raw_quality_and_spatial_matching_pending'))
    except Exception as e:
        records.append(dict(station=s,url=url,status='download_failed',error=str(e)))
manifest=dict(retrieved_at=datetime.now(timezone.utc).isoformat(),station_inventory=record,stations=records,
              limitations=['Quality codes and accumulation periods must be decoded before labelling rain',
                           'Station observation does not establish weather at every road segment',
                           'Do not use future weather observations as available forecast inputs'])
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
if not records or all(r['status']=='download_failed' for r in records):raise SystemExit(2)
