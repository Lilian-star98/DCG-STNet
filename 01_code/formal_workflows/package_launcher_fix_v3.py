"""Package only the verified launcher repair; preserve the previous snapshot."""
import hashlib
import json
import zipfile
from datetime import datetime
from paper.formal.common import ROOT, digest, source_snapshot

frozen = json.loads((ROOT/'results/formal_v3/study/frozen_protocol.json').read_text(encoding='utf-8'))
assert frozen['source'] == source_snapshot(), 'Frozen source mismatch'
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out = ROOT/'results/formal_v3/delivery'/stamp
out.mkdir(parents=True, exist_ok=False)
archive = out/'第三版续训启动器修复.zip'
files = ['experiments/start_formal_v3.ps1', 'docs/06_续训启动器修复_v3.md']
manifest = {}
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in files:
        content = (ROOT/name).read_bytes()
        manifest[name] = hashlib.sha256(content).hexdigest()
        z.writestr(name, content)
    z.writestr('文件校验清单.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(archive) as z:
    assert all(hashlib.sha256(z.read(name)).hexdigest() == value for name, value in manifest.items())
verification = dict(created_at=datetime.now().astimezone().isoformat(), archive=archive.name,
                    archive_sha256=digest(archive), verified_entries=len(files),
                    frozen_source_verified=True, complete_paper_reproduction=False)
(out/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
(out/'交付说明.md').write_text((ROOT/files[1]).read_text(encoding='utf-8'), encoding='utf-8')
print(stamp)
