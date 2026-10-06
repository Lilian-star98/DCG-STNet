from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def source_snapshot():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'src').rglob('*.py'))}


def environment():
    import torch
    return dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
                cuda=torch.version.cuda, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def repository(path):
    p = subprocess.run(['git','-c','safe.directory='+str(Path(path).resolve()).replace('\\','/'),'-C',str(path),'rev-parse','HEAD'],
                       capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    return dict(commit=p.stdout.strip(), path=str(path),
                files={str(f.relative_to(path)):digest(f) for f in Path(path).rglob('*.py') if '.git' not in f.parts})
