"""Build and verify an additive evidence snapshot; never include live checkpoints."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import zipfile
from paper.formal.common import ROOT, digest, source_snapshot


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT/'results/formal_v3/delivery'/stamp
    out.mkdir(parents=True, exist_ok=False)
    study = ROOT/'results/formal_v3/study'
    frozen = json.loads((study/'frozen_protocol.json').read_text(encoding='utf-8'))
    if source_snapshot() != frozen['source']:
        raise RuntimeError('Source differs from frozen protocol')
    selected = set()
    for directory in ['src', 'third_party/official', 'data/evidence_v3', 'results/formal_v3/preflight',
                      'results/formal_v3/progress_snapshot']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                selected.add(path)
    for pattern in ['*formal*v3*', '*weather*evidence*', '*weather*quality*v3*', '*real_evidence*v3*']:
        selected.update(p for p in (ROOT/'experiments').glob(pattern) if p.is_file())
    for relative in ['docs/05_真实证据与正式训练_v3.md', 'results/formal_v3/tests.log',
                     'results/formal_v3/requirements-lock.txt', 'results/formal_v3/evidence_readiness.json',
                     'results/formal_v3/study/frozen_protocol.json',
                     'data/formal_v3/metr-la/manifest.json', 'data/formal_v3/pems-bay/manifest.json']:
        selected.add(ROOT/relative)
    for path in (ROOT/'data/formal_v3/metr-la').glob('pdformer_train_features.*'):
        selected.add(path)
    archive = out/'真实证据与正式训练_v3_阶段快照.zip'
    manifest = {}
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for path in sorted(selected):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest[relative] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            z.writestr(relative, data)
        z.writestr('文件校验清单.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as z:
        for name, expected in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected['sha256']:
                raise RuntimeError('Archive verification failed: '+name)
        for name, expected in frozen['source'].items():
            if manifest[name.replace('\\', '/')]['sha256'] != expected:
                raise RuntimeError('Packaged source differs: '+name)
    readme = '''# 第三版真实证据与正式训练：阶段交付

此包包含冻结训练源码、五个官方基线仓库及许可证、依赖清单、数据哈希、NOAA原始天气和质量解析、10项测试记录、6种模型GPU检查记录、验证曲线与输入契约。

这是训练过程中的可核验快照，尚非完整论文复现。公开速度基准的正式参数搜索和五种子训练仍在执行。最佳参数及模型优越性只能在相应搜索、测试和统计比较完成后报告。

压缩包逐文件SHA256清单位于包内，外部 verification.json 记录整包哈希。解压后先阅读 docs/05_真实证据与正式训练_v3.md 和 results/formal_v3/progress_snapshot/进度说明.md。

实时训练目录：F:\\动态因果\\results\\formal_v3\\study。快照不包含运行中权重或完整基准数据副本；原始数据位置及哈希见 data/formal_v3 中的清单。脚本含当前机器绝对路径，迁移时需要调整路径和安装依赖，不能直接从压缩包启动训练。

真实Q补值前数据与缺失掩码、实测流量、独立密度、历史POI、网络边界/转向/初始状态以及独立干预事件仍需补充。北京机场天气站不能直接替代全部道路天气。合成数据测试仅证明代码行为，不证明真实效果。
'''
    (out/'交付说明.md').write_text(readme, encoding='utf-8')
    verification = dict(created_at=datetime.now().astimezone().isoformat(), archive=archive.name,
                        archive_sha256=digest(archive), archive_bytes=archive.stat().st_size,
                        verified_entries=len(manifest), frozen_source_verified=True,
                        complete_paper_reproduction=False)
    (out/'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(directory=str(out), **verification), ensure_ascii=False))


if __name__ == '__main__':
    main()
