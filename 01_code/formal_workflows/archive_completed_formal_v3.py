"""Archive completed runs only, with explicit partial-study reporting."""
import hashlib
import json
import zipfile
from datetime import datetime
from paper.formal.common import ROOT, digest, save, source_snapshot
from paper.formal.report import report


def main():
    study = ROOT/'results/formal_v3/study'
    frozen = json.loads((study/'frozen_protocol.json').read_text(encoding='utf-8'))
    assert frozen['source'] == source_snapshot(), 'Source mismatch'
    assert frozen['config_sha256'] == digest(ROOT/'experiments/formal_v3_study.json'), 'Config mismatch'
    rows = json.loads((study/'final_results.json').read_text(encoding='utf-8'))
    selections = json.loads((study/'selected_parameters.json').read_text(encoding='utf-8'))
    contents = {}
    for row in rows:
        run = ROOT/row['path']
        run.resolve().relative_to(study.resolve())
        result = json.loads((run/'result.json').read_text(encoding='utf-8'))
        assert all(row[k] == v for k, v in result.items()), 'Result index differs from run'
        contract = json.loads((run/'contract.json').read_text(encoding='utf-8'))
        assert contract['sources'] == frozen['source'], 'Run source mismatch'
        if row['model'] == 'STCausalNet':
            selection = next(s for s in selections if s['dataset'] == row['dataset'] and s['model'] == row['model'])
            assert selection['settings'] == row['settings'], 'Selected configuration mismatch'
        assert row['test_evaluated'] and row['phase'] == 'final'
        for name in ['result.json', 'contract.json', 'settings.json', 'history.json', 'console.log']:
            contents[(run/name).relative_to(ROOT).as_posix()] = (run/name).read_bytes()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT/'results/formal_v3/delivery'/stamp
    out.mkdir(parents=True, exist_ok=False)
    provisional = out/'partial_report'
    save(provisional/'final_results.json', rows)
    summary = report(provisional)
    lines = ['# 正式测试增量归档（未完成全矩阵）', '',
             '采集时间：'+datetime.now().astimezone().isoformat(), '',
             f'本次保存{len(rows)}个已完成的正式运行。均核对原始结果、冻结源码和已选参数。', '',
             '以下汇总当前已完成种子；是否完成五种子以各行种子列表为准。全实验矩阵尚未完成，优越性结论须等待完整基线比较。', '',
             '| 数据集 | 模型 | 时域/分钟 | 种子 | MAE均值 | 种子间样本标准差 |',
             '|---|---|---:|---|---:|---:|']
    for row in summary:
        if row['metric'] == 'MAE':
            sd = f"{row['std']:.6f}" if row['std'] is not None else '不适用'
            lines.append(f"| {row['dataset']} | {row['model']} | {row['horizon_minutes']} | {row['seeds']} | {row['mean']:.6f} | {sd} |")
    lines += ['', '物理标定未完成。当前主模型为公开速度预测适配版本；论文表值对照的协议等价性尚未确认。',
              '真实Q流量、独立密度、路网标定输入与独立干预记录仍缺失。',
              '包中只包含已完成运行的指标、配置、训练记录和临时核对报告，不包含运行中的权重。',
              '实时结果目录：F:\\动态因果\\results\\formal_v3\\study。此前交付保持不变。']
    readme = '\n'.join(lines)+'\n'
    (out/'交付说明.md').write_text(readme, encoding='utf-8')
    for path in provisional.glob('*.json'):
        contents['partial_report/'+path.name] = path.read_bytes()
    for name in ['frozen_protocol.json', 'selected_parameters.json']:
        contents['study_protocol/'+name] = (study/name).read_bytes()
    contents['交付说明.md'] = readme.encode('utf-8')
    archive = out/'已完成正式测试_增量快照.zip'
    manifest = {name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name,data in contents.items():
            z.writestr(name,data)
        z.writestr('文件校验清单.json',json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert all(hashlib.sha256(z.read(name)).hexdigest() == value for name,value in manifest.items())
    save(out/'verification.json', dict(archive=archive.name,archive_sha256=digest(archive),
         verified_entries=len(contents),completed_final_runs=len(rows),partial_study=True,
         complete_paper_reproduction=False,frozen_source_verified=True))
    print(json.dumps(dict(snapshot=stamp,completed_final_runs=len(rows),verified_entries=len(contents)),ensure_ascii=False))


if __name__ == '__main__':
    main()
