"""Capture auditable progress without reading live checkpoints or changing training."""
import json
from datetime import datetime
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper.formal.common import ROOT, save, source_snapshot, digest


def main():
    study = ROOT/'results/formal_v3/study'
    out = ROOT/'results/formal_v3/progress_snapshot'
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((study/'frozen_protocol.json').read_text(encoding='utf-8'))
    if frozen['source'] != source_snapshot():
        raise RuntimeError('Frozen source mismatch: do not publish as the running study')
    status = json.loads((study/'status.json').read_text(encoding='utf-8'))
    runs = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    histories = sorted(study.rglob('history.json'))
    for path in histories:
        history = json.loads(path.read_text(encoding='utf-8'))
        if not history:
            continue
        relative = path.parent.relative_to(study).as_posix()
        destination = out/'runs'/relative
        save(destination/'history.json', history)
        for name in ['contract.json', 'settings.json', 'result.json']:
            source = path.parent/name
            if source.exists():
                save(destination/name, json.loads(source.read_text(encoding='utf-8')))
        best = min(history, key=lambda r: r['validation_mae'])
        runs.append(dict(run=relative, completed_epochs=len(history),
                         best_validation_mae=best['validation_mae'], best_epoch=best['epoch'],
                         completed=(path.parent/'result.json').exists()))
        if len(histories) <= 12:
            ax.plot([r['epoch'] for r in history], [r['validation_mae'] for r in history], label=relative)
    ax.set(xlabel='Completed epoch', ylabel='Validation MAE (native speed unit)',
           title='Validation progress only; no superiority conclusion')
    ax.grid(alpha=.25)
    if runs and len(histories) <= 12:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/'validation_progress.png', dpi=160)
    plt.close(fig)
    summary = dict(captured_at=datetime.now().astimezone().isoformat(), coordinator_status=status,
                   frozen_source_verified=True, config_sha256=digest(ROOT/'experiments/formal_v3_study.json'),
                   runs=runs, disk_free_gib=round(shutil.disk_usage(ROOT).free/1024**3, 2),
                   note='Files are individually captured during live training, not a simultaneous checkpoint; no weights included.')
    save(out/'snapshot.json', summary)
    lines = ['# 第三版正式训练进度快照', '', '采集时间：'+summary['captured_at'], '',
             '这是运行中验证结果的快照，不能作为最优参数或模型优越性的结论。', '',
             '| 运行 | 已完成轮数 | 当前最佳验证MAE | 最佳轮次 | 该运行完成 |',
             '|---|---:|---:|---:|---|']
    for row in runs:
        lines.append(f"| {row['run']} | {row['completed_epochs']} | {row['best_validation_mae']:.6f} | {row['best_epoch']} | {row['completed']} |")
    lines.extend(['', '验证MAE按完整12个预测时域计算，单位沿用原始速度数据；不等同于论文单时域表值。', '',
                  '训练代码哈希已与冻结协议逐文件核对。实时结果目录：`F:\\动态因果\\results\\formal_v3\\study`。', '',
                  '真实Q原始缺失记录、实测流量、独立密度、转向与边界计数，以及可核验干预事件仍未齐备。',
                  '天气累计降水解析已经完成，但机场站到道路的空间匹配与完整降雨标签仍未完成。'])
    (out/'进度说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
