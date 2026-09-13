#!/usr/bin/env python3
"""Plot matching ROC, PR and FPR95 panels from the same per-cell predictions.

The default score source uses the verified 1-NN, no-TTA rerun configuration.
--preview explicitly uses the surviving replay source, not the formal result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from plot_rf_roc_comparison import ROOT, SOURCES, resolve_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--ours-scores', type=Path,
                        default=ROOT / 'analysis_outputs/20260911_inhouse_1nn_no_tta/fusion/scores')
    args = parser.parse_args()
    output = ROOT / 'analysis_outputs/20260911_inhouse_metric_triptych'
    grid = np.linspace(0, 1, 501)
    results = {}
    for source in SOURCES:
        root = resolve_root(source, 'inhouse')
        if source.label == 'SpectraMemAD' and not args.preview:
            root = args.ours_scores
        paths = sorted(root.glob('*.npz'))
        if len(paths) != source.expected_inhouse:
            raise RuntimeError(f'{source.label}: expected {source.expected_inhouse} score files in {root}, found {len(paths)}')
        rocs, prs, metrics = [], [], []
        for path in paths:
            with np.load(path, allow_pickle=True) as data:
                labels = np.asarray(data['labels']).reshape(-1)
                scores = np.asarray(data[source.key], dtype=float).reshape(-1)
            if labels.shape != scores.shape or not np.isfinite(scores).all():
                raise ValueError(f'Invalid predictions: {path}')
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(f'Expected binary labels with both classes: {path}')
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            roc = np.interp(grid, fpr, tpr)
            roc[0], roc[-1] = 0, 1
            rocs.append(roc)
            # Recall is returned in descending order; retain the rightmost
            # precision at duplicate recall values after reversing.
            prs.append(np.interp(grid, recall[::-1], precision[::-1]))
            metrics.append([roc_auc_score(labels, scores), average_precision_score(labels, scores),
                            fpr[np.flatnonzero(tpr >= .95)[0]]])
        values = np.mean(metrics, axis=0) * 100
        results[source.label] = dict(root=str(root), count=len(paths),
                                    auroc=float(values[0]), average_precision=float(values[1]),
                                    fpr95=float(values[2]),
                                    roc=np.mean(rocs, axis=0).tolist(),
                                    pr=np.mean(prs, axis=0).tolist())
    if not args.preview:
        run_root = args.ours_scores.parent.parent
        protocol = json.loads((run_root / 'run_protocol.json').read_text())
        vit = json.loads((run_root / 'vit/summary.json').read_text())
        if (protocol['nearest_neighbors'] != 1 or protocol['tta'] != 'none'
                or vit['nn_topk'] != 1 or vit['paired_tta'] != 'none'):
            raise RuntimeError('Expected the 1-NN, no-TTA experiment configuration.')

    # Same canvas and resolution as the existing three-shot figures.
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8.2,
                         'axes.titlesize': 8.5, 'xtick.labelsize': 7.3,
                         'ytick.labelsize': 7.3, 'legend.fontsize': 7.3})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.55))
    handles = []
    for source in SOURCES:
        data = results[source.label]
        style = dict(color=source.color, linestyle=source.linestyle,
                     linewidth=source.linewidth)
        line, = axes[0].plot(grid, data['roc'], label=source.label, **style)
        axes[1].plot(grid, data['pr'], **style)
        handles.append(line)
    axes[0].plot([0, 1], [0, 1], ':', color='#A0A0A0', linewidth=.75)
    for ax, title, xlabel, ylabel in zip(
            axes[:2], ['(a) ROC', '(b) Precision–recall'],
            ['False positive rate', 'Recall'], ['True positive rate', 'Precision']):
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
        ax.set_title(title, loc='left', fontweight='semibold')
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
    ys = np.arange(len(SOURCES))
    bars = axes[2].barh(ys, [results[s.label]['fpr95'] for s in SOURCES],
                        color=[s.color for s in SOURCES], height=.65)
    axes[2].set_yticks(ys, [s.label for s in SOURCES], fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set(xlim=(0, 112), xlabel='FPR@95%TPR (%)')
    axes[2].set_xticks([0, 50, 100])
    axes[2].set_title('(c) False alarms', loc='left', fontweight='semibold')
    for bar in bars:
        axes[2].text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                     f'{bar.get_width():.1f}', va='center', fontsize=7)
    for ax in axes:
        ax.grid(color='#D9DEE3', linewidth=.5, axis='x' if ax is axes[2] else 'both')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.legend(handles, [s.label for s in SOURCES], loc='lower center',
               bbox_to_anchor=(.5, .005), ncol=6, columnspacing=.9, handlelength=2)
    fig.subplots_adjust(left=.075, right=.985, bottom=.27, top=.91, wspace=.55)
    output.mkdir(parents=True, exist_ok=True)
    stem = 'inhouse_metrics_preview' if args.preview else 'inhouse_metrics_formal'
    fig.savefig(output / (stem + '.png'), dpi=400)
    fig.savefig(output / (stem + '.pdf'))
    plt.close(fig)
    report = dict(status='replay_preview_not_for_main_table' if args.preview else 'formal',
                  aggregation='equal-weight per-cell curves; FPR95 is mean of per-cell first TPR>=0.95 operating points',
                  canvas_inches=[7.15, 2.55], results=results)
    (output / (stem + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(output / (stem + '.png'))
    print({name: {k: d[k] for k in ['auroc', 'average_precision', 'fpr95']} for name, d in results.items()})


if __name__ == '__main__':
    main()
