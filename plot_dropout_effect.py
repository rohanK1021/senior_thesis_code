"""
Generate Fig 6: Dropout eliminates GEE-Direct's context dependency.
Uses multi-seed data (5 seeds each) for error bars.

Run: python plot_dropout_effect.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [42, 123, 456, 789, 1024]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.dpi': 200,
})


def load_seeds(experiment):
    results = []
    for seed in SEEDS:
        path = os.path.join(BASE, 'results', 'multi_seed', experiment,
                            f'seed_{seed}', 'results.json')
        with open(path) as f:
            results.append(json.load(f))
    return results


def main():
    # Load no-dropout data
    no_drop = load_seeds('webb_no_dropout')
    no_drop_with = [r['normal']['overall'] for r in no_drop]
    no_drop_none = [r['none']['overall'] for r in no_drop]

    # Load with-dropout data (from webb experiment, GEE-Direct)
    with_drop = load_seeds('webb')
    wd_with = [r['GEE-Direct']['normal']['overall'] for r in with_drop]
    wd_none = [r['GEE-Direct']['none']['overall'] for r in with_drop]

    # Also load ESBN for reference
    esbn_with = [r['ESBN']['normal']['overall'] for r in with_drop]
    esbn_none = [r['ESBN']['none']['overall'] for r in with_drop]

    # And GEE-Separated
    sep_with = [r['GEE-Separated']['normal']['overall'] for r in with_drop]
    sep_none = [r['GEE-Separated']['none']['overall'] for r in with_drop]

    # ── Figure: 2-panel ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: With vs Without task_id (all models, dropout ON)
    ax = axes[0]
    models = ['ESBN', 'GEE-Direct\n(no dropout)', 'GEE-Direct\n(dropout p=0.5)', 'GEE-Separated']
    with_means = [np.mean(esbn_with), np.mean(no_drop_with), np.mean(wd_with), np.mean(sep_with)]
    with_stds = [np.std(esbn_with), np.std(no_drop_with), np.std(wd_with), np.std(sep_with)]
    none_means = [np.mean(esbn_none), np.mean(no_drop_none), np.mean(wd_none), np.mean(sep_none)]
    none_stds = [np.std(esbn_none), np.std(no_drop_none), np.std(wd_none), np.std(sep_none)]

    x = np.arange(len(models))
    w = 0.32

    bars1 = ax.bar(x - w/2, with_means, w, yerr=with_stds, capsize=4,
                   label='With task ID', color='#27AE60', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w/2, none_means, w, yerr=none_stds, capsize=4,
                   label='Without task ID', color='#F39C12', alpha=0.85, edgecolor='white')

    # Add value labels
    for bar, mean in zip(bars1, with_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.1%}', ha='center', fontsize=9, fontweight='bold', color='#27AE60')
    for bar, mean in zip(bars2, none_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.1%}', ha='center', fontsize=9, fontweight='bold', color='#E67E22')

    # Highlight the crash
    ax.annotate('', xy=(1 + w/2, np.mean(no_drop_none) + 0.01),
                xytext=(1 + w/2, np.mean(no_drop_with) - 0.01),
                arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=2))
    ax.text(1 + w/2 + 0.15, (np.mean(no_drop_with) + np.mean(no_drop_none)) / 2,
            f'-{np.mean(no_drop_with) - np.mean(no_drop_none):.1%}',
            fontsize=11, fontweight='bold', color='#E74C3C', va='center')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1.18)
    ax.set_title('A.  Task-ID Robustness (Webb Tasks)', fontweight='bold', loc='left')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, axis='y')

    # Panel B: Drop magnitude comparison
    ax = axes[1]
    drops_no = [w - n for w, n in zip(no_drop_with, no_drop_none)]
    drops_wd = [w - n for w, n in zip(wd_with, wd_none)]
    drops_esbn = [w - n for w, n in zip(esbn_with, esbn_none)]
    drops_sep = [w - n for w, n in zip(sep_with, sep_none)]

    labels = ['ESBN', 'GEE-Direct\n(no dropout)', 'GEE-Direct\n(dropout p=0.5)', 'GEE-Separated']
    means = [np.mean(drops_esbn)*100, np.mean(drops_no)*100,
             np.mean(drops_wd)*100, np.mean(drops_sep)*100]
    stds = [np.std(drops_esbn)*100, np.std(drops_no)*100,
            np.std(drops_wd)*100, np.std(drops_sep)*100]
    colors = ['#5B9BD5', '#E74C3C', '#9B59B6', '#2ECC71']

    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)
    for bar, m, s in zip(bars, means, stds):
        sign = '+' if m > 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.8,
                f'{sign}{m:.1f}%', ha='center', fontsize=10, fontweight='bold',
                color=bar.get_facecolor())

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Accuracy Drop Without Task ID (%)')
    ax.set_title('B.  Context Dependency (lower = more robust)', fontweight='bold', loc='left')
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, axis='y')

    plt.suptitle('Dropout Eliminates Context Dependency in GEE-Direct',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_ms_dropout_effect.png', bbox_inches='tight', facecolor='white')
    print("Saved: fig_ms_dropout_effect.png")
    plt.close()

    # Print summary
    print(f"\nGEE-Direct NO dropout:   {np.mean(no_drop_with):.1%} -> {np.mean(no_drop_none):.1%} "
          f"(drop: {np.mean(drops_no)*100:.1f}% +/- {np.std(drops_no)*100:.1f}%)")
    print(f"GEE-Direct WITH dropout: {np.mean(wd_with):.1%} -> {np.mean(wd_none):.1%} "
          f"(drop: {np.mean(drops_wd)*100:.1f}% +/- {np.std(drops_wd)*100:.1f}%)")
    print(f"ESBN:                    {np.mean(esbn_with):.1%} -> {np.mean(esbn_none):.1%} "
          f"(drop: {np.mean(drops_esbn)*100:.1f}% +/- {np.std(drops_esbn)*100:.1f}%)")
    print(f"GEE-Separated:           {np.mean(sep_with):.1%} -> {np.mean(sep_none):.1%} "
          f"(drop: {np.mean(drops_sep)*100:.1f}% +/- {np.std(drops_sep)*100:.1f}%)")


if __name__ == '__main__':
    main()
