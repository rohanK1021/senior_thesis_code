"""
plot_multiseed.py — Multi-seed thesis figures with error bars.

Reads results from results/multi_seed/{experiment}/seed_{seed}/results.json
for seeds [42, 123, 456, 789, 1024] and generates figures with mean ± std.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results', 'multi_seed')
SEEDS = [42, 123, 456, 789, 1024]
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 200,
})

MODEL_COLORS  = ['#5B9BD5', '#9B59B6', '#2ECC71']   # ESBN, GEE-Direct/GEE, GEE-Sep/GEE-h0
COND_COLORS   = ['#555555', '#9B59B6', '#E67E22']    # Global, h0, RCM
TASK_COLORS   = ['#5B9BD5', '#E74C3C', '#F39C12', '#2ECC71']


def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(experiment, seeds=SEEDS):
    """Return list of dicts (one per seed found). Missing seeds are skipped."""
    results = []
    for seed in seeds:
        path = os.path.join(BASE_DIR, experiment, f'seed_{seed}', 'results.json')
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return results


def collect_values(records, *keys):
    """
    Walk nested keys into each record and return a numpy array of values.
    E.g. collect_values(records, 'ESBN', 'normal', 'overall')
    """
    vals = []
    for r in records:
        node = r
        for k in keys:
            node = node[k]
        vals.append(float(node))
    return np.array(vals)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def bar_group(ax, group_labels, series_labels, means, stds, colors,
              width=0.22, ylabel='Accuracy', ylim=(0, 1.05),
              title='', legend_loc='upper right'):
    """
    Generic grouped-bar helper.
    means / stds : shape (n_series, n_groups)
    """
    n_groups  = len(group_labels)
    n_series  = len(series_labels)
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for i, (label, offset, color) in enumerate(zip(series_labels, offsets, colors)):
        ax.bar(x + offset, means[i], width,
               yerr=stds[i], capsize=4,
               color=color, alpha=0.85, label=label,
               error_kw={'elinewidth': 1.2, 'ecolor': 'black'})

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    if legend_loc == 'above':
        ax.legend(fontsize=9, framealpha=0.9, ncol=len(series_labels),
                  loc='lower center', bbox_to_anchor=(0.5, 1.0))
    else:
        ax.legend(loc=legend_loc, fontsize=9, framealpha=0.9)
    clean_axes(ax)


# ---------------------------------------------------------------------------
# Figure 1 — Unified Tasks
# ---------------------------------------------------------------------------

def fig_unified(records):
    """fig_ms_unified.png — 2-panel unified task results."""
    if not records:
        print('  [SKIP] unified: no data')
        return

    models       = ['ESBN', 'GEE', 'GEE-h0']
    model_labels = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    tasks      = ['same_different', 'rmts', 'identity_rules', 'dist3']
    task_short = ['S/D', 'RMTS', 'IdRules', 'Dist3']
    modes      = ['normal', 'none', 'wrong']
    mode_labels = ['With ID', 'No ID', 'Wrong ID']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Unified Tasks — Multi-Seed Results', fontsize=14, y=1.03)

    # --- Panel A: per-task accuracy WITH task_id ---
    # shape: (n_models, n_tasks)
    means_a = np.zeros((len(models), len(tasks)))
    stds_a  = np.zeros((len(models), len(tasks)))
    for mi, m in enumerate(models):
        for ti, t in enumerate(tasks):
            vals = collect_values(records, m, 'normal', 'per_task', t)
            means_a[mi, ti] = vals.mean()
            stds_a[mi, ti]  = vals.std()

    bar_group(ax1, task_short, model_labels, means_a, stds_a, MODEL_COLORS,
              width=0.22, ylabel='Accuracy',
              title='A  Per-Task Accuracy (with task_id)', ylim=(0, 1.08),
              legend_loc='lower right')

    # --- Panel B: overall accuracy by mode ---
    # shape: (n_models, n_modes)
    means_b = np.zeros((len(models), len(modes)))
    stds_b  = np.zeros((len(models), len(modes)))
    for mi, m in enumerate(models):
        for moi, mo in enumerate(modes):
            vals = collect_values(records, m, mo, 'overall')
            means_b[mi, moi] = vals.mean()
            stds_b[mi, moi]  = vals.std()

    bar_group(ax2, mode_labels, model_labels, means_b, stds_b, MODEL_COLORS,
              width=0.22, ylabel='Overall Accuracy',
              title='B  Overall Accuracy by Task-ID Condition', ylim=(0, 1.08),
              legend_loc='lower right')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_unified.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_unified.png  (n={len(records)} seeds)')
    _print_summary_unified(records, models, modes, tasks, task_short)


def _print_summary_unified(records, models, modes, tasks, task_short):
    print(f'\n  Unified summary (mean ± std across {len(records)} seeds):')
    header = f"  {'Model':<12}" + ''.join(f" {'Overall '+m:>16}" for m in modes)
    print(header)
    for m in models:
        row = f"  {m:<12}"
        for mo in modes:
            vals = collect_values(records, m, mo, 'overall')
            row += f"  {vals.mean():.3f} ± {vals.std():.3f}  "
        print(row)


# ---------------------------------------------------------------------------
# Figure 2 — Webb Three-Way
# ---------------------------------------------------------------------------

def fig_webb(records):
    """fig_ms_webb.png — Webb three-way comparison."""
    if not records:
        print('  [SKIP] webb: no data found')
        return
    if len(records) < 2:
        print(f'  [NOTE] webb: only {len(records)} seed(s) — error bars will be zero')

    models      = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    tasks       = ['same_different', 'webb_rmts', 'webb_idrules', 'webb_dist3']
    task_short  = ['S/D', 'RMTS', 'IdRules', 'Dist3']
    modes       = ['normal', 'none', 'wrong']
    mode_labels = ['With ID', 'No ID', 'Wrong ID']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('Webb Three-Way — Multi-Seed Results', fontsize=14, y=1.03)

    means_a = np.zeros((len(models), len(tasks)))
    stds_a  = np.zeros((len(models), len(tasks)))
    for mi, m in enumerate(models):
        for ti, t in enumerate(tasks):
            vals = collect_values(records, m, 'normal', 'per_task', t)
            means_a[mi, ti] = vals.mean()
            stds_a[mi, ti]  = vals.std()

    bar_group(ax1, task_short, models, means_a, stds_a, MODEL_COLORS,
              width=0.22, ylabel='Accuracy',
              title='A  Per-Task Accuracy (with task_id)', ylim=(0.85, 1.02),
              legend_loc='lower left')

    means_b = np.zeros((len(models), len(modes)))
    stds_b  = np.zeros((len(models), len(modes)))
    for mi, m in enumerate(models):
        for moi, mo in enumerate(modes):
            vals = collect_values(records, m, mo, 'overall')
            means_b[mi, moi] = vals.mean()
            stds_b[mi, moi]  = vals.std()

    bar_group(ax2, mode_labels, models, means_b, stds_b, MODEL_COLORS,
              width=0.22, ylabel='Overall Accuracy',
              title='B  Overall Accuracy by Task-ID Condition', ylim=(0.85, 1.02),
              legend_loc='lower left')

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_webb.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_webb.png  (n={len(records)} seeds)')


# ---------------------------------------------------------------------------
# Figure 3 — WCST
# ---------------------------------------------------------------------------

def fig_wcst(records):
    """fig_ms_wcst.png — WCST 3-panel overview."""
    if not records:
        print('  [SKIP] wcst: no data')
        return

    models       = ['ESBN', 'GEE-Direct', 'GEE-Separated']
    model_labels = ['ESBN', 'GEE-Direct', 'GEE-Separated']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('WCST — Multi-Seed Results', fontsize=14, y=1.01)

    def _collect_wcst(key):
        """Return (means, stds) arrays shape (n_models,) for the 'normal' condition."""
        means, stds = [], []
        for m in models:
            vals = collect_values(records, m, 'normal', key)
            means.append(vals.mean())
            stds.append(vals.std())
        return np.array(means), np.array(stds)

    x = np.arange(len(models))
    width = 0.5

    panels = [
        ('accuracy_mean',      'Accuracy',              'A  Accuracy',              (0, 1.1)),
        ('persev_errors_mean', 'Perseverative Errors',  'B  Perseverative Errors',  None),
        ('completions_mean',   'Rule Completions',      'C  Rule Completions',      (0, 7)),
    ]

    for ax, (key, ylabel, title, ylim) in zip(axes, panels):
        means, stds = _collect_wcst(key)
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=MODEL_COLORS, alpha=0.85,
                      error_kw={'elinewidth': 1.2, 'ecolor': 'black'})
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        clean_axes(ax)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_wcst.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_wcst.png  (n={len(records)} seeds)')
    _print_summary_wcst(records, models)


def _print_summary_wcst(records, models):
    print(f'\n  WCST summary (mean ± std across {len(records)} seeds, normal condition):')
    keys = ['accuracy_mean', 'persev_errors_mean', 'completions_mean']
    header = f"  {'Model':<16}" + ''.join(f" {k:>24}" for k in keys)
    print(header)
    for m in models:
        row = f"  {m:<16}"
        for k in keys:
            vals = collect_values(records, m, 'normal', k)
            row += f"  {vals.mean():.3f} ± {vals.std():.3f}          "
        print(row)


# ---------------------------------------------------------------------------
# Figure 4 — Noise on Static Tasks (accuracy curves)
# ---------------------------------------------------------------------------

def fig_noise_static(records):
    """fig_ms_noise_static.png — Noise accuracy curves with std shading."""
    if not records:
        print('  [SKIP] noise: no data')
        return

    sigmas     = [0.0, 0.5, 1.0, 2.0]
    conditions = ['global', 'h0_only', 'rcm_only']
    cond_labels = ['Global', 'h\u2080 only', 'RCM only']

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        means, stds = [], []
        for s in sigmas:
            vals = collect_values(records, cond, str(s), 'overall')
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(sigmas, means, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax.fill_between(sigmas, means - stds, means + stds, alpha=0.2, color=color)

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance')
    ax.set_xlabel('Noise \u03c3')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Noise Injection — Static Tasks (mean \u00b1 1 std)')
    ax.set_xticks(sigmas)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    clean_axes(ax)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_noise_static.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_noise_static.png  (n={len(records)} seeds)')


# ---------------------------------------------------------------------------
# Figure 5 — Noise on WCST
# ---------------------------------------------------------------------------

def fig_noise_wcst(records):
    """fig_ms_noise_wcst.png — Noise on WCST, 2 panels."""
    if not records:
        print('  [SKIP] wcst_noise: no data')
        return

    sigmas      = [0.0, 0.5, 1.0, 2.0]
    conditions  = ['global', 'h0_only', 'rcm_only']
    cond_labels = ['Global', 'h\u2080 only', 'RCM only']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Noise Injection — WCST (GEE-Combined)', fontsize=14, y=1.01)

    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        acc_m, acc_s, pe_m, pe_s = [], [], [], []
        for s in sigmas:
            acc_vals = collect_values(records, cond, str(s), 'accuracy_mean')
            pe_vals  = collect_values(records, cond, str(s), 'persev_errors_mean')
            acc_m.append(acc_vals.mean());  acc_s.append(acc_vals.std())
            pe_m.append(pe_vals.mean());    pe_s.append(pe_vals.std())

        acc_m = np.array(acc_m); acc_s = np.array(acc_s)
        pe_m  = np.array(pe_m);  pe_s  = np.array(pe_s)

        ax1.plot(sigmas, acc_m, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax1.fill_between(sigmas, acc_m - acc_s, acc_m + acc_s, alpha=0.2, color=color)

        ax2.plot(sigmas, pe_m, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax2.fill_between(sigmas, pe_m - pe_s, pe_m + pe_s, alpha=0.2, color=color)

    ax1.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance')
    ax1.set_xlabel('Noise \u03c3');   ax1.set_ylabel('Accuracy')
    ax1.set_title('A  Accuracy vs \u03c3');  ax1.set_xticks(sigmas)
    ax1.set_ylim(0.3, 1.05);   ax1.legend(fontsize=9, loc='lower left', framealpha=0.9)
    clean_axes(ax1)

    ax2.set_xlabel('Noise \u03c3');   ax2.set_ylabel('Perseverative Errors')
    ax2.set_title('B  Perseverative Errors vs \u03c3');  ax2.set_xticks(sigmas)
    ax2.legend(fontsize=9, loc='lower right', framealpha=0.9)
    clean_axes(ax2)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_noise_wcst.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_noise_wcst.png  (n={len(records)} seeds)')


# ---------------------------------------------------------------------------
# Figure 11 — Combined Noise (Static + WCST), uniform panels
# ---------------------------------------------------------------------------

def fig_figure11_noise(noise_records, wcst_noise_records):
    """fig_ms_figure11_noise.png — 1x3 row, uniform panel sizes.

    A: Static accuracy vs sigma  (data from `noise` records)
    B: WCST accuracy vs sigma    (data from `wcst_noise` records)
    C: WCST perseverative errors vs sigma  (data from `wcst_noise` records)

    All three panels share the GEE-Combined model (use_h0=True, use_rcm=True,
    no key modulation). Used as Figure 11 in the thesis.
    """
    if not noise_records:
        print('  [SKIP] figure11: no noise data')
        return
    if not wcst_noise_records:
        print('  [SKIP] figure11: no wcst_noise data')
        return

    sigmas      = [0.0, 0.5, 1.0, 2.0]
    conditions  = ['global', 'h0_only', 'rcm_only']
    cond_labels = ['Global', 'h\u2080 only', 'RCM only']

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Pathway-Specific Noise — GEE-Combined '
                 '(mean \u00b1 1 std, n=5 seeds)', fontsize=13, y=1.02)

    # --- Panel A: Static accuracy ---
    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        means, stds = [], []
        for s in sigmas:
            vals = collect_values(noise_records, cond, str(s), 'overall')
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means); stds = np.array(stds)
        axA.plot(sigmas, means, 'o-', color=color, label=label,
                 linewidth=2, markersize=5)
        axA.fill_between(sigmas, means - stds, means + stds,
                         alpha=0.2, color=color)
    axA.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance')
    axA.set_xlabel('Noise \u03c3'); axA.set_ylabel('Overall Accuracy')
    axA.set_title('A  Static Tasks — Accuracy')
    axA.set_xticks(sigmas); axA.set_ylim(0.3, 1.05)
    axA.legend(fontsize=9, loc='lower left', framealpha=0.9)
    clean_axes(axA)

    # --- Panel B: WCST accuracy ---
    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        means, stds = [], []
        for s in sigmas:
            vals = collect_values(wcst_noise_records, cond, str(s),
                                  'accuracy_mean')
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means); stds = np.array(stds)
        axB.plot(sigmas, means, 'o-', color=color, label=label,
                 linewidth=2, markersize=5)
        axB.fill_between(sigmas, means - stds, means + stds,
                         alpha=0.2, color=color)
    axB.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Chance')
    axB.set_xlabel('Noise \u03c3'); axB.set_ylabel('Accuracy')
    axB.set_title('B  WCST — Accuracy')
    axB.set_xticks(sigmas); axB.set_ylim(0.3, 1.05)
    axB.legend(fontsize=9, loc='lower left', framealpha=0.9)
    clean_axes(axB)

    # --- Panel C: WCST perseverative errors ---
    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        means, stds = [], []
        for s in sigmas:
            vals = collect_values(wcst_noise_records, cond, str(s),
                                  'persev_errors_mean')
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means); stds = np.array(stds)
        axC.plot(sigmas, means, 'o-', color=color, label=label,
                 linewidth=2, markersize=5)
        axC.fill_between(sigmas, means - stds, means + stds,
                         alpha=0.2, color=color)
    axC.set_xlabel('Noise \u03c3'); axC.set_ylabel('Perseverative Errors')
    axC.set_title('C  WCST — Perseverative Errors')
    axC.set_xticks(sigmas)
    axC.legend(fontsize=9, loc='upper left', framealpha=0.9)
    clean_axes(axC)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_figure11_noise.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_figure11_noise.png  '
          f'(noise n={len(noise_records)}, wcst_noise n={len(wcst_noise_records)} seeds)')


# ---------------------------------------------------------------------------
# Figure 6 — Double Dissociation
# ---------------------------------------------------------------------------

def fig_double_dissociation(noise_records, wcst_noise_records):
    """fig_ms_double_dissociation.png — THE thesis figure with error bars."""
    if not noise_records:
        print('  [SKIP] double_dissociation: no noise data')
        return
    if not wcst_noise_records:
        print('  [SKIP] double_dissociation: no wcst_noise data')
        return

    conditions  = ['global', 'h0_only', 'rcm_only']
    cond_labels = ['Global', 'h\u2080 only', 'RCM only']
    pathway_labels = ['h\u2080', 'RCM']  # for interaction plot lines

    # Compute per-seed accuracy drop (sigma=0 → sigma=2) for each condition
    def compute_drops(records, acc_key):
        """Returns dict: condition -> np.array of drops across seeds."""
        drops = {}
        for cond in conditions:
            d = []
            for r in records:
                acc0 = float(r[cond]['0.0'][acc_key])
                acc2 = float(r[cond]['2.0'][acc_key])
                d.append(acc0 - acc2)   # positive = degradation
            drops[cond] = np.array(d)
        return drops

    static_drops = compute_drops(noise_records,     'overall')
    wcst_drops   = compute_drops(wcst_noise_records, 'accuracy_mean')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Double Dissociation: Noise Pathway \u00d7 Task Structure',
                 fontsize=14, y=1.03)

    # --- Panel A: Interaction plot ---
    # x-axis: [Static, Dynamic (WCST)]
    # Lines: h0_only vs rcm_only
    x_pos = [0, 1]
    x_labels = ['Static Tasks', 'Dynamic (WCST)']

    for cond, label, color in zip(['h0_only', 'rcm_only'],
                                   pathway_labels, COND_COLORS[1:]):
        static_mean = static_drops[cond].mean()
        static_std  = static_drops[cond].std()
        wcst_mean   = wcst_drops[cond].mean()
        wcst_std    = wcst_drops[cond].std()

        means = [static_mean, wcst_mean]
        stds  = [static_std,  wcst_std]

        ax1.plot(x_pos, means, 'o-', color=color, label=label, linewidth=2.5,
                 markersize=8)
        ax1.errorbar(x_pos, means, yerr=stds, fmt='none',
                     ecolor=color, capsize=5, elinewidth=1.5)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel('Accuracy Drop (\u03c3=0 \u2192 \u03c3=2)')
    ax1.set_title('A  Pathway \u00d7 Task Interaction')
    ax1.legend(fontsize=10, framealpha=0.9, loc='center right')
    ax1.set_ylim(bottom=0, top=0.5)
    clean_axes(ax1)

    # --- Panel B: Grouped bars — Static vs WCST, by condition ---
    conditions_plot  = ['global', 'h0_only', 'rcm_only']
    cond_plot_labels = ['Global', 'h\u2080 only', 'RCM only']
    x = np.arange(len(conditions_plot))
    width = 0.3

    static_means = np.array([static_drops[c].mean() for c in conditions_plot])
    static_stds  = np.array([static_drops[c].std()  for c in conditions_plot])
    wcst_means   = np.array([wcst_drops[c].mean()   for c in conditions_plot])
    wcst_stds    = np.array([wcst_drops[c].std()    for c in conditions_plot])

    ax2.bar(x - width/2, static_means, width, yerr=static_stds, capsize=4,
            color='#5B9BD5', alpha=0.85, label='Static Tasks',
            error_kw={'elinewidth': 1.2, 'ecolor': 'black'})
    ax2.bar(x + width/2, wcst_means, width, yerr=wcst_stds, capsize=4,
            color='#E74C3C', alpha=0.85, label='WCST',
            error_kw={'elinewidth': 1.2, 'ecolor': 'black'})

    ax2.set_xticks(x)
    ax2.set_xticklabels(cond_plot_labels)
    ax2.set_ylabel('Accuracy Drop')
    ax2.set_title('B  Accuracy Drop by Pathway and Task Type')
    ax2.legend(fontsize=10, framealpha=0.9, loc='upper center')
    ax2.set_ylim(bottom=0, top=0.55)
    clean_axes(ax2)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_double_dissociation.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_double_dissociation.png  '
          f'(noise n={len(noise_records)}, wcst_noise n={len(wcst_noise_records)} seeds)')

    # Print dissociation summary
    print('\n  Double dissociation summary (drop = acc@σ=0 − acc@σ=2):')
    print(f"  {'Condition':<12}  {'Static (mean±std)':>20}  {'WCST (mean±std)':>20}")
    for cond, label in zip(conditions_plot, cond_plot_labels):
        sm, ss = static_drops[cond].mean(), static_drops[cond].std()
        wm, ws = wcst_drops[cond].mean(),   wcst_drops[cond].std()
        print(f'  {label:<12}  {sm:>8.3f} ± {ss:<6.3f}        {wm:>8.3f} ± {ws:<6.3f}')


# ---------------------------------------------------------------------------
# Figure 7 — Consistency
# ---------------------------------------------------------------------------

def fig_noise_consistency(records):
    """fig_ms_noise_consistency.png — Consistency vs sigma with std shading."""
    if not records:
        print('  [SKIP] noise_consistency: no noise data')
        return

    sigmas      = [0.0, 0.5, 1.0, 2.0]
    conditions  = ['global', 'h0_only', 'rcm_only']
    cond_labels = ['Global', 'h\u2080 only', 'RCM only']

    fig, ax = plt.subplots(figsize=(10, 5))

    for cond, label, color in zip(conditions, cond_labels, COND_COLORS):
        means, stds = [], []
        for s in sigmas:
            vals = collect_values(records, cond, str(s), 'consistency')
            means.append(vals.mean())
            stds.append(vals.std())
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(sigmas, means, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax.fill_between(sigmas, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel('Noise \u03c3')
    ax.set_ylabel('Consistency')
    ax.set_title('Response Consistency vs Noise \u03c3 (mean \u00b1 1 std)')
    ax.set_xticks(sigmas)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    clean_axes(ax)

    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'fig_ms_noise_consistency.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK]   fig_ms_noise_consistency.png  (n={len(records)} seeds)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('plot_multiseed.py — Generating multi-seed thesis figures')
    print('=' * 60)
    print(f'Seeds: {SEEDS}')
    print(f'Base dir: {BASE_DIR}\n')

    # Load all experiments
    unified_records   = load_experiment('unified')
    webb_records      = load_experiment('webb')
    wcst_records      = load_experiment('wcst')
    noise_records     = load_experiment('noise')
    wcst_noise_records = load_experiment('wcst_noise')

    print(f'Records found:')
    print(f'  unified   : {len(unified_records)}/{len(SEEDS)} seeds')
    print(f'  webb      : {len(webb_records)}/{len(SEEDS)} seeds')
    print(f'  wcst      : {len(wcst_records)}/{len(SEEDS)} seeds')
    print(f'  noise     : {len(noise_records)}/{len(SEEDS)} seeds')
    print(f'  wcst_noise: {len(wcst_noise_records)}/{len(SEEDS)} seeds')
    print()

    generated = []
    skipped   = []

    print('Generating figures...')

    # 1. Unified
    if unified_records:
        fig_unified(unified_records)
        generated.append('fig_ms_unified.png')
    else:
        print('  [SKIP] fig_ms_unified.png — no unified data')
        skipped.append('fig_ms_unified.png')

    print()

    # 2. Webb
    if webb_records:
        fig_webb(webb_records)
        generated.append('fig_ms_webb.png')
    else:
        print('  [SKIP] fig_ms_webb.png — no webb data (experiment not complete)')
        skipped.append('fig_ms_webb.png')

    print()

    # 3. WCST
    if wcst_records:
        fig_wcst(wcst_records)
        generated.append('fig_ms_wcst.png')
    else:
        print('  [SKIP] fig_ms_wcst.png — no wcst data')
        skipped.append('fig_ms_wcst.png')

    print()

    # 4. Noise static
    if noise_records:
        fig_noise_static(noise_records)
        generated.append('fig_ms_noise_static.png')
    else:
        print('  [SKIP] fig_ms_noise_static.png — no noise data')
        skipped.append('fig_ms_noise_static.png')

    print()

    # 5. Noise WCST
    if wcst_noise_records:
        fig_noise_wcst(wcst_noise_records)
        generated.append('fig_ms_noise_wcst.png')
    else:
        print('  [SKIP] fig_ms_noise_wcst.png — no wcst_noise data')
        skipped.append('fig_ms_noise_wcst.png')

    print()

    # 6. Double dissociation
    if noise_records and wcst_noise_records:
        fig_double_dissociation(noise_records, wcst_noise_records)
        generated.append('fig_ms_double_dissociation.png')
    else:
        print('  [SKIP] fig_ms_double_dissociation.png — need both noise and wcst_noise')
        skipped.append('fig_ms_double_dissociation.png')

    print()

    # 6b. Combined Figure 11 (uniform 1x3 noise panels)
    if noise_records and wcst_noise_records:
        fig_figure11_noise(noise_records, wcst_noise_records)
        generated.append('fig_ms_figure11_noise.png')
    else:
        print('  [SKIP] fig_ms_figure11_noise.png — need both noise and wcst_noise')
        skipped.append('fig_ms_figure11_noise.png')

    print()

    # 7. Consistency
    if noise_records:
        fig_noise_consistency(noise_records)
        generated.append('fig_ms_noise_consistency.png')
    else:
        print('  [SKIP] fig_ms_noise_consistency.png — no noise data')
        skipped.append('fig_ms_noise_consistency.png')

    print()
    print('=' * 60)
    print(f'Generated ({len(generated)}): ' + ', '.join(generated))
    if skipped:
        print(f'Skipped   ({len(skipped)}): ' + ', '.join(skipped))
    print('=' * 60)


if __name__ == '__main__':
    main()
