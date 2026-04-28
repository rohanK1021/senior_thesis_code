"""
nonlinearity_analysis.py
========================
Tests whether disrupting BOTH context pathways simultaneously is
worse/better than the sum of individual disruptions (superadditivity /
subadditivity).  Uses EXISTING results — no training needed.

Data sources:
  results/results_noise_experiment.json — unified task noise (sweep.{cond}.{sigma}.overall)
  results_wcst_noise_raw.pkl            — WCST noise (summary.sweep.{cond}.{sigma}.accuracy_mean)
    * sigma keys are floats in the pickle

Output:
  fig_nonlinearity.png  — 2-panel grouped bar plot with interaction annotations
  Printed table to stdout

Run: python -u nonlinearity_analysis.py
"""

import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UNIFIED_JSON = Path('results/results_noise_experiment.json')
WCST_PKL     = Path('results_wcst_noise_raw.pkl')

SIGMAS = [0.5, 1.0, 2.0]   # analysis sigmas
SIGMA_STRS = ['0.5', '1.0', '2.0']

COND_GLOBAL = 'global'
COND_H0     = 'h0_only'
COND_RCM    = 'rcm_only'


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_unified():
    with open(UNIFIED_JSON) as f:
        return json.load(f)


def load_wcst():
    with open(WCST_PKL, 'rb') as f:
        return pickle.load(f)


def get_unified_acc(data, cond, sigma_str):
    """Accuracy for unified tasks at given condition and sigma (string key)."""
    return data['sweep'][cond][sigma_str]['overall']


def get_wcst_acc(data, cond, sigma_f):
    """Accuracy for WCST at given condition and sigma (float key)."""
    return data['summary']['sweep'][cond][sigma_f]['accuracy_mean']


# ---------------------------------------------------------------------------
# Interaction analysis
# ---------------------------------------------------------------------------

def compute_interactions(acc_fn, sigmas):
    """
    For each sigma compute:
      h0_drop        = acc@0 - acc_h0@sigma
      rcm_drop       = acc@0 - acc_rcm@sigma
      predicted_drop = h0_drop + rcm_drop   (additive null)
      actual_drop    = acc@0 - acc_global@sigma
      interaction    = actual_drop - predicted_drop
        > 0  superadditive  (pathways compensate each other)
        < 0  subadditive    (diminishing returns)
        ≈ 0  independent

    acc_fn(cond, sigma) must return float accuracy.
    """
    # Baseline (sigma=0)
    # We use each condition's sigma=0 value and average for a stable baseline
    # because each condition should have nearly identical sigma=0 performance.
    base_global = acc_fn(COND_GLOBAL, 0)
    base_h0     = acc_fn(COND_H0,     0)
    base_rcm    = acc_fn(COND_RCM,    0)
    base = np.mean([base_global, base_h0, base_rcm])

    rows = []
    for sigma in sigmas:
        acc_global = acc_fn(COND_GLOBAL, sigma)
        acc_h0     = acc_fn(COND_H0,     sigma)
        acc_rcm    = acc_fn(COND_RCM,    sigma)

        h0_drop   = base - acc_h0
        rcm_drop  = base - acc_rcm
        pred_drop = h0_drop + rcm_drop
        act_drop  = base - acc_global
        interaction = act_drop - pred_drop

        rows.append({
            'sigma':       sigma,
            'base':        base,
            'acc_global':  acc_global,
            'acc_h0':      acc_h0,
            'acc_rcm':     acc_rcm,
            'h0_drop':     h0_drop,
            'rcm_drop':    rcm_drop,
            'pred_drop':   pred_drop,
            'act_drop':    act_drop,
            'interaction': interaction,
            'type': ('superadditive' if interaction > 0.01
                     else 'subadditive' if interaction < -0.01
                     else 'independent'),
        })
    return rows


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(rows, label):
    print(f'\n{"="*72}')
    print(f'Nonlinearity Analysis — {label}')
    print(f'{"="*72}')
    header = (f"{'σ':>6}  {'Base':>7}  {'h0_drop':>9}  {'rcm_drop':>9}  "
              f"{'pred_drop':>10}  {'act_drop':>9}  {'interact':>9}  {'type'}")
    print(header)
    print('-' * 72)
    for r in rows:
        print(
            f"{r['sigma']:>6.1f}  "
            f"{r['base']*100:>6.1f}%  "
            f"{r['h0_drop']*100:>8.1f}%  "
            f"{r['rcm_drop']*100:>8.1f}%  "
            f"{r['pred_drop']*100:>9.1f}%  "
            f"{r['act_drop']*100:>8.1f}%  "
            f"{r['interaction']*100:>+8.1f}%  "
            f"{r['type']}"
        )

    print()
    print('Interpretation:')
    print('  interaction > 0  → superadditive (actual drop > sum of parts → pathways compensate)')
    print('  interaction < 0  → subadditive   (actual drop < sum of parts → diminishing returns)')
    print('  interaction ≈ 0  → independent   (pathways act independently)')


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(rows_unified, rows_wcst, out_path='fig_nonlinearity.png'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Nonlinear Interaction Between Context Pathways', fontsize=14, fontweight='bold')

    datasets = [
        (axes[0], rows_unified, 'Unified Tasks'),
        (axes[1], rows_wcst,    'WCST'),
    ]

    x = np.arange(len(SIGMAS))
    bar_w = 0.35

    for ax, rows, title in datasets:
        pred_drops = [r['pred_drop'] * 100 for r in rows]
        act_drops  = [r['act_drop']  * 100 for r in rows]
        interactions = [r['interaction'] * 100 for r in rows]

        bars_pred = ax.bar(x - bar_w/2, pred_drops, bar_w,
                           color='#888888', label='Predicted drop\n(additive sum)',
                           edgecolor='black', linewidth=0.8, alpha=0.85)
        bars_act  = ax.bar(x + bar_w/2, act_drops,  bar_w,
                           color='#2E86C1',  label='Actual drop\n(global noise)',
                           edgecolor='black', linewidth=0.8, alpha=0.85)

        # Annotate interaction term above each pair
        for i, (pd, ad, ia) in enumerate(zip(pred_drops, act_drops, interactions)):
            top = max(pd, ad)
            sign = '+' if ia >= 0 else ''
            color = '#C0392B' if ia > 1 else '#1E8449' if ia < -1 else '#555555'
            ax.annotate(
                f'{sign}{ia:.1f}%',
                xy=(x[i], top + 0.5),
                ha='center', va='bottom',
                fontsize=9, color=color, fontweight='bold'
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f'σ={s}' for s in SIGMAS])
        ax.set_xlabel('Noise level (σ)', fontsize=11)
        ax.set_ylabel('Accuracy drop (%)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_ylim(0, max(max(pred_drops), max(act_drops)) * 1.35 + 5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add a note about interaction direction in bottom right
        ax.text(0.98, 0.02,
                'Red annotation = superadditive\nGreen = subadditive',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=7.5, color='#555555',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'\nSaved figure: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')

    # --- Unified tasks ---
    if not UNIFIED_JSON.exists():
        print(f'ERROR: {UNIFIED_JSON} not found.')
        unified_ok = False
    else:
        unified_data = load_unified()
        unified_ok = True
        print(f'  Loaded {UNIFIED_JSON}')

    # --- WCST ---
    if not WCST_PKL.exists():
        print(f'ERROR: {WCST_PKL} not found.')
        wcst_ok = False
    else:
        wcst_data = load_wcst()
        wcst_ok = True
        print(f'  Loaded {WCST_PKL}')

    # Compute interactions
    if unified_ok:
        def unified_acc(cond, sigma):
            return get_unified_acc(unified_data, cond, str(float(sigma)))
        rows_unified = compute_interactions(unified_acc, SIGMAS)
        print_table(rows_unified, 'Unified Tasks')
    else:
        rows_unified = None

    if wcst_ok:
        def wcst_acc(cond, sigma):
            return get_wcst_acc(wcst_data, cond, float(sigma))
        rows_wcst = compute_interactions(wcst_acc, SIGMAS)
        print_table(rows_wcst, 'WCST')
    else:
        rows_wcst = None

    # Figure requires both datasets; fall back gracefully
    if rows_unified is not None and rows_wcst is not None:
        make_figure(rows_unified, rows_wcst)
    elif rows_unified is not None:
        print('\nWCST data unavailable — generating single-panel figure (unified only).')
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle('Nonlinear Interaction Between Context Pathways', fontsize=13)
        _plot_panel(ax, rows_unified, 'Unified Tasks')
        plt.tight_layout()
        plt.savefig('fig_nonlinearity.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Saved figure: fig_nonlinearity.png')
    elif rows_wcst is not None:
        print('\nUnified data unavailable — generating single-panel figure (WCST only).')
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle('Nonlinear Interaction Between Context Pathways', fontsize=13)
        _plot_panel(ax, rows_wcst, 'WCST')
        plt.tight_layout()
        plt.savefig('fig_nonlinearity.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Saved figure: fig_nonlinearity.png')
    else:
        print('\nNo data available — cannot generate figure.')

    print('\nDone.')


def _plot_panel(ax, rows, title):
    """Helper for single-panel fallback."""
    x = np.arange(len(SIGMAS))
    bar_w = 0.35
    pred_drops = [r['pred_drop'] * 100 for r in rows]
    act_drops  = [r['act_drop']  * 100 for r in rows]
    interactions = [r['interaction'] * 100 for r in rows]

    ax.bar(x - bar_w/2, pred_drops, bar_w, color='#888888',
           label='Predicted (additive)', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + bar_w/2, act_drops,  bar_w, color='#2E86C1',
           label='Actual (global)',     edgecolor='black', linewidth=0.8, alpha=0.85)

    for i, (pd, ad, ia) in enumerate(zip(pred_drops, act_drops, interactions)):
        top = max(pd, ad)
        sign = '+' if ia >= 0 else ''
        color = '#C0392B' if ia > 1 else '#1E8449' if ia < -1 else '#555555'
        ax.annotate(f'{sign}{ia:.1f}%', xy=(x[i], top + 0.5),
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={s}' for s in SIGMAS])
    ax.set_xlabel('Noise level (σ)')
    ax.set_ylabel('Accuracy drop (%)')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


if __name__ == '__main__':
    main()
