"""
Experiment 3: Dropout Rate Ablation on WCST and Webb Tasks
==========================================================
Train GEE-Separated at multiple task_id dropout rates. Evaluate with/without/wrong
task_id on both WCST and Webb tasks. Find optimal dropout for each task type.

Hypothesis: WCST (all trials structurally identical) needs low dropout to preserve
context; Webb (structurally distinct tasks) benefits from high dropout for robustness.

Run: python -u gee_wcst_dropout_ablation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import DatasetFactory, EntityPool
from gee_model import GEEConfig

# WCST imports
from gee_wcst import (WCST_GEESeparated, train_wcst, run_evaluation,
                       generate_wcst_data, N_RULES)

# Webb imports
from gee_webb_comparison import (WebbGEESeparated, prepare_data, train_model,
                                  evaluate_model, TASK_CONFIGS, MTDataset,
                                  mt_forward_and_loss)


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Configuration
# =============================================================================

DROPOUT_RATES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
SEED = 42
RESULTS_DIR = 'results/dropout_ablation'
LOG_PATH = '/private/tmp/dropout_ablation.txt'


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def save_result(filename, data):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    log(f"    Saved: {path}")


def load_result(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None


# =============================================================================
# Part A: WCST Dropout Ablation
# =============================================================================

def run_wcst_ablation(device, cfg):
    log("=" * 70)
    log("  Part A: WCST Dropout Ablation")
    log("=" * 70)

    pool = EntityPool(n_train=40, n_test=16, seed=SEED)
    train_entities = pool.get('train')
    test_entities = pool.get('test')
    log(f"  Train entities: {len(train_entities)}  Test: {len(test_entities)}")
    log()

    wcst_results = {}

    for p in DROPOUT_RATES:
        result_file = f'wcst_p_{p:.2f}.json'
        existing = load_result(result_file)
        if existing is not None:
            log(f"  [SKIP] p={p:.2f} (results exist)")
            wcst_results[p] = existing
            continue

        log(f"  {'─' * 60}")
        log(f"  Dropout p={p:.2f}")
        log(f"  {'─' * 60}")

        set_seeds(SEED)
        model = WCST_GEESeparated(cfg).to(device)
        model.task_id_dropout = p
        n_params = sum(p_t.numel() for p_t in model.parameters() if p_t.requires_grad)
        log(f"    Params: {n_params:,}  task_id_dropout={model.task_id_dropout}")

        t0 = time.time()
        train_wcst(model, train_entities, device, lr=5e-4, n_epochs=50,
                   n_per_rule=1000, seed=SEED)
        train_time = time.time() - t0
        log(f"    Training: {train_time:.0f}s")

        result_data = {'dropout_rate': p, 'train_time_s': train_time}

        for mode in ['normal', 'none', 'wrong']:
            r = run_evaluation(model, test_entities, device, n_sessions=50,
                               task_id_mode=mode)
            result_data[mode] = {
                'accuracy_mean': r['accuracy_mean'],
                'accuracy_std': r['accuracy_std'],
                'persev_errors_mean': r['persev_errors_mean'],
                'persev_errors_std': r['persev_errors_std'],
                'completions_mean': r['completions_mean'],
                'completions_std': r['completions_std'],
            }
            log(f"    {mode:<8} acc={r['accuracy_mean']:.1%}  "
                f"persev={r['persev_errors_mean']:.1f}  "
                f"completions={r['completions_mean']:.1f}")

        save_result(result_file, result_data)
        wcst_results[p] = result_data
        log()

    return wcst_results


# =============================================================================
# Part B: Webb Tasks Dropout Ablation
# =============================================================================

def run_webb_ablation(device, cfg):
    log("=" * 70)
    log("  Part B: Webb Tasks Dropout Ablation")
    log("=" * 70)

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=SEED)
    log(f"  {factory.pool.summary()}")

    set_seeds(SEED)
    train_data = prepare_data(factory, 1500, 'train')
    test_data = prepare_data(factory, 500, 'test')
    log(f"  Train: {len(train_data[0])}  Test: {len(test_data[0])}")
    log()

    webb_results = {}

    for p in DROPOUT_RATES:
        result_file = f'webb_p_{p:.2f}.json'
        existing = load_result(result_file)
        if existing is not None:
            log(f"  [SKIP] p={p:.2f} (results exist)")
            webb_results[p] = existing
            continue

        log(f"  {'─' * 60}")
        log(f"  Dropout p={p:.2f}")
        log(f"  {'─' * 60}")

        set_seeds(SEED)
        model = WebbGEESeparated(cfg).to(device)
        model.task_id_dropout = p
        n_params = sum(p_t.numel() for p_t in model.parameters() if p_t.requires_grad)
        log(f"    Params: {n_params:,}  task_id_dropout={model.task_id_dropout}")

        t0 = time.time()
        train_model(model, train_data, device, lr=5e-4, n_epochs=100)
        train_time = time.time() - t0
        log(f"    Training: {train_time:.0f}s")

        result_data = {'dropout_rate': p, 'train_time_s': train_time}

        for mode in ['normal', 'none', 'wrong']:
            overall, per_task = evaluate_model(model, test_data, device,
                                               task_id_mode=mode)
            result_data[mode] = {
                'overall': overall,
                'per_task': per_task,
            }
            task_str = "  ".join(f"{tc['short']}={per_task[tc['name']]:.0%}"
                                 for tc in TASK_CONFIGS)
            log(f"    {mode:<8} overall={overall:.1%}  {task_str}")

        save_result(result_file, result_data)
        webb_results[p] = result_data
        log()

    return webb_results


# =============================================================================
# Figures
# =============================================================================

def setup_plot_style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 200,
    })


def plot_wcst_ablation(wcst_results):
    """Fig 1: WCST accuracy, persev errors, completions vs dropout rate."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    rates = sorted(wcst_results.keys())
    acc_normal = [wcst_results[p]['normal']['accuracy_mean'] for p in rates]
    acc_none = [wcst_results[p]['none']['accuracy_mean'] for p in rates]
    acc_wrong = [wcst_results[p]['wrong']['accuracy_mean'] for p in rates]
    persev_normal = [wcst_results[p]['normal']['persev_errors_mean'] for p in rates]
    comp_normal = [wcst_results[p]['normal']['completions_mean'] for p in rates]

    # Panel A: Accuracy vs dropout
    ax = axes[0]
    ax.plot(rates, acc_normal, '-o', color='#27AE60', label='With task ID',
            linewidth=2, markersize=6)
    ax.plot(rates, acc_none, '-s', color='#F39C12', label='No task ID',
            linewidth=2, markersize=6)
    ax.plot(rates, acc_wrong, '-^', color='#E74C3C', label='Wrong task ID',
            linewidth=2, markersize=6)
    # Mark existing p=0.5 result
    idx_05 = rates.index(0.5)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.annotate('default', xy=(0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.35),
                xytext=(0.5, 0.35), fontsize=8, color='gray', ha='center',
                fontstyle='italic')
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Accuracy')
    ax.set_title('A.  WCST accuracy vs dropout', fontweight='bold', loc='left')
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Perseveration errors
    ax = axes[1]
    ax.plot(rates, persev_normal, '-o', color='#E74C3C', linewidth=2, markersize=6)
    for i, (r, v) in enumerate(zip(rates, persev_normal)):
        ax.annotate(f'{v:.1f}', (r, v), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Perseverative errors')
    ax.set_title('B.  Perseveration (with task ID)', fontweight='bold', loc='left')
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Rule completions
    ax = axes[2]
    ax.plot(rates, comp_normal, '-o', color='#2ECC71', linewidth=2, markersize=6)
    for i, (r, v) in enumerate(zip(rates, comp_normal)):
        ax.annotate(f'{v:.1f}', (r, v), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8)
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Rules completed')
    ax.set_title('C.  Rule completions (with task ID)', fontweight='bold', loc='left')
    ax.set_ylim(0, 7)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('WCST: Effect of Task ID Dropout Rate (GEE-Separated)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_dropout_ablation_wcst.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    log("  Saved: fig_dropout_ablation_wcst.png")
    plt.close()


def plot_webb_ablation(webb_results):
    """Fig 2: Webb accuracy (3 modes) + per-task breakdown vs dropout rate."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    rates = sorted(webb_results.keys())
    acc_normal = [webb_results[p]['normal']['overall'] for p in rates]
    acc_none = [webb_results[p]['none']['overall'] for p in rates]
    acc_wrong = [webb_results[p]['wrong']['overall'] for p in rates]

    # Panel A: Overall accuracy vs dropout
    ax = axes[0]
    ax.plot(rates, acc_normal, '-o', color='#27AE60', label='With task ID',
            linewidth=2, markersize=6)
    ax.plot(rates, acc_none, '-s', color='#F39C12', label='No task ID',
            linewidth=2, markersize=6)
    ax.plot(rates, acc_wrong, '-^', color='#E74C3C', label='Wrong task ID',
            linewidth=2, markersize=6)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.annotate('default', xy=(0.5, 0.35), fontsize=8, color='gray',
                ha='center', fontstyle='italic')
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Overall accuracy')
    ax.set_title('A.  Webb accuracy vs dropout', fontweight='bold', loc='left')
    ax.set_ylim(0.3, 1.05)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.axhline(0.25, color='gray', ls=':', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Per-task with task ID
    ax = axes[1]
    task_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']
    for tid, tc in enumerate(TASK_CONFIGS):
        per_task_acc = [webb_results[p]['normal']['per_task'][tc['name']] for p in rates]
        ax.plot(rates, per_task_acc, '-o', color=task_colors[tid],
                label=tc['short'], linewidth=1.5, markersize=5)
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Per-task accuracy')
    ax.set_title('B.  Per-task (with task ID)', fontweight='bold', loc='left')
    ax.set_ylim(0.3, 1.05)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel C: Per-task without task ID
    ax = axes[2]
    for tid, tc in enumerate(TASK_CONFIGS):
        per_task_acc = [webb_results[p]['none']['per_task'][tc['name']] for p in rates]
        ax.plot(rates, per_task_acc, '-o', color=task_colors[tid],
                label=tc['short'], linewidth=1.5, markersize=5)
    ax.set_xlabel('Task ID dropout rate')
    ax.set_ylabel('Per-task accuracy')
    ax.set_title('C.  Per-task (no task ID)', fontweight='bold', loc='left')
    ax.set_ylim(0.3, 1.05)
    ax.axvline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Webb Tasks: Effect of Task ID Dropout Rate (GEE-Separated)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_dropout_ablation_webb.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    log("  Saved: fig_dropout_ablation_webb.png")
    plt.close()


def plot_tradeoff(wcst_results, webb_results):
    """Fig 3: Pareto frontier — with-ID vs without-ID accuracy."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    rates = sorted(wcst_results.keys())

    # WCST curve
    wcst_with = [wcst_results[p]['normal']['accuracy_mean'] for p in rates]
    wcst_without = [wcst_results[p]['none']['accuracy_mean'] for p in rates]
    ax.plot(wcst_with, wcst_without, '-o', color='#E67E22', linewidth=2,
            markersize=8, label='WCST', zorder=5)
    for i, p in enumerate(rates):
        offset_x, offset_y = 0.01, 0.01
        # Adjust label positions to reduce overlap
        if i > 0 and abs(wcst_with[i] - wcst_with[i-1]) < 0.03:
            offset_y = -0.025
        ax.annotate(f'p={p}', (wcst_with[i], wcst_without[i]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color='#E67E22', fontweight='bold')

    # Webb curve
    webb_with = [webb_results[p]['normal']['overall'] for p in rates]
    webb_without = [webb_results[p]['none']['overall'] for p in rates]
    ax.plot(webb_with, webb_without, '-s', color='#2980B9', linewidth=2,
            markersize=8, label='Webb', zorder=5)
    for i, p in enumerate(rates):
        ax.annotate(f'p={p}', (webb_with[i], webb_without[i]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color='#2980B9', fontweight='bold')

    # Diagonal reference (perfect robustness)
    lim_min = 0.3
    lim_max = 1.02
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='gray',
            alpha=0.4, linewidth=1, label='Perfect robustness')

    # Ideal corner annotation
    ax.annotate('Ideal\n(high both)', xy=(0.97, 0.97), fontsize=9,
                color='#555', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F8E8',
                          edgecolor='#27AE60', alpha=0.7))

    ax.set_xlabel('Accuracy WITH task ID')
    ax.set_ylabel('Accuracy WITHOUT task ID')
    ax.set_title('Robustness-Performance Tradeoff\nacross Dropout Rates',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('fig_dropout_tradeoff.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    log("  Saved: fig_dropout_tradeoff.png")
    plt.close()


# =============================================================================
# Summary tables
# =============================================================================

def print_summary(wcst_results, webb_results):
    log()
    log("=" * 70)
    log("  WCST Dropout Ablation Summary")
    log("=" * 70)
    log(f"  {'p':>6}  {'With ID':>8}  {'No ID':>8}  {'Wrong':>8}  "
        f"{'Drop':>7}  {'Persev':>7}  {'Compl':>6}")
    log(f"  {'─' * 60}")
    for p in sorted(wcst_results.keys()):
        r = wcst_results[p]
        norm = r['normal']['accuracy_mean']
        noid = r['none']['accuracy_mean']
        wrong = r['wrong']['accuracy_mean']
        drop = norm - noid
        persev = r['normal']['persev_errors_mean']
        compl = r['normal']['completions_mean']
        marker = " <-- default" if p == 0.5 else ""
        log(f"  {p:>6.2f}  {norm:>7.1%}  {noid:>7.1%}  {wrong:>7.1%}  "
            f"{drop:>+6.1%}  {persev:>6.1f}  {compl:>5.1f}{marker}")

    log()
    log("=" * 70)
    log("  Webb Dropout Ablation Summary")
    log("=" * 70)
    log(f"  {'p':>6}  {'With ID':>8}  {'No ID':>8}  {'Wrong':>8}  {'Drop':>7}")
    log(f"  {'─' * 44}")
    for p in sorted(webb_results.keys()):
        r = webb_results[p]
        norm = r['normal']['overall']
        noid = r['none']['overall']
        wrong = r['wrong']['overall']
        drop = norm - noid
        marker = " <-- default" if p == 0.5 else ""
        log(f"  {p:>6.2f}  {norm:>7.1%}  {noid:>7.1%}  {wrong:>7.1%}  "
            f"{drop:>+6.1%}{marker}")

    # Webb per-task detail
    log()
    log("  Webb per-task (with task ID):")
    header = f"  {'p':>6}"
    for tc in TASK_CONFIGS:
        header += f"  {tc['short']:>8}"
    log(header)
    log(f"  {'─' * (6 + 10 * len(TASK_CONFIGS))}")
    for p in sorted(webb_results.keys()):
        r = webb_results[p]
        row = f"  {p:>6.2f}"
        for tc in TASK_CONFIGS:
            row += f"  {r['normal']['per_task'][tc['name']]:>7.1%}"
        log(row)

    # Find optimal for each
    log()
    log("  Optimal dropout rates:")
    # WCST: maximize with-ID accuracy
    best_wcst_p = max(wcst_results.keys(),
                      key=lambda p: wcst_results[p]['normal']['accuracy_mean'])
    log(f"    WCST (max with-ID acc):  p={best_wcst_p:.2f}  "
        f"acc={wcst_results[best_wcst_p]['normal']['accuracy_mean']:.1%}")
    # Webb: maximize no-ID accuracy while keeping with-ID > 95%
    viable_webb = {p: r for p, r in webb_results.items()
                   if r['normal']['overall'] >= 0.90}
    if viable_webb:
        best_webb_p = max(viable_webb.keys(),
                         key=lambda p: viable_webb[p]['none']['overall'])
        log(f"    Webb (max no-ID, with-ID>=90%): p={best_webb_p:.2f}  "
            f"with={webb_results[best_webb_p]['normal']['overall']:.1%}  "
            f"no={webb_results[best_webb_p]['none']['overall']:.1%}")
    else:
        best_webb_p = max(webb_results.keys(),
                         key=lambda p: webb_results[p]['none']['overall'])
        log(f"    Webb (max no-ID, unconstrained): p={best_webb_p:.2f}  "
            f"with={webb_results[best_webb_p]['normal']['overall']:.1%}  "
            f"no={webb_results[best_webb_p]['none']['overall']:.1%}")


# =============================================================================
# Main
# =============================================================================

def main():
    set_seeds(SEED)
    ensure_dirs()

    device = torch.device('cpu')
    cfg = GEEConfig()

    log("=" * 70)
    log("  Experiment 3: Dropout Rate Ablation")
    log("  GEE-Separated on WCST + Webb Tasks")
    log("=" * 70)
    log(f"  Dropout rates: {DROPOUT_RATES}")
    log(f"  Device: {device}")
    log(f"  value_dim={cfg.value_dim}  key_dim={cfg.key_dim}  hidden={cfg.hidden_dim}")
    log(f"  Results dir: {RESULTS_DIR}")
    log()

    t_start = time.time()

    # ── Part A: WCST ──
    wcst_results = run_wcst_ablation(device, cfg)

    # ── Part B: Webb ──
    webb_results = run_webb_ablation(device, cfg)

    total_time = time.time() - t_start

    # ── Summary ──
    print_summary(wcst_results, webb_results)

    log()
    log(f"  Total time: {total_time/60:.1f} min")
    log()

    # ── Figures ──
    log("  Generating figures...")
    plot_wcst_ablation(wcst_results)
    plot_webb_ablation(webb_results)
    plot_tradeoff(wcst_results, webb_results)

    # ── Save combined summary ──
    combined = {
        'dropout_rates': DROPOUT_RATES,
        'wcst': {f'{p:.2f}': wcst_results[p] for p in sorted(wcst_results.keys())},
        'webb': {f'{p:.2f}': webb_results[p] for p in sorted(webb_results.keys())},
        'total_time_min': total_time / 60,
    }
    save_result('combined_summary.json', combined)

    # ── Save log ──
    log(f"\n  Done. Results in {RESULTS_DIR}/")
    log(f"  Figures: fig_dropout_ablation_wcst.png, fig_dropout_ablation_webb.png, "
        f"fig_dropout_tradeoff.png")


if __name__ == '__main__':
    # Tee output to log file
    import sys
    import io

    class TeeWriter:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log_file = open(LOG_PATH, 'w')
    sys.stdout = TeeWriter(sys.__stdout__, log_file)

    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()
        log(f"  Log saved: {LOG_PATH}")
