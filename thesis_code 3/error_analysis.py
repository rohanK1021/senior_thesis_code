"""
error_analysis.py
=================
Deeper analysis of error patterns under noise. Uses EXISTING results — no
new training.

Part A — Unified Tasks:
  Confusion matrices under h0_only vs rcm_only noise.
  confusion[actual_task][applied_rule] normalized to fractions.
  Shape: (4 tasks) x (4 tasks + "other"), rows normalized.

Part B — WCST:
  Perseverative / random / other error counts.
  Post-switch recovery curves.

Figures:
  fig_error_confusion.png        — side-by-side confusion matrices at sigma=1.0
  fig_error_wcst_recovery.png    — recovery curves per noise type at sigma=1.0
  fig_error_types_detail.png     — stacked bar charts of error type proportions

Run: python -u error_analysis.py
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

TASK_NAMES   = ['S/D', 'RMTS', 'IdRules', 'Dist3']
TASK_KEYS    = ['same_different', 'rmts', 'identity_rules', 'dist3']
COL_LABELS   = ['S/D rule', 'RMTS rule', 'IdRules rule', 'Dist3 rule', 'Other']

COND_LABELS  = {'global': 'Global', 'h0_only': 'h₀ noise', 'rcm_only': 'RCM noise'}

# Color scheme: global=gray, h0=purple, rcm=orange
COND_COLORS  = {
    'global':   '#555555',
    'h0_only':  '#9B59B6',
    'rcm_only': '#E67E22',
}

PLOT_SIGMAS  = ['0.0', '0.5', '1.0', '2.0']
ANAL_SIGMA   = '1.0'          # primary analysis sigma (string key for JSON)
ANAL_SIGMA_F = 1.0            # float key for pickle
RECOVERY_WINDOW = 20          # trials post-switch to track


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_unified():
    if not UNIFIED_JSON.exists():
        print(f'WARNING: {UNIFIED_JSON} not found.')
        return None
    with open(UNIFIED_JSON) as f:
        return json.load(f)


def load_wcst():
    if not WCST_PKL.exists():
        print(f'WARNING: {WCST_PKL} not found.')
        return None
    with open(WCST_PKL, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Part A — Unified task confusion analysis
# ---------------------------------------------------------------------------

def get_confusion(data, cond, sigma_str):
    """Returns confusion matrix as numpy array (4 x 5)."""
    cm = data['confusion'][cond][sigma_str]
    return np.array(cm, dtype=float)


def print_confusion(cm, title):
    print(f'\n  {title}')
    header = f"  {'':12}" + ''.join(f'{c[:10]:>11}' for c in COL_LABELS)
    print(header)
    print('  ' + '-' * (12 + 11 * len(COL_LABELS)))
    for i, task in enumerate(TASK_NAMES):
        row_str = f'  {task:<12}' + ''.join(f'{cm[i,j]*100:>10.1f}%' for j in range(cm.shape[1]))
        print(row_str)


def analyze_confusion(data):
    print('\n' + '='*72)
    print('PART A — Unified Task Confusion Analysis')
    print('='*72)

    for cond in ['global', 'h0_only', 'rcm_only']:
        print(f'\nCondition: {COND_LABELS[cond]}')
        for sigma_str in PLOT_SIGMAS:
            if sigma_str not in data['confusion'][cond]:
                continue
            cm = get_confusion(data, cond, sigma_str)
            diag = [cm[i, i] for i in range(4)]
            print(f'  σ={sigma_str}: accuracy per task = '
                  + ', '.join(f'{TASK_NAMES[i]}={diag[i]*100:.1f}%' for i in range(4))
                  + f'  mean={np.mean(diag)*100:.1f}%')

    # Deeper comparison at ANAL_SIGMA
    print(f'\n--- Detailed comparison at σ={ANAL_SIGMA} ---')
    cm_h0  = get_confusion(data, 'h0_only',  ANAL_SIGMA)
    cm_rcm = get_confusion(data, 'rcm_only', ANAL_SIGMA)
    print_confusion(cm_h0,  f'h₀ noise (σ={ANAL_SIGMA})')
    print_confusion(cm_rcm, f'RCM noise (σ={ANAL_SIGMA})')

    # Which tasks are most vulnerable under each condition?
    print('\n  Task vulnerability comparison (diagonal accuracy):')
    header = f"  {'Task':<14}  {'h0 noise':>12}  {'RCM noise':>12}  {'Δ (h0-rcm)':>12}"
    print(header)
    print('  ' + '-' * 54)
    for i, task in enumerate(TASK_NAMES):
        h0_acc  = cm_h0[i, i]
        rcm_acc = cm_rcm[i, i]
        delta   = h0_acc - rcm_acc
        bar = '←h0 more damaging' if delta < -0.02 else ('←rcm more damaging' if delta > 0.02 else '≈ equal')
        print(f'  {task:<14}  {h0_acc*100:>11.1f}%  {rcm_acc*100:>11.1f}%  {delta*100:>+11.1f}%  {bar}')

    # Most common confusions under h0 noise
    print(f'\n  Top off-diagonal confusions (h₀ noise, σ={ANAL_SIGMA}):')
    offdiag = [(cm_h0[i, j], TASK_NAMES[i], COL_LABELS[j])
               for i in range(4) for j in range(5) if j != i]
    offdiag.sort(reverse=True)
    for val, actual, predicted in offdiag[:5]:
        print(f'    {actual} → {predicted}: {val*100:.1f}%')

    print(f'\n  Top off-diagonal confusions (RCM noise, σ={ANAL_SIGMA}):')
    offdiag_rcm = [(cm_rcm[i, j], TASK_NAMES[i], COL_LABELS[j])
                   for i in range(4) for j in range(5) if j != i]
    offdiag_rcm.sort(reverse=True)
    for val, actual, predicted in offdiag_rcm[:5]:
        print(f'    {actual} → {predicted}: {val*100:.1f}%')


# ---------------------------------------------------------------------------
# Part B — WCST analysis
# ---------------------------------------------------------------------------

def compute_error_type_proportions(sessions):
    """
    Given a list of session dicts, compute mean proportions of each error type.
    Returns (persev_prop, random_prop, other_prop).
    """
    all_persev, all_random, all_other, all_total = [], [], [], []
    for sess in sessions:
        n_persev  = sum(1 for t in sess['trials'] if t['is_perseverative'])
        n_random  = sum(1 for t in sess['trials'] if t['is_random'])
        n_other   = sum(1 for t in sess['trials'] if t['is_other_rule'])
        n_err     = n_persev + n_random + n_other
        if n_err > 0:
            all_persev.append(n_persev / n_err)
            all_random.append(n_random / n_err)
            all_other.append(n_other  / n_err)
            all_total.append(n_err)
    if not all_persev:
        return 0.0, 0.0, 0.0
    return np.mean(all_persev), np.mean(all_random), np.mean(all_other)


def compute_recovery_curve(sessions, window=RECOVERY_WINDOW):
    """
    For each switch point in each session, compute accuracy in the N trials
    following the switch. Returns mean accuracy curve (length=window).

    Trials-to-recovery is reported as first trial where accuracy (over last 3)
    exceeds 0.7 sustained for 3 consecutive trials.
    """
    # Aggregate accuracy at each relative position post-switch
    counts  = np.zeros(window, dtype=float)
    correct = np.zeros(window, dtype=float)

    recovery_lengths = []

    for sess in sessions:
        trials = {t['trial_idx']: t for t in sess['trials']}
        trial_indices = sorted(trials.keys())

        for sp in sess['switch_points']:
            # Trials after switch
            post_switch = [i for i in trial_indices if i >= sp]
            for rel, tidx in enumerate(post_switch[:window]):
                correct[rel] += trials[tidx]['correct']
                counts[rel]  += 1

            # Compute trials-to-recovery for this switch
            correct_streak = 0
            recovery_trial = None
            for rel, tidx in enumerate(post_switch[:window]):
                if trials[tidx]['correct']:
                    correct_streak += 1
                    if correct_streak >= 3:
                        recovery_trial = rel - 2  # first of the 3
                        break
                else:
                    correct_streak = 0
            if recovery_trial is not None:
                recovery_lengths.append(recovery_trial)
            else:
                recovery_lengths.append(window)  # didn't recover

    with np.errstate(divide='ignore', invalid='ignore'):
        acc_curve = np.where(counts > 0, correct / counts, np.nan)

    mean_recovery = np.mean(recovery_lengths) if recovery_lengths else np.nan
    return acc_curve, mean_recovery


def analyze_wcst(data):
    print('\n' + '='*72)
    print('PART B — WCST Error Analysis')
    print('='*72)

    sigma_f = ANAL_SIGMA_F
    conditions = ['global', 'h0_only', 'rcm_only']

    print(f'\nError type proportions at σ={ANAL_SIGMA_F}:')
    header = f"  {'Condition':<14}  {'Persev%':>10}  {'Random%':>10}  {'Other%':>10}  {'Recover(trials)':>16}"
    print(header)
    print('  ' + '-' * 66)

    recovery_data = {}
    for cond in conditions:
        sessions = data['sessions'][cond].get(sigma_f, [])
        if not sessions:
            print(f'  {COND_LABELS[cond]:<14}  (no data)')
            continue
        pp, rp, op = compute_error_type_proportions(sessions)
        acc_curve, mean_rec = compute_recovery_curve(sessions)
        recovery_data[cond] = acc_curve
        print(f'  {COND_LABELS[cond]:<14}  {pp*100:>9.1f}%  {rp*100:>9.1f}%  {op*100:>9.1f}%  {mean_rec:>14.1f}')

    # Additional sigma sweep for error types
    print(f'\nPerseveration proportion across sigma levels:')
    header = f"  {'Condition':<14}" + ''.join(f'  σ={s:>4}' for s in [0.0, 0.5, 1.0, 2.0])
    print(header)
    print('  ' + '-' * 55)
    for cond in conditions:
        row = f'  {COND_LABELS[cond]:<14}'
        for sf in [0.0, 0.5, 1.0, 2.0]:
            sessions = data['sessions'][cond].get(sf, [])
            if not sessions:
                row += '     N/A'
                continue
            pp, rp, op = compute_error_type_proportions(sessions)
            row += f'  {pp*100:>6.1f}%'
        print(row)

    return recovery_data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_error_confusion(data, out_path='fig_error_confusion.png'):
    """Side-by-side confusion matrices for h0 vs RCM noise at sigma=1.0."""
    cm_h0  = get_confusion(data, 'h0_only',  ANAL_SIGMA)
    cm_rcm = get_confusion(data, 'rcm_only', ANAL_SIGMA)

    n_tasks = len(TASK_NAMES)
    n_cols  = len(COL_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Confusion Matrices at σ={ANAL_SIGMA}  (row = actual task, col = rule applied)',
                 fontsize=12, fontweight='bold')

    for ax, cm, cond in [(axes[0], cm_h0, 'h0_only'), (axes[1], cm_rcm, 'rcm_only')]:
        # Only show 4x4 (drop "other" column) for cleaner viz; mention it in title
        cm4 = cm[:, :n_tasks]
        im = ax.imshow(cm4, vmin=0, vmax=1, cmap='Blues', aspect='auto')

        ax.set_xticks(range(n_tasks))
        ax.set_xticklabels(TASK_NAMES, fontsize=10)
        ax.set_yticks(range(n_tasks))
        ax.set_yticklabels(TASK_NAMES, fontsize=10)
        ax.set_xlabel('Rule applied by model', fontsize=11)
        ax.set_ylabel('Actual task', fontsize=11)
        ax.set_title(f'{COND_LABELS[cond]}\n(col 5 = "other" excluded: '
                     f'{", ".join(f"{cm[i,4]*100:.0f}%" for i in range(n_tasks))})',
                     fontsize=10, color=COND_COLORS[cond])

        # Annotate cells
        for i in range(n_tasks):
            for j in range(n_tasks):
                val = cm4[i, j]
                text_color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=text_color, fontweight='bold' if i == j else 'normal')

        # Highlight diagonal
        for i in range(n_tasks):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='gold', lw=2))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def fig_error_wcst_recovery(recovery_data, out_path='fig_error_wcst_recovery.png'):
    """Recovery curves (accuracy vs trials-since-switch) per noise type at sigma=1.0."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(RECOVERY_WINDOW)

    for cond in ['global', 'h0_only', 'rcm_only']:
        if cond not in recovery_data:
            continue
        curve = recovery_data[cond]
        ax.plot(x, curve * 100,
                color=COND_COLORS[cond],
                label=COND_LABELS[cond],
                linewidth=2.0, marker='o', markersize=4)

    ax.axhline(70, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, label='70% threshold')
    ax.set_xlabel('Trials since rule switch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Post-Switch Recovery Curves  (σ={ANAL_SIGMA_F})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, RECOVERY_WINDOW - 1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def fig_error_types_detail(wcst_data, out_path='fig_error_types_detail.png'):
    """
    Stacked bar charts of error type proportions across noise conditions and
    selected sigma values.
    """
    sigmas_f  = [0.0, 0.5, 1.0, 2.0]
    sigmas_lb = [f'σ={s}' for s in sigmas_f]
    conditions = ['global', 'h0_only', 'rcm_only']

    n_cond = len(conditions)
    n_sig  = len(sigmas_f)

    fig, axes = plt.subplots(1, n_cond, figsize=(14, 5), sharey=True)
    fig.suptitle('Error Type Proportions by Noise Condition', fontsize=13, fontweight='bold')

    err_colors = {
        'Perseverative': '#C0392B',
        'Random':        '#3498DB',
        'Other rule':    '#95A5A6',
    }

    for ax, cond in zip(axes, conditions):
        persev_vals, random_vals, other_vals = [], [], []
        valid_labels = []

        for sf in sigmas_f:
            sessions = wcst_data['sessions'][cond].get(sf, [])
            if not sessions:
                continue
            pp, rp, op = compute_error_type_proportions(sessions)
            persev_vals.append(pp * 100)
            random_vals.append(rp * 100)
            other_vals.append(op * 100)
            valid_labels.append(f'σ={sf}')

        x = np.arange(len(valid_labels))
        w = 0.6

        # Stacked bars
        b1 = ax.bar(x, persev_vals, w, color=err_colors['Perseverative'],
                    label='Perseverative', alpha=0.9)
        b2 = ax.bar(x, random_vals, w, bottom=persev_vals,
                    color=err_colors['Random'], label='Random', alpha=0.9)
        bottom_other = [p + r for p, r in zip(persev_vals, random_vals)]
        b3 = ax.bar(x, other_vals, w, bottom=bottom_other,
                    color=err_colors['Other rule'], label='Other rule', alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, fontsize=10)
        ax.set_title(COND_LABELS[cond], fontsize=11, color=COND_COLORS[cond], fontweight='bold')
        ax.set_xlabel('Noise level', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('% of errors', fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Only add legend to first panel
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='upper left')

        # Annotate total error counts
        for i, sf in enumerate(sigmas_f):
            sessions = wcst_data['sessions'][cond].get(sf, [])
            if not sessions:
                continue
            total_errs = [sum(1 for t in s['trials']
                              if t['is_perseverative'] or t['is_random'] or t['is_other_rule'])
                          for s in sessions]
            mean_err = np.mean(total_errs)
            ax.text(i, 101, f'n={mean_err:.0f}', ha='center', va='bottom',
                    fontsize=7.5, color='#333333')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Also: per-task vulnerability across sigmas (Part A extension)
# ---------------------------------------------------------------------------

def fig_task_vulnerability(unified_data, out_path='fig_error_confusion.png'):
    """
    The primary figure: side-by-side confusion matrices for h0 vs RCM at sigma=1.0.
    Delegates to fig_error_confusion.
    """
    fig_error_confusion(unified_data, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data...')
    unified_data = load_unified()
    wcst_data    = load_wcst()

    # ---- Part A: Unified task confusion ---
    if unified_data is not None:
        analyze_confusion(unified_data)
        fig_error_confusion(unified_data, 'fig_error_confusion.png')
    else:
        print('\nSkipping Part A (unified_data not available).')

    # ---- Part B: WCST error analysis ---
    if wcst_data is not None:
        recovery_data = analyze_wcst(wcst_data)
        if recovery_data:
            fig_error_wcst_recovery(recovery_data, 'fig_error_wcst_recovery.png')
        fig_error_types_detail(wcst_data, 'fig_error_types_detail.png')
    else:
        print('\nSkipping Part B (WCST data not available).')

    print('\nDone.')


if __name__ == '__main__':
    main()
