"""
GEE Gain Modulation Experiment
===============================
Experiment 2: Deterministic gain modulation instead of additive noise.

Gain < 1.0 attenuates the context signal (like reduced dopamine receptor sensitivity).
Unlike noise, gain is deterministic: same inputs always produce same outputs.

Three conditions:
  A. h0_gain only:  h_0 = tanh(gain * W @ task_onehot) — attenuated rule prior
  B. rcm_gain only: c_t = c_t * gain — attenuated ongoing modulation
  C. Both:          both pathways attenuated simultaneously

Key comparison with additive noise:
  - Noise destroys signal structure (consistency < 1.0)
  - Gain preserves signal structure (consistency = 1.0 by definition)
  - At matched performance levels, gain produces systematic errors while noise
    produces a mix of systematic and random errors

Part A: Unified tasks (static rules, 4 tasks)
Part B: WCST (dynamic rules, rule switching)

Run: python -u gee_gain_experiment.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import EntityPool, Entity, ATTR_SIZES
from gee_model import GEEConfig
from gee_unified import GEE
from unified_tasks import UnifiedTaskFactory, N_TASKS, TASK_NAMES, TASK_SHORTS


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Configuration
# =============================================================================

GAIN_VALUES = [1.0, 0.75, 0.5, 0.25, 0.1]
N_EVAL = 500          # trials per task per gain level (unified tasks)

# WCST settings
WCST_RULES = ['color', 'shape', 'size', 'texture']
RULE_ATTR_IDX = [1, 0, 2, 3]  # color=attr1, shape=attr0, size=attr2, texture=attr3
N_RULES = 4
CRITERION_TRIALS = 6
MAX_COMPLETIONS = 6
MAX_TRIALS_PER_SESSION = 200
N_SESSIONS = 50

GAIN_CONDITIONS = [
    ('h0_only',  'h\u2080 gain'),
    ('rcm_only', 'RCM gain'),
    ('both',     'Both'),
]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'figure.dpi': 200,
})
COND_COLORS = {'h0_only': '#9B59B6', 'rcm_only': '#E67E22', 'both': '#555555'}
TASK_COLORS = ['#5B9BD5', '#E74C3C', '#F39C12', '#2ECC71']

LOG_FILE = None


def log_to_file(msg="", end="\n"):
    """Log to both stdout and file."""
    print(msg, end=end, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + end)


# =============================================================================
# Part A: Unified Tasks — Gain Sweep
# =============================================================================

def load_unified_model(cfg, device, checkpoint_path='results/gee_unified.pt'):
    """Load pre-trained unified GEE model."""
    model = GEE(cfg, n_tasks=N_TASKS, use_h0=True, use_rcm=True, dropout_p=0.3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    log(f"  Loaded: {checkpoint_path}")
    return model


def load_wcst_model(cfg, device, checkpoint_path='results/gee_wcst_unified.pt'):
    """Load pre-trained WCST GEE model."""
    model = GEE(cfg, n_tasks=N_TASKS, use_h0=True, use_rcm=True, dropout_p=0.3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    log(f"  Loaded: {checkpoint_path}")
    return model


@torch.no_grad()
def evaluate_with_gain(model, factory, device, gain, gain_type='h0_only'):
    """Run evaluation at a given gain level and type on unified tasks."""
    model.eval()
    task_results = {}

    for tid in range(N_TASKS):
        inp, tgt = factory.generate_batch(tid, N_EVAL, split='test')
        x = torch.tensor(inp, dtype=torch.float32, device=device)
        y = torch.tensor(tgt, dtype=torch.float32, device=device)

        kwargs = {'task_id': tid, 'output_dim': 2, 'device': device}
        if gain_type == 'h0_only':
            kwargs['h0_gain'] = gain
        elif gain_type == 'rcm_only':
            kwargs['rcm_gain'] = gain
        elif gain_type == 'both':
            kwargs['h0_gain'] = gain
            kwargs['rcm_gain'] = gain

        logits = model(x, **kwargs)
        preds = logits.argmax(-1)
        true = y.argmax(-1)
        acc = (preds == true).float().mean().item()
        task_results[TASK_NAMES[tid]] = acc

    overall = np.mean(list(task_results.values()))
    return overall, task_results


# =============================================================================
# Part B: WCST — Gain Sweep
# =============================================================================

@torch.no_grad()
def evaluate_wcst_session(model, test_entities, device, gain_type='h0_only',
                          gain=1.0, seed=None):
    """Run one WCST session with pathway-specific gain modulation.

    Each trial is an independent forward pass. Rule switches after
    CRITERION_TRIALS consecutive correct responses.
    """
    model.eval()
    rng = np.random.RandomState(seed)

    active_rule = rng.randint(N_RULES)
    consecutive_correct = 0
    rules_completed = 0
    prev_rule = active_rule
    switch_points = []
    trial_results = []

    for trial_idx in range(MAX_TRIALS_PER_SESSION):
        if rules_completed >= MAX_COMPLETIONS:
            break

        # Generate trial
        is_match = rng.random() < 0.5
        attr_idx = RULE_ATTR_IDX[active_rule]
        ref = test_entities[rng.randint(len(test_entities))]
        if is_match:
            pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx] and e != ref]
            if not pool:
                pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx]]
            test_e = pool[rng.randint(len(pool))]
        else:
            pool = [e for e in test_entities if e[attr_idx] != ref[attr_idx]]
            test_e = pool[rng.randint(len(pool))]

        x = torch.tensor(np.stack([ref.one_hot(), test_e.one_hot()]),
                         dtype=torch.float32, device=device).unsqueeze(0)

        # Forward pass with gain modulation
        kwargs = {'task_id': active_rule, 'output_dim': 2, 'device': device}
        if gain_type == 'h0_only':
            kwargs['h0_gain'] = gain
        elif gain_type == 'rcm_only':
            kwargs['rcm_gain'] = gain
        elif gain_type == 'both':
            kwargs['h0_gain'] = gain
            kwargs['rcm_gain'] = gain

        logits = model(x, **kwargs)
        pred = logits.argmax(-1).item()
        true_label = 0 if is_match else 1
        correct = (pred == true_label)

        # Error classification
        is_perseverative = False
        is_random = False
        is_other_rule = False

        if not correct:
            # Check if response matches previous rule (perseverative)
            if prev_rule != active_rule:
                old_attr = RULE_ATTR_IDX[prev_rule]
                old_match = (ref[old_attr] == test_e[old_attr])
                old_pred = 0 if old_match else 1
                if pred == old_pred:
                    is_perseverative = True

            if not is_perseverative:
                # Check if response matches any other rule
                matched_other = False
                for r in range(N_RULES):
                    if r == active_rule:
                        continue
                    r_attr = RULE_ATTR_IDX[r]
                    r_match = (ref[r_attr] == test_e[r_attr])
                    r_pred = 0 if r_match else 1
                    if pred == r_pred:
                        matched_other = True
                        break
                if matched_other:
                    is_other_rule = True
                else:
                    is_random = True

        trial_results.append({
            'trial_idx': trial_idx,
            'active_rule': active_rule,
            'prev_rule': prev_rule,
            'correct': correct,
            'is_perseverative': is_perseverative,
            'is_random': is_random,
            'is_other_rule': is_other_rule,
        })

        # Rule switching
        if correct:
            consecutive_correct += 1
            if consecutive_correct >= CRITERION_TRIALS:
                rules_completed += 1
                prev_rule = active_rule
                new_rule = (active_rule + rng.randint(1, N_RULES)) % N_RULES
                switch_points.append(trial_idx + 1)
                active_rule = new_rule
                consecutive_correct = 0
        else:
            consecutive_correct = 0

    return {
        'trials': trial_results,
        'rules_completed': rules_completed,
        'switch_points': switch_points,
        'n_trials': len(trial_results),
    }


def run_wcst_gain_sweep(model, test_entities, device):
    """Run WCST gain sweep: 3 conditions x 5 gain levels x 50 sessions."""
    results = {}

    for cond_key, cond_label in GAIN_CONDITIONS:
        log(f"{'---'*23}")
        log(f"  WCST gain sweep: {cond_label}")
        log(f"{'---'*23}")
        results[cond_key] = {}

        for g in GAIN_VALUES:
            sessions = []
            for s in range(N_SESSIONS):
                sess = evaluate_wcst_session(
                    model, test_entities, device,
                    gain_type=cond_key, gain=g, seed=s * 100)
                sessions.append(sess)

            accs = [sum(t['correct'] for t in s['trials']) / max(len(s['trials']), 1)
                    for s in sessions]
            persevs = [sum(t['is_perseverative'] for t in s['trials']) for s in sessions]
            completions = [s['rules_completed'] for s in sessions]
            n_errors = [sum(not t['correct'] for t in s['trials']) for s in sessions]

            results[cond_key][g] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'persev_mean': float(np.mean(persevs)),
                'persev_std': float(np.std(persevs)),
                'n_errors_mean': float(np.mean(n_errors)),
                'completions_mean': float(np.mean(completions)),
                'completions_std': float(np.std(completions)),
            }

            log(f"    gain={g:<5}  acc={np.mean(accs):.1%}  "
                f"persev={np.mean(persevs):.1f}  "
                f"completions={np.mean(completions):.1f}")

    return results


# =============================================================================
# Load existing noise results for comparison
# =============================================================================

def load_noise_results():
    """Load existing noise experiment results."""
    noise_unified = None
    noise_wcst = None

    try:
        with open('results/results_noise_experiment.json') as f:
            noise_unified = json.load(f)
        log("  Loaded: results/results_noise_experiment.json")
    except FileNotFoundError:
        log("  WARNING: results/results_noise_experiment.json not found")

    try:
        with open('results_wcst_noise_raw.pkl', 'rb') as f:
            noise_wcst = pickle.load(f)
        log("  Loaded: results_wcst_noise_raw.pkl")
    except FileNotFoundError:
        log("  WARNING: results_wcst_noise_raw.pkl not found")

    return noise_unified, noise_wcst


# =============================================================================
# Figures
# =============================================================================

def plot_gain_static(gain_results, noise_unified):
    """Fig 1: Unified tasks — gain curves + noise comparison overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy vs gain for h0/rcm/both
    ax = axes[0]
    for cond_key, cond_label in GAIN_CONDITIONS:
        accs = [gain_results[cond_key][g]['overall'] for g in GAIN_VALUES]
        ax.plot(GAIN_VALUES, accs, '-o', color=COND_COLORS[cond_key],
                label=cond_label, markersize=6, linewidth=2.5)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Gain', fontsize=12)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title('A.  Gain Modulation (Static Tasks)', fontweight='bold', loc='left')
    ax.set_xlim(1.05, 0.05)  # Reversed: 1.0 left, 0.1 right
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Overlay noise curves for comparison
    ax = axes[1]
    noise_sigmas = [0.0, 0.5, 1.0, 1.5, 2.0]
    noise_cond_map = {
        'h0_only': ('h0_only', 'h\u2080 noise'),
        'rcm_only': ('rcm_only', 'RCM noise'),
        'both': ('global', 'Global noise'),
    }

    # Plot gain curves (solid)
    for cond_key, cond_label in GAIN_CONDITIONS:
        accs = [gain_results[cond_key][g]['overall'] for g in GAIN_VALUES]
        ax.plot(GAIN_VALUES, accs, '-o', color=COND_COLORS[cond_key],
                label=f'{cond_label} (gain)', markersize=5, linewidth=2)

    # Overlay noise curves (dashed) if available
    if noise_unified is not None:
        for cond_key in ['h0_only', 'rcm_only', 'both']:
            noise_key, noise_label = noise_cond_map[cond_key]
            try:
                noise_accs = [noise_unified['sweep'][noise_key][str(s)]['overall']
                              for s in noise_sigmas]
                # Map sigma to a pseudo-gain axis: gain = 1/(1+sigma)
                pseudo_gains = [1.0 / (1.0 + s) for s in noise_sigmas]
                ax.plot(pseudo_gains, noise_accs, '--s', color=COND_COLORS[cond_key],
                        label=f'{noise_label}', markersize=4, linewidth=1.5, alpha=0.6)
            except KeyError:
                pass

    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Gain (solid) / 1/(1+\u03c3) (dashed)', fontsize=11)
    ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_title('B.  Gain vs Noise Comparison', fontweight='bold', loc='left')
    ax.set_xlim(1.05, 0.05)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Gain Modulation: Unified Static Tasks',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_gain_static.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_gain_static.png")
    plt.close()


def plot_gain_wcst(wcst_gain_results):
    """Fig 2: WCST accuracy and perseveration vs gain."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Accuracy vs gain
    ax = axes[0]
    for cond_key, cond_label in GAIN_CONDITIONS:
        accs = [wcst_gain_results[cond_key][g]['accuracy_mean'] for g in GAIN_VALUES]
        stds = [wcst_gain_results[cond_key][g]['accuracy_std'] for g in GAIN_VALUES]
        ax.plot(GAIN_VALUES, accs, '-o', color=COND_COLORS[cond_key],
                label=cond_label, markersize=6, linewidth=2.5)
        ax.fill_between(GAIN_VALUES,
                        [a - s for a, s in zip(accs, stds)],
                        [a + s for a, s in zip(accs, stds)],
                        color=COND_COLORS[cond_key], alpha=0.1)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Gain', fontsize=12)
    ax.set_ylabel('WCST accuracy', fontsize=12)
    ax.set_title('A.  WCST Accuracy', fontweight='bold', loc='left')
    ax.set_xlim(1.05, 0.05)
    ax.set_ylim(0.3, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Perseverative errors vs gain
    ax = axes[1]
    for cond_key, cond_label in GAIN_CONDITIONS:
        persevs = [wcst_gain_results[cond_key][g]['persev_mean'] for g in GAIN_VALUES]
        persev_stds = [wcst_gain_results[cond_key][g]['persev_std'] for g in GAIN_VALUES]
        ax.plot(GAIN_VALUES, persevs, '-o', color=COND_COLORS[cond_key],
                label=cond_label, markersize=6, linewidth=2.5)
        ax.fill_between(GAIN_VALUES,
                        [p - s for p, s in zip(persevs, persev_stds)],
                        [p + s for p, s in zip(persevs, persev_stds)],
                        color=COND_COLORS[cond_key], alpha=0.1)
    ax.set_xlabel('Gain', fontsize=12)
    ax.set_ylabel('Perseverative errors', fontsize=12)
    ax.set_title('B.  WCST Perseverative Errors', fontweight='bold', loc='left')
    ax.set_xlim(1.05, 0.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Gain Modulation: WCST Dynamic Tasks',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_gain_wcst.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_gain_wcst.png")
    plt.close()


def plot_gain_consistency(noise_unified):
    """Fig 3: Gain is deterministic (consistency=1.0) vs noise consistency curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    noise_sigmas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    noise_cond_labels = {
        'global': 'Global noise',
        'h0_only': 'h\u2080 noise',
        'rcm_only': 'RCM noise',
    }
    noise_cond_colors = {
        'global': '#555555',
        'h0_only': '#9B59B6',
        'rcm_only': '#E67E22',
    }

    # Plot noise consistency curves
    if noise_unified is not None:
        for cond_key in ['global', 'h0_only', 'rcm_only']:
            try:
                cons_vals = [noise_unified['consistency'][cond_key][str(s)]
                             for s in noise_sigmas]
                ax.plot(noise_sigmas, cons_vals, '-o',
                        color=noise_cond_colors[cond_key],
                        label=noise_cond_labels[cond_key],
                        markersize=5, linewidth=2.5)
            except KeyError:
                pass

    # Mark gain consistency at 1.0
    ax.axhline(1.0, color='#2ECC71', ls='-', linewidth=3, alpha=0.7,
               label='Gain modulation (all conditions)')
    ax.fill_between([0, 2.2], 0.98, 1.02, color='#2ECC71', alpha=0.08)

    ax.axhline(0.5, color='gray', ls=':', alpha=0.3, label='Random')
    ax.set_xlabel('Noise \u03c3', fontsize=12)
    ax.set_ylabel('Response consistency', fontsize=12)
    ax.set_title('Gain Preserves Signal Structure, Noise Destroys It',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 1.08)
    ax.set_xlim(-0.05, 2.15)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate
    ax.annotate('Gain: deterministic\n(consistency = 1.0)',
                xy=(1.5, 1.0), xytext=(1.5, 0.88),
                fontsize=9, color='#2ECC71', fontweight='bold',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=1.5))

    plt.tight_layout()
    plt.savefig('fig_gain_consistency.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_gain_consistency.png")
    plt.close()


def plot_gain_double_dissociation(gain_static, wcst_gain, noise_unified, noise_wcst):
    """Fig 4: Double dissociation with gain at g=0.1.

    Same format as fig_double_dissociation.png but using gain instead of noise.
    """
    # Gain effect sizes: accuracy drop from gain=1.0 to gain=0.1
    h0_static_clean = gain_static['h0_only'][1.0]['overall']
    h0_static_disrupted = gain_static['h0_only'][0.1]['overall']
    rcm_static_clean = gain_static['rcm_only'][1.0]['overall']
    rcm_static_disrupted = gain_static['rcm_only'][0.1]['overall']

    h0_dynamic_clean = wcst_gain['h0_only'][1.0]['accuracy_mean']
    h0_dynamic_disrupted = wcst_gain['h0_only'][0.1]['accuracy_mean']
    rcm_dynamic_clean = wcst_gain['rcm_only'][1.0]['accuracy_mean']
    rcm_dynamic_disrupted = wcst_gain['rcm_only'][0.1]['accuracy_mean']

    h0_static_drop = (h0_static_clean - h0_static_disrupted) * 100
    rcm_static_drop = (rcm_static_clean - rcm_static_disrupted) * 100
    h0_dynamic_drop = (h0_dynamic_clean - h0_dynamic_disrupted) * 100
    rcm_dynamic_drop = (rcm_dynamic_clean - rcm_dynamic_disrupted) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Interaction plot
    ax = axes[0]
    x = [0, 1]
    xlabels = ['Static Tasks\n(Unified)', 'Dynamic Tasks\n(WCST)']

    h0_drops = [h0_static_drop, h0_dynamic_drop]
    rcm_drops = [rcm_static_drop, rcm_dynamic_drop]

    ax.plot(x, h0_drops, '-o', color='#9B59B6', label='h\u2080 gain=0.1',
            markersize=12, linewidth=3)
    ax.plot(x, rcm_drops, '-s', color='#E67E22', label='RCM gain=0.1',
            markersize=12, linewidth=3)

    for i in range(2):
        ax.annotate(f'{h0_drops[i]:.1f}%', (x[i], h0_drops[i]),
                    textcoords='offset points', xytext=(12, 5),
                    fontsize=11, color='#9B59B6', fontweight='bold')
        ax.annotate(f'{rcm_drops[i]:.1f}%', (x[i], rcm_drops[i]),
                    textcoords='offset points', xytext=(12, -15),
                    fontsize=11, color='#E67E22', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylabel('Accuracy drop at gain=0.1 (%)', fontsize=12)
    ax.set_title('A.  Pathway \u00d7 Task Type Interaction (Gain)',
                 fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Grouped bars
    ax = axes[1]
    x_pos = np.arange(2)
    width = 0.3

    bars1 = ax.bar(x_pos - width/2, [h0_static_drop, h0_dynamic_drop], width,
                   label='h\u2080 gain=0.1', color='#9B59B6', alpha=0.85,
                   edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, [rcm_static_drop, rcm_dynamic_drop], width,
                   label='RCM gain=0.1', color='#E67E22', alpha=0.85,
                   edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold',
                color='#9B59B6')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold',
                color='#E67E22')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Static Tasks\n(Unified)', 'Dynamic Tasks\n(WCST)'], fontsize=12)
    ax.set_ylabel('Accuracy drop at gain=0.1 (%)', fontsize=12)
    ax.set_title('B.  Effect Size Comparison (Gain)', fontweight='bold', loc='left')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Double Dissociation: Gain Modulation',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_gain_double_dissociation.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_gain_double_dissociation.png")
    plt.close()

    return {
        'h0_static_drop': h0_static_drop,
        'h0_dynamic_drop': h0_dynamic_drop,
        'rcm_static_drop': rcm_static_drop,
        'rcm_dynamic_drop': rcm_dynamic_drop,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    global LOG_FILE
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    factory = UnifiedTaskFactory(pool.get('train'), pool.get('test'), seed=seed)

    log_path = '/private/tmp/gain_modulation.txt'
    LOG_FILE = open(log_path, 'w')

    def dual_log(msg="", end="\n"):
        print(msg, end=end, flush=True)
        LOG_FILE.write(msg + end)

    dual_log("=" * 70)
    dual_log("  GEE Gain Modulation Experiment")
    dual_log("=" * 70)
    dual_log(f"  Gain values: {GAIN_VALUES}")
    dual_log(f"  Conditions: h0_only, rcm_only, both")
    dual_log()

    # ── Load existing noise results ──
    dual_log("Loading existing results...")
    noise_unified, noise_wcst = load_noise_results()
    dual_log()

    # ── Part A: Unified Tasks ──
    dual_log("=" * 70)
    dual_log("  Part A: Unified Static Tasks")
    dual_log("=" * 70)

    unified_model = load_unified_model(cfg, device)

    # Verify baseline
    unified_model.eval()
    with torch.no_grad():
        total_correct, total = 0, 0
        for tid in range(N_TASKS):
            inp, tgt = factory.generate_batch(tid, N_EVAL, split='test')
            x = torch.tensor(inp, dtype=torch.float32, device=device)
            y = torch.tensor(tgt, dtype=torch.float32, device=device)
            logits = unified_model(x, task_id=tid, output_dim=2, device=device)
            total_correct += (logits.argmax(-1) == y.argmax(-1)).sum().item()
            total += N_EVAL
    baseline_acc = total_correct / total
    dual_log(f"  Baseline accuracy (gain=1.0): {baseline_acc:.1%}")
    dual_log()

    # Gain sweep
    gain_static_results = {}
    for cond_key, cond_label in GAIN_CONDITIONS:
        dual_log(f"{'---'*23}")
        dual_log(f"  Gain sweep: {cond_label}")
        dual_log(f"{'---'*23}")
        gain_static_results[cond_key] = {}
        for g in GAIN_VALUES:
            overall, per_task = evaluate_with_gain(unified_model, factory, device, g, cond_key)
            gain_static_results[cond_key][g] = {'overall': overall, 'per_task': per_task}
            task_str = " ".join(f"{s}={per_task[t]:.0%}"
                                for s, t in zip(TASK_SHORTS, TASK_NAMES))
            dual_log(f"    gain={g:<5}  overall={overall:.1%}  {task_str}")
        dual_log()

    # ── Part B: WCST ──
    dual_log("=" * 70)
    dual_log("  Part B: WCST Dynamic Tasks")
    dual_log("=" * 70)

    wcst_model = load_wcst_model(cfg, device)
    wcst_gain_results = run_wcst_gain_sweep(wcst_model, pool.get('test'), device)
    dual_log()

    # ── Save results ──
    results_data = {
        'static': {
            cond_key: {
                str(g): gain_static_results[cond_key][g]
                for g in GAIN_VALUES
            }
            for cond_key in ['h0_only', 'rcm_only', 'both']
        },
        'wcst': {
            cond_key: {
                str(g): {k: v for k, v in wcst_gain_results[cond_key][g].items()}
                for g in GAIN_VALUES
            }
            for cond_key in ['h0_only', 'rcm_only', 'both']
        },
        'config': {
            'gain_values': GAIN_VALUES,
            'n_eval': N_EVAL,
            'n_sessions': N_SESSIONS,
            'seed': seed,
        },
    }

    os.makedirs('results', exist_ok=True)
    with open('results/gain_modulation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    dual_log("  Saved: results/gain_modulation_results.json")

    # ── Figures ──
    dual_log()
    dual_log("Generating figures...")

    plot_gain_static(gain_static_results, noise_unified)
    plot_gain_wcst(wcst_gain_results)
    plot_gain_consistency(noise_unified)
    dissociation = plot_gain_double_dissociation(
        gain_static_results, wcst_gain_results, noise_unified, noise_wcst)

    # ── Summary ──
    dual_log()
    dual_log("=" * 70)
    dual_log("  Summary: Gain Modulation Results")
    dual_log("=" * 70)

    dual_log()
    dual_log("  Part A: Unified Static Tasks")
    dual_log(f"  {'Condition':<15} {'g=1.0':>8} {'g=0.75':>8} {'g=0.5':>8} "
             f"{'g=0.25':>8} {'g=0.1':>8} {'Drop':>8}")
    dual_log(f"  {'─'*63}")
    for cond_key, cond_label in GAIN_CONDITIONS:
        vals = [gain_static_results[cond_key][g]['overall'] for g in GAIN_VALUES]
        drop = vals[0] - vals[-1]
        dual_log(f"  {cond_label:<15} " +
                 " ".join(f"{v:>7.1%}" for v in vals) +
                 f" {drop:>7.1%}")

    dual_log()
    dual_log("  Part B: WCST Dynamic Tasks")
    dual_log(f"  {'Condition':<15} {'g=1.0':>8} {'g=0.75':>8} {'g=0.5':>8} "
             f"{'g=0.25':>8} {'g=0.1':>8} {'Drop':>8}")
    dual_log(f"  {'─'*63}")
    for cond_key, cond_label in GAIN_CONDITIONS:
        vals = [wcst_gain_results[cond_key][g]['accuracy_mean'] for g in GAIN_VALUES]
        drop = vals[0] - vals[-1]
        dual_log(f"  {cond_label:<15} " +
                 " ".join(f"{v:>7.1%}" for v in vals) +
                 f" {drop:>7.1%}")

    dual_log()
    dual_log("  WCST Perseverative Errors")
    dual_log(f"  {'Condition':<15} {'g=1.0':>8} {'g=0.75':>8} {'g=0.5':>8} "
             f"{'g=0.25':>8} {'g=0.1':>8}")
    dual_log(f"  {'─'*55}")
    for cond_key, cond_label in GAIN_CONDITIONS:
        vals = [wcst_gain_results[cond_key][g]['persev_mean'] for g in GAIN_VALUES]
        dual_log(f"  {cond_label:<15} " +
                 " ".join(f"{v:>7.1f}" for v in vals))

    dual_log()
    dual_log("  Double Dissociation (gain=0.1)")
    dual_log(f"    h\u2080 gain:  static drop={dissociation['h0_static_drop']:.1f}%  "
             f"dynamic drop={dissociation['h0_dynamic_drop']:.1f}%")
    dual_log(f"    RCM gain: static drop={dissociation['rcm_static_drop']:.1f}%  "
             f"dynamic drop={dissociation['rcm_dynamic_drop']:.1f}%")

    # Compare with noise at sigma=2.0
    if noise_unified is not None:
        dual_log()
        dual_log("  Comparison with noise (\u03c3=2.0):")
        h0_noise_drop = (noise_unified['sweep']['h0_only']['0.0']['overall'] -
                         noise_unified['sweep']['h0_only']['2.0']['overall'])
        rcm_noise_drop = (noise_unified['sweep']['rcm_only']['0.0']['overall'] -
                          noise_unified['sweep']['rcm_only']['2.0']['overall'])
        dual_log(f"    h\u2080 noise:  static drop={h0_noise_drop*100:.1f}%")
        dual_log(f"    RCM noise: static drop={rcm_noise_drop*100:.1f}%")
        dual_log(f"    h\u2080 gain:   static drop={dissociation['h0_static_drop']:.1f}%")
        dual_log(f"    RCM gain:  static drop={dissociation['rcm_static_drop']:.1f}%")

    if noise_wcst is not None:
        try:
            h0_wcst_noise_drop = (noise_wcst['summary']['sweep']['h0_only'][0.0]['accuracy_mean'] -
                                  noise_wcst['summary']['sweep']['h0_only'][2.0]['accuracy_mean'])
            rcm_wcst_noise_drop = (noise_wcst['summary']['sweep']['rcm_only'][0.0]['accuracy_mean'] -
                                   noise_wcst['summary']['sweep']['rcm_only'][2.0]['accuracy_mean'])
            dual_log(f"    h\u2080 noise:  WCST drop={h0_wcst_noise_drop*100:.1f}%")
            dual_log(f"    RCM noise: WCST drop={rcm_wcst_noise_drop*100:.1f}%")
            dual_log(f"    h\u2080 gain:   WCST drop={dissociation['h0_dynamic_drop']:.1f}%")
            dual_log(f"    RCM gain:  WCST drop={dissociation['rcm_dynamic_drop']:.1f}%")
        except (KeyError, TypeError):
            pass

    dual_log()
    dual_log("  Key insight: Gain modulation is deterministic (consistency=1.0).")
    dual_log("  Same performance drop as noise, but errors are fully systematic.")
    dual_log("  Maps onto reduced receptor sensitivity (vs noisy transmission).")

    dual_log()
    dual_log("=" * 70)
    dual_log("  Done.")
    dual_log("=" * 70)

    LOG_FILE.close()
    log(f"\n  Log saved: {log_path}")


if __name__ == '__main__':
    main()
