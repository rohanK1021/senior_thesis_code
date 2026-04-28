"""
WCST Pathway-Specific Noise Experiment
=======================================
Tests double dissociation hypothesis:
  h_0 is load-bearing for static tasks, RCM is load-bearing for dynamic tasks.

Uses the unified GEE (h_0 + RCM) from gee_unified.py, trained from scratch on WCST.
Three noise conditions: global (task_onehot), h_0-only, RCM-only.

Key analyses:
  1. Accuracy and perseveration vs sigma (3 conditions)
  2. Post-switch recovery curves
  3. Error type classification (perseverative / random / other)
  4. Response consistency
  5. Double dissociation figure (combining with unified-task results from S4.9)

Design notes:
  - Each WCST trial is an independent forward pass (no inter-trial hidden state)
  - h_0 is set per-trial with current rule's task_id
  - RCM receives current rule's task_id at each trial
  - Rule switching is performance-dependent (6 consecutive correct)
  - Noise freshly sampled per trial

Run: python -u gee_wcst_noise.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import EntityPool, Entity, ATTR_SIZES
from gee_model import GEEConfig
from gee_unified import GEE


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# WCST Configuration
# =============================================================================

WCST_RULES = ['color', 'shape', 'size', 'texture']
RULE_ATTR_IDX = [1, 0, 2, 3]  # color=attr1, shape=attr0, size=attr2, texture=attr3
N_RULES = 4
CRITERION_TRIALS = 6
MAX_COMPLETIONS = 6
MAX_TRIALS_PER_SESSION = 200

SIGMA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_SESSIONS = 50
N_CONSISTENCY = 200   # fixed trials for consistency test
N_REPS = 20           # repetitions per trial

NOISE_CONDITIONS = [
    ('global',   'Global'),
    ('h0_only',  'h\u2080 only'),
    ('rcm_only', 'RCM only'),
]

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'figure.dpi': 200,
})
COND_COLORS = ['#555555', '#9B59B6', '#E67E22']


# =============================================================================
# WCST Data Generation (same as gee_wcst.py)
# =============================================================================

def generate_wcst_data(entities, n_per_rule, rng):
    """Generate WCST training data: attribute-specific Same/Different for all 4 rules."""
    inputs, labels, task_ids = [], [], []
    for rule in range(N_RULES):
        attr_idx = RULE_ATTR_IDX[rule]
        for _ in range(n_per_rule):
            is_match = rng.random() < 0.5
            ref = entities[rng.randint(len(entities))]
            if is_match:
                pool = [e for e in entities if e[attr_idx] == ref[attr_idx] and e != ref]
                if not pool:
                    pool = [e for e in entities if e[attr_idx] == ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]
                label = np.array([1., 0.], dtype=np.float32)
            else:
                pool = [e for e in entities if e[attr_idx] != ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]
                label = np.array([0., 1.], dtype=np.float32)

            seq = np.stack([ref.one_hot(), test_e.one_hot()])
            inputs.append(seq)
            labels.append(label)
            task_ids.append(rule)

    inputs = np.array(inputs, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    task_ids = np.array(task_ids, dtype=np.int64)
    perm = rng.permutation(len(inputs))
    return inputs[perm], labels[perm], task_ids[perm]


# =============================================================================
# Training
# =============================================================================

def train_wcst_unified(model, train_entities, device, lr=5e-4, n_epochs=50,
                       n_per_rule=1000, seed=42):
    """Train unified GEE on WCST (attribute-specific Same/Different)."""
    rng = np.random.RandomState(seed)
    inputs, labels, task_ids = generate_wcst_data(train_entities, n_per_rule, rng)

    ds = torch.utils.data.TensorDataset(
        torch.tensor(inputs), torch.tensor(labels), torch.tensor(task_ids))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b, tid_b in loader:
            x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)

            # Group by task_id (unified GEE requires scalar task_id)
            batch_logits = torch.zeros(x_b.shape[0], 2, device=device)
            for tid in tid_b.unique():
                mask = tid_b == tid
                batch_logits[mask] = model(x_b[mask], task_id=tid.item(),
                                           output_dim=2, device=device)

            loss = F.cross_entropy(batch_logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x_b.shape[0]
            correct += (batch_logits.argmax(-1) == y_b.argmax(-1)).sum().item()
            total += x_b.shape[0]

        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")

    return model


# =============================================================================
# WCST Session Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_wcst_session(model, test_entities, device, noise_type='global',
                          noise_sigma=0.0, seed=None):
    """Run one WCST session with pathway-specific noise.

    Each trial is an independent forward pass. h_0 and RCM both receive
    the current rule's task_id. Rule switches after CRITERION_TRIALS
    consecutive correct responses.

    Returns dict with trial-by-trial results, switch points, etc.
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

        # Forward pass with pathway-specific noise
        kwargs = {'task_id': active_rule, 'output_dim': 2, 'device': device}
        if noise_type == 'global':
            kwargs['noise_sigma'] = noise_sigma
        elif noise_type == 'h0_only':
            kwargs['h0_noise_sigma'] = noise_sigma
        elif noise_type == 'rcm_only':
            kwargs['rcm_noise_sigma'] = noise_sigma

        logits = model(x, **kwargs)
        pred = logits.argmax(-1).item()
        true_label = 0 if is_match else 1
        correct = (pred == true_label)

        # ── Error classification ──
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


# =============================================================================
# Noise Sweep
# =============================================================================

def run_noise_sweep(model, test_entities, device):
    """Run pathway-specific noise sweep: 3 conditions x 9 sigmas x 50 sessions."""
    results = {}

    for cond_key, cond_label in NOISE_CONDITIONS:
        log(f"{'─'*70}")
        log(f"  Noise sweep: {cond_label}")
        log(f"{'─'*70}")
        results[cond_key] = {}

        for sigma in SIGMA_VALUES:
            sessions = []
            for s in range(N_SESSIONS):
                sess = evaluate_wcst_session(
                    model, test_entities, device,
                    noise_type=cond_key, noise_sigma=sigma, seed=s * 100)
                sessions.append(sess)

            accs = [sum(t['correct'] for t in s['trials']) / max(len(s['trials']), 1)
                    for s in sessions]
            persevs = [sum(t['is_perseverative'] for t in s['trials']) for s in sessions]
            randoms = [sum(t['is_random'] for t in s['trials']) for s in sessions]
            others = [sum(t['is_other_rule'] for t in s['trials']) for s in sessions]
            completions = [s['rules_completed'] for s in sessions]
            n_errors = [sum(not t['correct'] for t in s['trials']) for s in sessions]

            results[cond_key][sigma] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'persev_mean': float(np.mean(persevs)),
                'persev_std': float(np.std(persevs)),
                'random_mean': float(np.mean(randoms)),
                'other_mean': float(np.mean(others)),
                'n_errors_mean': float(np.mean(n_errors)),
                'completions_mean': float(np.mean(completions)),
                'completions_std': float(np.std(completions)),
                'all_sessions': sessions,
            }

            log(f"    \u03c3={sigma:<4}  acc={np.mean(accs):.1%}  "
                f"persev={np.mean(persevs):.1f}  completions={np.mean(completions):.1f}")
        log()

    return results


# =============================================================================
# Post-switch Recovery Curves
# =============================================================================

def compute_recovery_curve(all_sessions, max_offset=15):
    """Compute accuracy at each trial offset relative to rule switches."""
    offsets = {i: [] for i in range(-5, max_offset + 1)}

    for sess in all_sessions:
        for sp in sess['switch_points']:
            for t in sess['trials']:
                rel = t['trial_idx'] - sp
                if -5 <= rel <= max_offset:
                    offsets[rel].append(1.0 if t['correct'] else 0.0)

    positions = list(range(-5, max_offset + 1))
    means = [np.mean(offsets[p]) if offsets[p] else np.nan for p in positions]
    sems = [np.std(offsets[p]) / max(np.sqrt(len(offsets[p])), 1) if offsets[p]
            else np.nan for p in positions]
    return positions, means, sems


# =============================================================================
# Consistency Analysis
# =============================================================================

@torch.no_grad()
def compute_wcst_consistency(model, test_entities, device, noise_type, noise_sigma):
    """Generate fixed WCST trials and measure response consistency under noise.

    Batches trials by rule for efficiency.
    """
    model.eval()
    rng = np.random.RandomState(42)

    # Generate fixed trials: N_CONSISTENCY/N_RULES per rule
    per_rule = N_CONSISTENCY // N_RULES
    rule_batches = {}  # rule -> (x_batch_tensor,)

    for rule in range(N_RULES):
        attr_idx = RULE_ATTR_IDX[rule]
        seqs = []
        for _ in range(per_rule):
            is_match = rng.random() < 0.5
            ref = test_entities[rng.randint(len(test_entities))]
            if is_match:
                pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx] and e != ref]
                if not pool:
                    pool = [e for e in test_entities if e[attr_idx] == ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]
            else:
                pool = [e for e in test_entities if e[attr_idx] != ref[attr_idx]]
                test_e = pool[rng.randint(len(pool))]
            seqs.append(np.stack([ref.one_hot(), test_e.one_hot()]))

        rule_batches[rule] = torch.tensor(np.array(seqs), dtype=torch.float32, device=device)

    # Run each batch N_REPS times with fresh noise
    consistencies = []
    for rule in range(N_RULES):
        x_batch = rule_batches[rule]
        all_preds = []

        for _ in range(N_REPS):
            kwargs = {'task_id': rule, 'output_dim': 2, 'device': device}
            if noise_type == 'global':
                kwargs['noise_sigma'] = noise_sigma
            elif noise_type == 'h0_only':
                kwargs['h0_noise_sigma'] = noise_sigma
            elif noise_type == 'rcm_only':
                kwargs['rcm_noise_sigma'] = noise_sigma

            logits = model(x_batch, **kwargs)
            all_preds.append(logits.argmax(-1).cpu().numpy())

        all_preds = np.stack(all_preds, axis=0)  # (N_REPS, per_rule)

        for trial_idx in range(per_rule):
            trial_preds = all_preds[:, trial_idx]
            mode_count = np.bincount(trial_preds).max()
            consistencies.append(mode_count / N_REPS)

    return float(np.mean(consistencies))


# =============================================================================
# Figures
# =============================================================================

def plot_accuracy(sweep_results, gee_direct_data=None):
    """Fig 1: Accuracy vs sigma, three noise conditions."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        accs = [sweep_results[cond_key][s]['accuracy_mean'] for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, accs, '-o', color=COND_COLORS[i], label=cond_label,
                markersize=5, linewidth=2.5)

    # GEE-Direct comparison from S4.7 (if available)
    if gee_direct_data:
        gd_sigmas = sorted([float(k) for k in gee_direct_data.keys()])
        gd_accs = [gee_direct_data[k]['accuracy']
                   for k in sorted(gee_direct_data.keys(), key=float)]
        ax.plot(gd_sigmas, gd_accs, '--s', color='#E74C3C',
                label='GEE-Direct (\u00a74.7)', markersize=4, linewidth=1.5, alpha=0.7)

    ax.axhline(0.5, color='gray', ls=':', alpha=0.3, label='Chance')
    ax.set_xlabel('Noise \u03c3')
    ax.set_ylabel('Accuracy')
    ax.set_title('WCST Accuracy Under Pathway-Specific Noise',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_wcst_noise_accuracy.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_wcst_noise_accuracy.png")
    plt.close()


def plot_perseveration(sweep_results):
    """Fig 2: Perseverative errors vs sigma."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        persevs = [sweep_results[cond_key][s]['persev_mean'] for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, persevs, '-o', color=COND_COLORS[i], label=cond_label,
                markersize=5, linewidth=2.5)

    ax.set_xlabel('Noise \u03c3')
    ax.set_ylabel('Perseverative errors per session')
    ax.set_title('WCST Perseveration Under Pathway-Specific Noise',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_wcst_noise_perseveration.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_wcst_noise_perseveration.png")
    plt.close()


def plot_recovery(sweep_results):
    """Fig 3: Post-switch recovery curves. One panel per noise condition."""
    selected_sigmas = [0.0, 0.25, 0.5, 1.0, 2.0]
    sigma_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.9, len(selected_sigmas)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax_idx, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        ax = axes[ax_idx]
        for s_idx, sigma in enumerate(selected_sigmas):
            sessions = sweep_results[cond_key][sigma]['all_sessions']
            positions, means, sems = compute_recovery_curve(sessions)
            means_arr = np.array(means, dtype=float)
            sems_arr = np.array(sems, dtype=float)
            ax.plot(positions, means_arr, '-o', color=sigma_colors[s_idx],
                    label=f'\u03c3={sigma}', markersize=3, linewidth=1.8)
            valid = ~np.isnan(means_arr) & ~np.isnan(sems_arr)
            pos_v = np.array(positions)[valid]
            ax.fill_between(pos_v, means_arr[valid] - sems_arr[valid],
                            means_arr[valid] + sems_arr[valid],
                            color=sigma_colors[s_idx], alpha=0.1)

        ax.axvline(0, color='red', ls='--', alpha=0.5)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel('Trials relative to rule switch')
        ax.set_title(cond_label, fontweight='bold')
        ax.set_ylim(0.2, 1.05)
        ax.grid(True, alpha=0.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel('Accuracy')
            ax.legend(fontsize=8)

    plt.suptitle('Post-Switch Recovery Under Noise', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_wcst_noise_recovery.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_wcst_noise_recovery.png")
    plt.close()


def plot_error_types(sweep_results):
    """Fig 4: Error type breakdown — stacked bars by condition x selected sigmas."""
    selected_sigmas = [0.0, 0.5, 1.0, 2.0]
    n_conds = len(NOISE_CONDITIONS)
    n_sigmas = len(selected_sigmas)

    fig, axes = plt.subplots(1, n_conds, figsize=(4.5 * n_conds, 5), sharey=True)

    for ax_idx, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        ax = axes[ax_idx]
        x = np.arange(n_sigmas)
        width = 0.6

        persev_vals, random_vals, other_vals = [], [], []
        for sigma in selected_sigmas:
            r = sweep_results[cond_key][sigma]
            persev_vals.append(r['persev_mean'])
            random_vals.append(r['random_mean'])
            other_vals.append(r['other_mean'])

        persev_vals = np.array(persev_vals)
        random_vals = np.array(random_vals)
        other_vals = np.array(other_vals)

        ax.bar(x, persev_vals, width, label='Perseverative', color='#E74C3C', alpha=0.85)
        ax.bar(x, random_vals, width, bottom=persev_vals,
               label='Random', color='#F39C12', alpha=0.85)
        ax.bar(x, other_vals, width, bottom=persev_vals + random_vals,
               label='Other rule', color='#95A5A6', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([f'\u03c3={s}' for s in selected_sigmas])
        ax.set_title(cond_label, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel('Errors per session')
            ax.legend(fontsize=9)

    plt.suptitle('Error Type Classification Under Noise',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_wcst_noise_error_types.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_wcst_noise_error_types.png")
    plt.close()


def plot_double_dissociation(wcst_results, unified_results=None):
    """Fig 5: THE THESIS FIGURE — pathway x task-type interaction.

    Pulls unified-task data from S4.9 results (or hardcoded fallback).
    Shows crossover: h_0 matters more for static, RCM for dynamic.
    """
    # WCST effect sizes (accuracy drop from sigma=0 to sigma=2.0)
    h0_wcst_clean = wcst_results['h0_only'][0.0]['accuracy_mean']
    h0_wcst_noisy = wcst_results['h0_only'][2.0]['accuracy_mean']
    rcm_wcst_clean = wcst_results['rcm_only'][0.0]['accuracy_mean']
    rcm_wcst_noisy = wcst_results['rcm_only'][2.0]['accuracy_mean']

    h0_wcst_drop = h0_wcst_clean - h0_wcst_noisy
    rcm_wcst_drop = rcm_wcst_clean - rcm_wcst_noisy

    # Unified-task effect sizes from S4.9
    if unified_results:
        try:
            h0_uni_clean = unified_results['sweep']['h0_only']['0.0']['overall']
            h0_uni_noisy = unified_results['sweep']['h0_only']['2.0']['overall']
            rcm_uni_clean = unified_results['sweep']['rcm_only']['0.0']['overall']
            rcm_uni_noisy = unified_results['sweep']['rcm_only']['2.0']['overall']
        except (KeyError, TypeError):
            h0_uni_clean, h0_uni_noisy = 0.932, 0.693
            rcm_uni_clean, rcm_uni_noisy = 0.916, 0.852
    else:
        # Hardcoded from CLAUDE.md S4.9 results
        h0_uni_clean, h0_uni_noisy = 0.932, 0.693
        rcm_uni_clean, rcm_uni_noisy = 0.916, 0.852

    h0_uni_drop = h0_uni_clean - h0_uni_noisy
    rcm_uni_drop = rcm_uni_clean - rcm_uni_noisy

    dissociation = {
        'h0_static_drop': h0_uni_drop,
        'h0_dynamic_drop': h0_wcst_drop,
        'rcm_static_drop': rcm_uni_drop,
        'rcm_dynamic_drop': rcm_wcst_drop,
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Interaction plot
    ax = axes[0]
    x = [0, 1]
    xlabels = ['Static Tasks\n(Unified)', 'Dynamic Tasks\n(WCST)']

    h0_drops = [h0_uni_drop * 100, h0_wcst_drop * 100]
    rcm_drops = [rcm_uni_drop * 100, rcm_wcst_drop * 100]

    ax.plot(x, h0_drops, '-o', color='#9B59B6', label='h\u2080 noise',
            markersize=12, linewidth=3)
    ax.plot(x, rcm_drops, '-s', color='#E67E22', label='RCM noise',
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
    ax.set_ylabel('Accuracy drop at \u03c3=2.0 (%)', fontsize=12)
    ax.set_title('A.  Pathway \u00d7 Task Type Interaction', fontweight='bold', loc='left')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Grouped bar chart
    ax = axes[1]
    x_pos = np.arange(2)
    width = 0.3

    bars1 = ax.bar(x_pos - width/2, [h0_uni_drop * 100, h0_wcst_drop * 100], width,
                   label='h\u2080 noise', color='#9B59B6', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, [rcm_uni_drop * 100, rcm_wcst_drop * 100], width,
                   label='RCM noise', color='#E67E22', alpha=0.85, edgecolor='white')

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
    ax.set_ylabel('Accuracy drop at \u03c3=2.0 (%)', fontsize=12)
    ax.set_title('B.  Effect Size Comparison', fontweight='bold', loc='left')
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('Double Dissociation: Pathway \u00d7 Task Type',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_double_dissociation.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_double_dissociation.png")
    plt.close()

    return dissociation


def plot_consistency(consistency_results):
    """Fig 6: Response consistency vs sigma."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        vals = [consistency_results[cond_key][s] for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, vals, '-o', color=COND_COLORS[i], label=cond_label,
                markersize=5, linewidth=2.5)

    ax.axhline(0.5, color='gray', ls=':', alpha=0.3, label='Random')
    ax.set_xlabel('Noise \u03c3')
    ax.set_ylabel('Response consistency (1=deterministic, 0.5=random)')
    ax.set_title('WCST Response Consistency Under Noise',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_wcst_noise_consistency.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_wcst_noise_consistency.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    train_entities = pool.get('train')
    test_entities = pool.get('test')

    log("=" * 70)
    log("  WCST Pathway-Specific Noise Experiment")
    log("  Model: Unified GEE (h\u2080 + RCM) from gee_unified.py")
    log("=" * 70)
    log(f"  Train entities: {len(train_entities)}  Test: {len(test_entities)}")
    log(f"  Rules: {WCST_RULES}")
    log(f"  Criterion: {CRITERION_TRIALS} consecutive correct")
    log(f"  Design: h\u2080 set per-trial, RCM receives task_id per-trial")
    log(f"  Noise levels: {SIGMA_VALUES}")
    log(f"  Sessions per condition: {N_SESSIONS}")
    log()

    # ── Step 1: Train unified GEE on WCST ──
    log("=" * 70)
    log("  Step 1: Train unified GEE on WCST")
    log("=" * 70)

    checkpoint_path = 'results/gee_wcst_unified.pt'
    model = GEE(cfg, n_tasks=N_RULES, use_h0=True, use_rcm=True,
                dropout_p=0.3).to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                         weights_only=True))
        log(f"  Loaded checkpoint: {checkpoint_path}")
    except (FileNotFoundError, RuntimeError):
        log("  No checkpoint found. Training from scratch...")
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        train_wcst_unified(model, train_entities, device, lr=5e-4, n_epochs=50,
                           n_per_rule=1000, seed=seed)
        elapsed = time.time() - t0
        log(f"  Training: {elapsed:.0f}s")
        torch.save(model.state_dict(), checkpoint_path)
        log(f"  Saved: {checkpoint_path}")

    # Verify baseline
    model.eval()
    with torch.no_grad():
        rng = np.random.RandomState(99)
        inputs, labels, task_ids = generate_wcst_data(test_entities, 200, rng)
        x_all = torch.tensor(inputs, dtype=torch.float32, device=device)
        y_all = torch.tensor(labels, dtype=torch.float32, device=device)
        tid_all = torch.tensor(task_ids, dtype=torch.int64, device=device)

        total_correct, total = 0, 0
        for tid in range(N_RULES):
            mask = tid_all == tid
            if mask.sum() > 0:
                logits = model(x_all[mask], task_id=tid, output_dim=2, device=device)
                total_correct += (logits.argmax(-1) == y_all[mask].argmax(-1)).sum().item()
                total += mask.sum().item()

    baseline_acc = total_correct / total
    log(f"  Baseline accuracy: {baseline_acc:.1%}")
    if baseline_acc < 0.80:
        log(f"  WARNING: Baseline accuracy low ({baseline_acc:.1%}). Results may be unreliable.")

    # Quick WCST session sanity check
    sess_test = evaluate_wcst_session(model, test_entities, device,
                                      noise_type='global', noise_sigma=0.0, seed=0)
    test_acc = sum(t['correct'] for t in sess_test['trials']) / len(sess_test['trials'])
    log(f"  Session sanity check: acc={test_acc:.1%}  "
        f"completions={sess_test['rules_completed']}  "
        f"trials={sess_test['n_trials']}")
    log()

    # ── Step 2: Noise Sweep ──
    log("=" * 70)
    log("  Step 2: Pathway-Specific Noise Sweep")
    log(f"  {len(NOISE_CONDITIONS)} conditions x {len(SIGMA_VALUES)} sigmas x "
        f"{N_SESSIONS} sessions = {len(NOISE_CONDITIONS)*len(SIGMA_VALUES)*N_SESSIONS} sessions")
    log("=" * 70)
    t0 = time.time()
    sweep_results = run_noise_sweep(model, test_entities, device)
    log(f"  Noise sweep completed: {time.time()-t0:.0f}s")

    # ── Step 3: Consistency ──
    log(f"\n{'='*70}")
    log("  Step 3: Response Consistency")
    log(f"{'='*70}")
    consistency_results = {}
    for cond_key, cond_label in NOISE_CONDITIONS:
        consistency_results[cond_key] = {}
        for sigma in SIGMA_VALUES:
            c = compute_wcst_consistency(model, test_entities, device, cond_key, sigma)
            consistency_results[cond_key][sigma] = c
            log(f"    {cond_label:<25} \u03c3={sigma:<4}  consistency={c:.3f}")
    log()

    # ── Load existing results for comparison ──
    unified_results = None
    try:
        with open('results/results_noise_experiment.json', 'r') as f:
            unified_results = json.load(f)
        log("  Loaded unified-task noise results for double dissociation comparison")
    except FileNotFoundError:
        log("  No unified-task noise results found. Using hardcoded values from S4.9.")

    gee_direct_data = None
    try:
        with open('results/wcst_results.json', 'r') as f:
            wcst_prev = json.load(f)
            gee_direct_data = wcst_prev.get('noise', {}).get('GEE-Direct', None)
        if gee_direct_data:
            log("  Loaded GEE-Direct WCST noise data for accuracy comparison")
    except FileNotFoundError:
        pass

    # ── Figures ──
    log("\n  Generating figures...")
    plot_accuracy(sweep_results, gee_direct_data)
    plot_perseveration(sweep_results)
    plot_recovery(sweep_results)
    plot_error_types(sweep_results)
    dissociation = plot_double_dissociation(sweep_results, unified_results)
    plot_consistency(consistency_results)

    # ── Save results ──
    log("\n  Saving results...")

    # Text output
    with open('results_wcst_noise.txt', 'w') as f:
        f.write("WCST Pathway-Specific Noise Experiment Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Model: Unified GEE (h_0 + RCM) from gee_unified.py\n")
        f.write("Design: h_0 set per-trial with current rule's task_id\n")
        f.write("        RCM receives current rule's task_id at each trial\n")
        f.write(f"Sessions per condition: {N_SESSIONS}\n")
        f.write(f"Criterion: {CRITERION_TRIALS} consecutive correct\n\n")

        for cond_key, cond_label in NOISE_CONDITIONS:
            f.write(f"\n{cond_label}\n{'-'*50}\n")
            f.write(f"  {'sigma':<6} {'Acc':>8} {'Persev':>8} {'Random':>8} "
                    f"{'Other':>8} {'Compl':>8}\n")
            for sigma in SIGMA_VALUES:
                r = sweep_results[cond_key][sigma]
                f.write(f"  {sigma:<6.2f} {r['accuracy_mean']:>7.1%} "
                        f"{r['persev_mean']:>7.1f} {r['random_mean']:>7.1f} "
                        f"{r['other_mean']:>7.1f} {r['completions_mean']:>7.1f}\n")

        f.write(f"\n\nConsistency\n{'='*50}\n")
        for cond_key, cond_label in NOISE_CONDITIONS:
            f.write(f"\n{cond_label}\n")
            for sigma in SIGMA_VALUES:
                f.write(f"  sigma={sigma:<4}  {consistency_results[cond_key][sigma]:.3f}\n")

        if dissociation:
            f.write(f"\n\nDouble Dissociation (accuracy drop at sigma=2.0)\n{'='*50}\n")
            f.write(f"  {'':>15} {'Static':>10} {'Dynamic':>10}\n")
            f.write(f"  {'h_0 noise':>15} {dissociation['h0_static_drop']:>9.1%} "
                    f"{dissociation['h0_dynamic_drop']:>9.1%}\n")
            f.write(f"  {'RCM noise':>15} {dissociation['rcm_static_drop']:>9.1%} "
                    f"{dissociation['rcm_dynamic_drop']:>9.1%}\n")

    log("  Saved: results_wcst_noise.txt")

    # Raw data (pickle)
    summary_data = {
        'sweep': {ck: {s: {k: v for k, v in d.items() if k != 'all_sessions'}
                       for s, d in cond.items()}
                  for ck, cond in sweep_results.items()},
        'consistency': consistency_results,
        'dissociation': dissociation,
        'config': {
            'sigma_values': SIGMA_VALUES,
            'n_sessions': N_SESSIONS,
            'criterion_trials': CRITERION_TRIALS,
            'max_completions': MAX_COMPLETIONS,
            'n_rules': N_RULES,
        },
    }

    # Full session data for detailed post-hoc analysis
    full_sessions = {ck: {s: d['all_sessions'] for s, d in cond.items()}
                     for ck, cond in sweep_results.items()}

    with open('results_wcst_noise_raw.pkl', 'wb') as f:
        pickle.dump({'summary': summary_data, 'sessions': full_sessions}, f)
    log("  Saved: results_wcst_noise_raw.pkl")

    # ── Summary ──
    log(f"\n{'='*70}")
    log("  Summary")
    log(f"{'='*70}")

    s0, s05, s10, s20 = "\u03c3=0", "\u03c3=0.5", "\u03c3=1.0", "\u03c3=2.0"

    log("\n  Accuracy:")
    log(f"  {'Condition':<25} {s0:>8} {s05:>8} {s10:>8} {s20:>8}")
    log(f"  {'─'*57}")
    for cond_key, cond_label in NOISE_CONDITIONS:
        accs = [sweep_results[cond_key][s]['accuracy_mean'] for s in [0.0, 0.5, 1.0, 2.0]]
        log(f"  {cond_label:<25} {accs[0]:>7.1%} {accs[1]:>7.1%} "
            f"{accs[2]:>7.1%} {accs[3]:>7.1%}")

    log("\n  Perseverative errors:")
    log(f"  {'Condition':<25} {s0:>8} {s05:>8} {s10:>8} {s20:>8}")
    log(f"  {'─'*57}")
    for cond_key, cond_label in NOISE_CONDITIONS:
        persevs = [sweep_results[cond_key][s]['persev_mean'] for s in [0.0, 0.5, 1.0, 2.0]]
        log(f"  {cond_label:<25} {persevs[0]:>7.1f} {persevs[1]:>7.1f} "
            f"{persevs[2]:>7.1f} {persevs[3]:>7.1f}")

    log("\n  Consistency:")
    log(f"  {'Condition':<25} {s0:>8} {s10:>8} {s20:>8}")
    log(f"  {'─'*49}")
    for cond_key, cond_label in NOISE_CONDITIONS:
        vals = [consistency_results[cond_key][s] for s in [0.0, 1.0, 2.0]]
        log(f"  {cond_label:<25} {vals[0]:>7.3f} {vals[1]:>7.3f} {vals[2]:>7.3f}")

    if dissociation:
        h0_label = "h\u2080 noise"
        log(f"\n  Double dissociation (accuracy drop at {s20}):")
        log(f"  {'':>18} {'Static':>10} {'Dynamic':>10}")
        log(f"  {'─'*40}")
        h0s = dissociation['h0_static_drop']
        h0d = dissociation['h0_dynamic_drop']
        rcms = dissociation['rcm_static_drop']
        rcmd = dissociation['rcm_dynamic_drop']
        log(f"  {h0_label:>18} {h0s:>9.1%} {h0d:>9.1%}")
        log(f"  {'RCM noise':>18} {rcms:>9.1%} {rcmd:>9.1%}")

        # Interpret
        h0_larger_static = h0s > rcms
        rcm_larger_dynamic = rcmd > h0d
        if h0_larger_static and rcm_larger_dynamic:
            log("\n  RESULT: Double dissociation confirmed!")
            log("    h\u2080 more important for static tasks, RCM more important for dynamic tasks.")
        elif h0_larger_static:
            log("\n  RESULT: Partial dissociation.")
            log("    h\u2080 more important for static tasks, but no reversal on dynamic tasks.")
        else:
            log("\n  RESULT: No crossover detected.")
            log("    Same pathway dominates both task types.")

    log(f"\n{'='*70}")
    log("  Done.")


if __name__ == '__main__':
    main()
