"""
GEE Noise Injection & Cognitive Disorganization Experiment
============================================================
Extends the WCST noise finding to the unified tasks.

Three noise conditions:
  A. Global: noise on task_onehot → affects both h_0 and RCM
  B. h_0 only: noise on initial hidden state → disrupts rule prior
  C. RCM only: noise on context output → disrupts ongoing modulation

Key analyses:
  1. Accuracy vs σ per task
  2. 4×4 confusion matrix (which rule does the model apply?)
  3. Response consistency (systematic vs random errors)

Run: python -u gee_noise_experiment.py
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
from matplotlib.colors import LinearSegmentedColormap

from dataset import EntityPool
from gee_model import GEEConfig
from gee_unified import GEE
from unified_tasks import UnifiedTaskFactory, SEQ_LEN, N_TASKS, TASK_NAMES, TASK_SHORTS
from gee_main import train_model


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


SIGMA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_EVAL = 500       # trials per task per noise level
N_CONSISTENCY = 200  # unique trials for consistency test
N_REPS = 20         # repetitions per trial for consistency

NOISE_CONDITIONS = [
    ('global',   'Global (task_onehot)'),
    ('h0_only',  'h₀ only'),
    ('rcm_only', 'RCM only'),
]


# =============================================================================
# Step 1: Train or load model
# =============================================================================

def get_trained_model(cfg, factory, device, checkpoint_path='results/gee_unified.pt'):
    model = GEE(cfg, n_tasks=N_TASKS, use_h0=True, use_rcm=True, dropout_p=0.3).to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        log(f"  Loaded checkpoint: {checkpoint_path}")
    except (FileNotFoundError, RuntimeError):
        log("  No checkpoint found. Training from scratch...")
        t0 = time.time()
        train_model(model, factory, device, n_epochs=50, batches_per_task=100,
                    batch_size=64, lr=1e-4)
        log(f"  Training: {time.time()-t0:.0f}s")
        torch.save(model.state_dict(), checkpoint_path)
        log(f"  Saved: {checkpoint_path}")

    # Verify performance
    model.eval()
    with torch.no_grad():
        total_correct, total = 0, 0
        for tid in range(N_TASKS):
            inp, tgt = factory.generate_batch(tid, 500, split='test')
            x = torch.tensor(inp, dtype=torch.float32, device=device)
            y = torch.tensor(tgt, dtype=torch.float32, device=device)
            logits = model(x, task_id=tid, output_dim=2, device=device)
            total_correct += (logits.argmax(-1) == y.argmax(-1)).sum().item()
            total += 500
    acc = total_correct / total
    log(f"  Baseline accuracy: {acc:.1%}")
    assert acc > 0.85, f"Model accuracy too low ({acc:.1%}), something is wrong"
    return model


# =============================================================================
# Step 2 & 4: Noise sweep with per-task accuracy
# =============================================================================

@torch.no_grad()
def evaluate_with_noise(model, factory, device, sigma, noise_type='global'):
    """Run evaluation at a given noise level and type."""
    model.eval()
    task_results = {}

    for tid in range(N_TASKS):
        inp, tgt = factory.generate_batch(tid, N_EVAL, split='test')
        x = torch.tensor(inp, dtype=torch.float32, device=device)
        y = torch.tensor(tgt, dtype=torch.float32, device=device)

        kwargs = {'task_id': tid, 'output_dim': 2, 'device': device}
        if noise_type == 'global':
            kwargs['noise_sigma'] = sigma
        elif noise_type == 'h0_only':
            kwargs['h0_noise_sigma'] = sigma
        elif noise_type == 'rcm_only':
            kwargs['rcm_noise_sigma'] = sigma

        logits = model(x, **kwargs)
        preds = logits.argmax(-1)
        true = y.argmax(-1)
        acc = (preds == true).float().mean().item()
        task_results[TASK_NAMES[tid]] = acc

    overall = np.mean(list(task_results.values()))
    return overall, task_results


# =============================================================================
# Step 3: Confusion matrix analysis
# =============================================================================

@torch.no_grad()
def compute_confusion_matrix(model, factory, device, sigma, noise_type='global'):
    """For trials generated under each rule, check which rule the model
    effectively applies by comparing its answer to all rules' correct answers."""
    model.eval()
    entities = factory.test_entities

    # confusion[actual_rule][effective_rule] = count
    confusion = np.zeros((N_TASKS, N_TASKS + 1), dtype=np.float64)  # +1 for "other"
    counts = np.zeros(N_TASKS, dtype=np.float64)

    for actual_tid in range(N_TASKS):
        inp, tgt = factory.generate_batch(actual_tid, N_EVAL, split='test')
        x = torch.tensor(inp, dtype=torch.float32, device=device)

        kwargs = {'task_id': actual_tid, 'output_dim': 2, 'device': device}
        if noise_type == 'global':
            kwargs['noise_sigma'] = sigma
        elif noise_type == 'h0_only':
            kwargs['h0_noise_sigma'] = sigma
        elif noise_type == 'rcm_only':
            kwargs['rcm_noise_sigma'] = sigma

        logits = model(x, **kwargs)
        preds = logits.argmax(-1).cpu().numpy()

        for i in range(N_EVAL):
            # Reconstruct entities from one-hot
            trial_entities = []
            for pos in range(SEQ_LEN):
                oh = inp[i, pos]
                # Decode one-hot back to Entity
                shape = int(np.argmax(oh[0:4]))
                color = int(np.argmax(oh[4:8]))
                size = int(np.argmax(oh[8:10]))
                texture = int(np.argmax(oh[10:12]))
                from dataset import Entity
                trial_entities.append(Entity(shape, color, size, texture))

            pred = preds[i]
            matched_any = False

            for rule_tid in range(N_TASKS):
                answer = factory.compute_answer(rule_tid, trial_entities)
                if answer is not None and pred == answer:
                    confusion[actual_tid, rule_tid] += 1
                    matched_any = True

            if not matched_any:
                confusion[actual_tid, N_TASKS] += 1  # "other"

            counts[actual_tid] += 1

    # Normalize rows to fractions
    for i in range(N_TASKS):
        if counts[i] > 0:
            confusion[i] /= counts[i]

    return confusion


# =============================================================================
# Step 5: Consistency analysis
# =============================================================================

@torch.no_grad()
def compute_consistency(model, factory, device, sigma, noise_type='global'):
    """Run the same trials multiple times with fresh noise. Measure consistency."""
    model.eval()
    consistencies = []

    for tid in range(N_TASKS):
        inp, tgt = factory.generate_batch(tid, N_CONSISTENCY, split='test')
        x = torch.tensor(inp, dtype=torch.float32, device=device)

        all_preds = []
        for rep in range(N_REPS):
            kwargs = {'task_id': tid, 'output_dim': 2, 'device': device}
            if noise_type == 'global':
                kwargs['noise_sigma'] = sigma
            elif noise_type == 'h0_only':
                kwargs['h0_noise_sigma'] = sigma
            elif noise_type == 'rcm_only':
                kwargs['rcm_noise_sigma'] = sigma

            logits = model(x, **kwargs)
            preds = logits.argmax(-1).cpu().numpy()
            all_preds.append(preds)

        all_preds = np.stack(all_preds, axis=0)  # (N_REPS, N_CONSISTENCY)

        # Per-trial consistency: fraction matching the mode
        for trial_idx in range(N_CONSISTENCY):
            trial_preds = all_preds[:, trial_idx]
            mode_count = np.bincount(trial_preds).max()
            consistencies.append(mode_count / N_REPS)

    return np.mean(consistencies)


# =============================================================================
# Figures
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'figure.dpi': 200,
})
TASK_COLORS = ['#5B9BD5', '#E74C3C', '#F39C12', '#2ECC71']
COND_COLORS = ['#555555', '#9B59B6', '#E67E22']


def plot_accuracy_curves(sweep_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax_idx, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        ax = axes[ax_idx]
        for tid, (tname, tshort, tc) in enumerate(zip(TASK_NAMES, TASK_SHORTS, TASK_COLORS)):
            accs = [sweep_results[cond_key][s]['per_task'][tname] for s in SIGMA_VALUES]
            ax.plot(SIGMA_VALUES, accs, '-o', color=tc, label=tshort,
                    markersize=4, linewidth=2)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel('Noise σ')
        ax.set_title(cond_label, fontweight='bold')
        ax.set_ylim(0.2, 1.05)
        ax.grid(True, alpha=0.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel('Test accuracy')
            ax.legend(fontsize=8)
    plt.suptitle('Accuracy Under Context Noise', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_noise_accuracy_curves.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_noise_accuracy_curves.png")
    plt.close()


def plot_confusion_matrices(confusion_results):
    selected_sigmas = [0.0, 0.25, 0.5, 1.0]
    n_conds = len(NOISE_CONDITIONS)
    n_sigmas = len(selected_sigmas)

    fig, axes = plt.subplots(n_conds, n_sigmas, figsize=(3.5 * n_sigmas, 3.2 * n_conds))

    for row, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        for col, sigma in enumerate(selected_sigmas):
            ax = axes[row, col]
            cm = confusion_results[cond_key][sigma][:, :N_TASKS]  # drop "other" column
            im = ax.imshow(cm, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
            for i in range(N_TASKS):
                for j in range(N_TASKS):
                    color = 'white' if cm[i, j] > 0.6 else 'black'
                    ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center',
                            fontsize=8, color=color)
            ax.set_xticks(range(N_TASKS))
            ax.set_yticks(range(N_TASKS))
            if row == n_conds - 1:
                ax.set_xticklabels(TASK_SHORTS, fontsize=8)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_yticklabels(TASK_SHORTS, fontsize=8)
                ax.set_ylabel(cond_label, fontsize=10, fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(f'σ = {sigma}', fontsize=10)

    fig.suptitle('Rule Confusion Under Noise (rows=actual rule, cols=model\'s effective rule)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('fig_noise_confusion_matrices.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_noise_confusion_matrices.png")
    plt.close()


def plot_consistency(consistency_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        vals = [consistency_results[cond_key][s] for s in SIGMA_VALUES]
        ax.plot(SIGMA_VALUES, vals, '-o', color=COND_COLORS[i], label=cond_label,
                markersize=5, linewidth=2.5)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3, label='Random')
    ax.set_xlabel('Noise σ')
    ax.set_ylabel('Response consistency (1=always same, 0.5=random)')
    ax.set_title('Response Consistency Under Noise', fontsize=14, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('fig_noise_consistency.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_noise_consistency.png")
    plt.close()


def plot_default_rule(confusion_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax_idx, (cond_key, cond_label) in enumerate(NOISE_CONDITIONS):
        ax = axes[ax_idx]
        # For each σ, compute the average off-diagonal confusion per target rule
        for target_tid in range(N_TASKS):
            fracs = []
            for sigma in SIGMA_VALUES:
                cm = confusion_results[cond_key][sigma][:, :N_TASKS]
                # Average fraction of trials (across actual rules != target) that match target rule
                off_diag = [cm[actual, target_tid] for actual in range(N_TASKS) if actual != target_tid]
                fracs.append(np.mean(off_diag))
            ax.plot(SIGMA_VALUES, fracs, '-o', color=TASK_COLORS[target_tid],
                    label=TASK_SHORTS[target_tid], markersize=4, linewidth=2)
        ax.set_xlabel('Noise σ')
        ax.set_title(cond_label, fontweight='bold')
        ax.set_ylim(-0.02, 0.8)
        ax.grid(True, alpha=0.12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax_idx == 0:
            ax.set_ylabel('Fraction applying this rule (off-diagonal)')
            ax.legend(fontsize=8)
    plt.suptitle('Default Rule Under Noise (which rule does the model fall back to?)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_noise_default_rule.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_noise_default_rule.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device('cpu')
    cfg = GEEConfig()

    pool = EntityPool(n_train=40, n_test=16, seed=seed)
    factory = UnifiedTaskFactory(pool.get('train'), pool.get('test'), seed=seed)

    log("=" * 70)
    log("  GEE Noise Injection & Cognitive Disorganization Experiment")
    log("=" * 70)
    log()

    # Step 1: Get trained model
    model = get_trained_model(cfg, factory, device)
    log()

    # Step 2 & 4: Noise sweep
    sweep_results = {}
    for cond_key, cond_label in NOISE_CONDITIONS:
        log(f"{'─'*70}")
        log(f"  Noise sweep: {cond_label}")
        log(f"{'─'*70}")
        sweep_results[cond_key] = {}
        for sigma in SIGMA_VALUES:
            overall, per_task = evaluate_with_noise(model, factory, device, sigma, cond_key)
            sweep_results[cond_key][sigma] = {'overall': overall, 'per_task': per_task}
            task_str = " ".join(f"{s}={per_task[t]:.0%}" for s, t in zip(TASK_SHORTS, TASK_NAMES))
            log(f"    σ={sigma:<4}  overall={overall:.1%}  {task_str}")
        log()

    # Step 3: Confusion matrices
    log(f"{'─'*70}")
    log("  Computing confusion matrices...")
    log(f"{'─'*70}")
    confusion_results = {}
    selected_sigmas = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    for cond_key, cond_label in NOISE_CONDITIONS:
        confusion_results[cond_key] = {}
        for sigma in selected_sigmas:
            cm = compute_confusion_matrix(model, factory, device, sigma, cond_key)
            confusion_results[cond_key][sigma] = cm
            if sigma in [0.0, 0.5, 1.0]:
                log(f"  {cond_label} σ={sigma}:")
                for i in range(N_TASKS):
                    row = " ".join(f"{cm[i,j]:.2f}" for j in range(N_TASKS))
                    log(f"    {TASK_SHORTS[i]:<8} → {row}  other={cm[i,N_TASKS]:.2f}")
    log()

    # Step 5: Consistency
    log(f"{'─'*70}")
    log("  Computing response consistency...")
    log(f"{'─'*70}")
    consistency_results = {}
    for cond_key, cond_label in NOISE_CONDITIONS:
        consistency_results[cond_key] = {}
        for sigma in SIGMA_VALUES:
            c = compute_consistency(model, factory, device, sigma, cond_key)
            consistency_results[cond_key][sigma] = c
            log(f"    {cond_label:<25} σ={sigma:<4}  consistency={c:.3f}")
    log()

    # Save raw results
    results_data = {
        'sweep': sweep_results,
        'confusion': {ck: {s: cm.tolist() for s, cm in d.items()}
                      for ck, d in confusion_results.items()},
        'consistency': consistency_results,
    }
    with open('results/results_noise_experiment.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    log("  Saved: results/results_noise_experiment.json")

    # Figures
    log("\n  Generating figures...")
    plot_accuracy_curves(sweep_results)
    plot_confusion_matrices(confusion_results)
    plot_consistency(consistency_results)
    plot_default_rule(confusion_results)

    # Summary
    log(f"\n{'='*70}")
    log("  Summary")
    log(f"{'='*70}")
    for cond_key, cond_label in NOISE_CONDITIONS:
        log(f"\n  {cond_label}:")
        log(f"    σ=0.0: {sweep_results[cond_key][0.0]['overall']:.1%}  "
            f"σ=0.5: {sweep_results[cond_key][0.5]['overall']:.1%}  "
            f"σ=1.0: {sweep_results[cond_key][1.0]['overall']:.1%}  "
            f"σ=2.0: {sweep_results[cond_key][2.0]['overall']:.1%}")
        log(f"    Consistency: σ=0: {consistency_results[cond_key][0.0]:.3f}  "
            f"σ=1: {consistency_results[cond_key][1.0]:.3f}  "
            f"σ=2: {consistency_results[cond_key][2.0]:.3f}")

    log(f"\n{'='*70}")
    log("  Done.")


if __name__ == '__main__':
    main()
