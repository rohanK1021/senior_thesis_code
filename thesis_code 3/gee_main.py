"""
GEE Main Experiment — Unified Tasks
=====================================
Tests the unified GEE model on fixed-length tasks where context is
genuinely required (all tasks have seq_len=6, binary output).

Three models:
  ESBN:     GEE with use_h0=False, use_rcm=False (no context)
  GEE:      Full model (h_0 + RCM + dropout)
  GEE-h0:   GEE with use_rcm=False (h_0 only, no RCM)

Evaluations:
  A. Multi-task accuracy (with/without/wrong task_id)
  B. Per-task breakdown
  C. Sequence length generalization

Run: python -u gee_main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import EntityPool
from gee_model import GEEConfig
from gee_unified import GEE
from unified_tasks import UnifiedTaskFactory, SEQ_LEN, N_TASKS, TASK_NAMES, TASK_SHORTS


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Training
# =============================================================================

def train_model(model, factory, device, n_epochs=50, batches_per_task=100,
                batch_size=64, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for task_id in range(N_TASKS):
            for _ in range(batches_per_task):
                inputs, targets = factory.generate_batch(task_id, batch_size, split='train')
                x = torch.tensor(inputs, dtype=torch.float32, device=device)
                y = torch.tensor(targets, dtype=torch.float32, device=device)

                logits = model(x, task_id=task_id, output_dim=2, device=device)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * x.shape[0]
                correct += (logits.argmax(-1) == y.argmax(-1)).sum().item()
                total += x.shape[0]

        if epoch % 5 == 0 or epoch == 1:
            acc = correct / max(total, 1)
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={acc:.1%}")

    return model


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_model(model, factory, device, task_id_mode='normal',
                   n_eval=500, noise_sigma=0.0):
    model.eval()
    task_correct = {i: 0 for i in range(N_TASKS)}
    task_total = {i: 0 for i in range(N_TASKS)}

    for task_id in range(N_TASKS):
        inputs, targets = factory.generate_batch(task_id, n_eval, split='test')
        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        y = torch.tensor(targets, dtype=torch.float32, device=device)

        if task_id_mode == 'normal':
            tid = task_id
        elif task_id_mode == 'none':
            tid = None
        elif task_id_mode == 'wrong':
            tid = (task_id + random.randint(1, N_TASKS - 1)) % N_TASKS
        else:
            tid = task_id

        kwargs = {'task_id': tid, 'output_dim': 2, 'device': device}
        if noise_sigma > 0:
            kwargs['noise_sigma'] = noise_sigma

        logits = model(x, **kwargs)
        preds = logits.argmax(-1)
        true = y.argmax(-1)
        task_correct[task_id] = (preds == true).sum().item()
        task_total[task_id] = n_eval

    per_task = {TASK_NAMES[i]: task_correct[i] / max(task_total[i], 1)
                for i in range(N_TASKS)}
    overall = sum(task_correct.values()) / max(sum(task_total.values()), 1)
    return overall, per_task


# =============================================================================
# Sequence Length Generalization
# =============================================================================

@torch.no_grad()
def evaluate_length_transfer(model, factory, device, n_eval=500):
    """Test if rules learned at seq_len=6 transfer to other lengths."""
    model.eval()
    results = {}

    # Configurations: (task_id, test_seq_len, n_core_entities, description)
    configs = [
        (0, 2, 2, 'S/D len=2'),
        (0, 6, 2, 'S/D len=6'),
        (0, 10, 2, 'S/D len=10'),
        (1, 4, 4, 'RMTS len=4'),
        (1, 6, 4, 'RMTS len=6'),
        (1, 8, 4, 'RMTS len=8'),
        (2, 6, 6, 'IdRules len=6'),
        (2, 8, 6, 'IdRules len=8'),
        (3, 4, 4, 'Dist3 len=4'),
        (3, 6, 4, 'Dist3 len=6'),
        (3, 8, 4, 'Dist3 len=8'),
    ]

    entities = factory.test_entities

    for task_id, test_len, n_core, desc in configs:
        inputs_list = []
        targets_list = []

        for i in range(n_eval):
            is_match = (i < n_eval // 2)
            # Generate core entities using the task generator
            gen = [factory._gen_same_different, factory._gen_rmts,
                   factory._gen_identity_rules, factory._gen_dist3][task_id]
            seq = gen(entities, is_match)
            core = seq[:n_core]

            # Pad or truncate to test_len
            if test_len <= n_core:
                trial_seq = core[:test_len]
            else:
                n_pad = test_len - n_core
                distractors = factory._random_entities(entities, n_pad)
                trial_seq = core + distractors

            one_hots = np.stack([e.one_hot() for e in trial_seq])
            inputs_list.append(one_hots)
            targets_list.append(
                np.array([1., 0.] if is_match else [0., 1.], dtype=np.float32))

        inputs = np.array(inputs_list, dtype=np.float32)
        targets = np.array(targets_list, dtype=np.float32)
        perm = np.random.permutation(n_eval)
        inputs, targets = inputs[perm], targets[perm]

        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        y = torch.tensor(targets, dtype=torch.float32, device=device)

        logits = model(x, task_id=task_id, output_dim=2, device=device)
        acc = (logits.argmax(-1) == y.argmax(-1)).float().mean().item()
        results[desc] = acc

    return results


# =============================================================================
# Figures
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'figure.dpi': 200,
})
MODEL_COLORS = ['#5B9BD5', '#9B59B6', '#2ECC71']


def plot_results(all_eval, model_names, length_results=None):
    n_panels = 3 if length_results else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))

    # Panel A: Per-task accuracy with task_id
    ax = axes[0]
    x = np.arange(N_TASKS)
    bw = 0.7 / len(model_names)
    for i, name in enumerate(model_names):
        accs = [all_eval[name]['normal'][1].get(t, 0) for t in TASK_NAMES]
        offset = (i - (len(model_names) - 1) / 2) * bw
        bars = ax.bar(x + offset, accs, bw, label=name, color=MODEL_COLORS[i],
                      edgecolor='white', linewidth=0.8)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel('Test accuracy (novel entities)')
    ax.set_title('A.  Multi-task (with task_id)', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORTS)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.12, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel B: Robustness
    ax = axes[1]
    modes = ['normal', 'none', 'wrong']
    mode_labels = ['With ID', 'No ID', 'Wrong ID']
    mode_colors = ['#27AE60', '#F39C12', '#E74C3C']
    x = np.arange(len(model_names))
    bw = 0.22
    for m_idx, (mode, ml, mc) in enumerate(zip(modes, mode_labels, mode_colors)):
        accs = [all_eval[name][mode][0] for name in model_names]
        offset = (m_idx - 1) * bw
        bars = ax.bar(x + offset, accs, bw, label=ml, color=mc, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel('Overall accuracy')
    ax.set_title('B.  Task-ID robustness', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12, axis='y')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Panel C: Length generalization (GEE only)
    if length_results:
        ax = axes[2]
        tasks_to_plot = {'S/D': [], 'RMTS': [], 'IdRules': [], 'Dist3': []}
        for desc, acc in length_results.items():
            for task_short in tasks_to_plot:
                if desc.startswith(task_short):
                    length = int(desc.split('=')[1])
                    tasks_to_plot[task_short].append((length, acc))
        task_colors = ['#5B9BD5', '#E74C3C', '#F39C12', '#2ECC71']
        for (task_short, points), color in zip(tasks_to_plot.items(), task_colors):
            if points:
                points.sort()
                lengths, accs = zip(*points)
                ax.plot(lengths, accs, '-o', color=color, label=task_short,
                        markersize=5, linewidth=2)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.axvline(6, color='gray', ls='--', alpha=0.3, label='Training length')
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('Accuracy')
        ax.set_title('C.  Length generalization (GEE)', fontweight='bold', loc='left')
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.12)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.suptitle('GEE Unified Model — Fixed-Length Tasks',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_unified_results.png', bbox_inches='tight', facecolor='white')
    log("  Saved: fig_unified_results.png")
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
    log("  GEE Unified Experiment — Fixed-Length Tasks (seq_len=6)")
    log("=" * 70)
    log(f"  value_dim={cfg.value_dim}  key_dim={cfg.key_dim}  hidden={cfg.hidden_dim}")
    log(f"  Tasks: {TASK_NAMES} (all seq_len=6, binary output)")
    log(f"  Train entities: {len(factory.train_entities)}")
    log(f"  Test entities: {len(factory.test_entities)}")
    log()

    # ── Models ──
    model_configs = [
        ('ESBN',    {'use_h0': False, 'use_rcm': False, 'dropout_p': 0.0}),
        ('GEE',     {'use_h0': True,  'use_rcm': True,  'dropout_p': 0.3}),
        ('GEE-h0',  {'use_h0': True,  'use_rcm': False, 'dropout_p': 0.3}),
    ]

    all_eval = {}
    trained_models = {}

    for name, kwargs in model_configs:
        log(f"{'─' * 70}")
        log(f"  {name}")
        log(f"{'─' * 70}")

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = GEE(cfg, n_tasks=N_TASKS, **kwargs).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"  Params: {n_params:,}  h0={kwargs['use_h0']}  rcm={kwargs['use_rcm']}  "
            f"dropout={kwargs['dropout_p']}")

        t0 = time.time()
        train_model(model, factory, device, n_epochs=50, batches_per_task=100,
                    batch_size=64, lr=1e-4)
        log(f"  Training: {time.time()-t0:.0f}s")

        # Evaluate all conditions
        results = {}
        for mode, label in [('normal', 'With task_id'),
                             ('none', 'No task_id'),
                             ('wrong', 'Wrong task_id')]:
            overall, per_task = evaluate_model(model, factory, device,
                                               task_id_mode=mode)
            results[mode] = (overall, per_task)
            task_str = "  ".join(f"{TASK_SHORTS[i]}={per_task[t]:.0%}"
                                for i, t in enumerate(TASK_NAMES))
            log(f"  {label:<18} overall={overall:.1%}  {task_str}")

        all_eval[name] = results
        trained_models[name] = model
        log()

    # ── Sequence Length Generalization (GEE only) ──
    log(f"{'─' * 70}")
    log("  Sequence Length Generalization (GEE)")
    log(f"{'─' * 70}")
    length_results = evaluate_length_transfer(trained_models['GEE'], factory, device)
    for desc, acc in sorted(length_results.items()):
        log(f"  {desc:<20} {acc:.1%}")
    log()

    # ── Summary Tables ──
    log("=" * 70)
    log("  TABLE 1: Multi-task accuracy (with task_id)")
    log("=" * 70)
    header = f"  {'Task':<20}"
    for name, _ in model_configs:
        header += f" {name:>12}"
    log(header)
    log(f"  {'─' * (20 + 13 * len(model_configs))}")
    for task in TASK_NAMES:
        row = f"  {task:<20}"
        for name, _ in model_configs:
            acc = all_eval[name]['normal'][1].get(task, 0)
            row += f" {acc:>11.1%}"
        log(row)
    row = f"  {'OVERALL':<20}"
    for name, _ in model_configs:
        row += f" {all_eval[name]['normal'][0]:>11.1%}"
    log(row)
    log()

    log("=" * 70)
    log("  TABLE 2: Task-ID robustness")
    log("=" * 70)
    log(f"  {'Model':<12} {'Normal':>8} {'No ID':>8} {'Wrong':>8} {'Drop':>8}")
    log(f"  {'─' * 46}")
    for name, _ in model_configs:
        norm = all_eval[name]['normal'][0]
        noid = all_eval[name]['none'][0]
        wrong = all_eval[name]['wrong'][0]
        drop = norm - noid
        log(f"  {name:<12} {norm:>7.1%} {noid:>7.1%} {wrong:>7.1%} {drop:>+7.1%}")
    log("=" * 70)

    # ── Figure ──
    model_names = [name for name, _ in model_configs]
    plot_results(all_eval, model_names, length_results)

    log("\n  Done.")


if __name__ == '__main__':
    main()
