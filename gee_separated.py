"""
GEE Architecture Comparison — Three-Way Experiment
====================================================
Compares three architectures that differ ONLY in how context enters:

  ESBN          — No context. Webb-faithful baseline.
  GEE-Direct    — Task_id in write keys + RCM at output (PI's version).
  GEE-Separated — Context sets h_0 only. Pure ESBN during sequence.

All share the Webb-faithful ESBN core (nn.GRU, scalar gate, Webb confidence,
non-causal TCN, z_t queries M_v). GEE deviations are minimal and documented.

Three evaluation conditions per model:
  A. WITH task_id   — normal operation
  B. WITHOUT task_id — can the model reason without knowing the task?
  C. WRONG task_id  — does bad context hurt?

Run: python -u gee_separated.py
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

from dataset import DatasetFactory
from gee_model import GEEConfig, RCMContextModule
from esbn_pure import ContextNorm, AttributeAwareEncoder
from true_gee import TrueGEE


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Task setup (original tasks — all binary, shared output_dim=2)
# =============================================================================

TASK_NAMES = ['same_different', 'rmts', 'distribution_of_3', 'identity_rules']
TASK_SHORT = ['S/D', 'RMTS', 'Dist3', 'IdRules']
TASK_SEQ_LENS = [2, 4, 4, 6]
MAX_SEQ_LEN = max(TASK_SEQ_LENS)

MODEL_NAMES = ['ESBN', 'GEE-Direct', 'GEE-Separated']
MODEL_COLORS = ['#90CAF9', '#CE93D8', '#A5D6A7']


# =============================================================================
# ESBN baseline (Webb-faithful, ignores task_id)
# =============================================================================

class MultiTaskESBN(nn.Module):
    """Webb-faithful ESBN that accepts (and ignores) task_id for multi-task compat."""
    def __init__(self, cfg: GEEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq, task_id=None, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)

            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred_linear = self.y_out(gru_out).squeeze(1)

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias
                )
                key_r = g * (
                    torch.cat([M_k, c_k.unsqueeze(2)], dim=2)
                    * w_k.unsqueeze(2)
                ).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w
                M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1)
                M_v = torch.cat([M_v, z_t], dim=1)

        return y_pred_linear


# =============================================================================
# GEE-Separated Model
# =============================================================================

class SeparatedGEE(nn.Module):
    """
    GEE-Separated: context enters ONLY at h_0.

    Built on Webb-faithful ESBN core. Single deviation:
      h_0 = tanh(Linear(task_onehot))  — learned starting state per task

    During the sequence: pure ESBN. No key modulation, no output modulation,
    no RCM. Removing task_id reverts h_0 to a learned default (bias term).
    """
    def __init__(self, cfg: GEEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder + context normalization (Webb-faithful)
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)

        # GRU controller (Webb-faithful)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        # Webb-faithful confidence
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # === GEE-Separated addition (single deviation from ESBN) ===
        # Context → h_0: each task gets a learned starting state
        self.h0_proj = nn.Linear(cfg.n_tasks, cfg.hidden_dim)
        self.task_id_dropout = 0.5  # probability of dropping task_id during training

    def forward(self, x_seq, task_id=None, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        # Encode + context normalization (Webb-faithful)
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # h_0 from task context (GEE-Separated deviation)
        # Task_id dropout: randomly withhold task_id during training so the model
        # learns to function both with and without context. Forces graceful
        # degradation to ESBN-level when task_id is unavailable at test time.
        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, cfg.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)  # (1, batch, hidden_dim)

        # Everything below is pure Webb-faithful ESBN
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)

            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))
            y_pred_linear = self.y_out(gru_out).squeeze(1)

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias
                )
                key_r = g * (
                    torch.cat([M_k, c_k.unsqueeze(2)], dim=2)
                    * w_k.unsqueeze(2)
                ).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w
                M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1)
                M_v = torch.cat([M_v, z_t], dim=1)

        return y_pred_linear


# =============================================================================
# Data helpers
# =============================================================================

def prepare_data(factory, n_per_task, split):
    tasks = [factory.same_different, factory.rmts,
             factory.distribution_of_3, factory.identity_rules]
    all_inputs, all_labels, all_task_ids = [], [], []
    embed_dim = factory.embedder.embed_dim

    for tid, task in enumerate(tasks):
        inputs, labels, _ = task.generate(n_per_task, split=split)
        sl = task.SEQ_LEN
        for inp, lbl in zip(inputs, labels):
            padded = np.zeros((MAX_SEQ_LEN, embed_dim), dtype=np.float32)
            for t in range(sl):
                padded[t] = inp[t]
            all_inputs.append(padded)
            all_labels.append(lbl)
            all_task_ids.append(tid)

    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)
    all_task_ids = np.array(all_task_ids, dtype=np.int64)
    perm = np.random.permutation(len(all_inputs))
    return all_inputs[perm], all_labels[perm], all_task_ids[perm]


class MTDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, task_ids):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.task_ids = torch.tensor(task_ids, dtype=torch.int64)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.task_ids[idx]


def mt_forward(model, x_batch, task_ids, device):
    batch_size = x_batch.shape[0]
    all_logits = torch.zeros(batch_size, 2, device=device)
    for tid in task_ids.unique():
        mask = task_ids == tid
        sl = TASK_SEQ_LENS[tid.item()]
        x_sub = x_batch[mask][:, :sl, :]
        all_logits[mask] = model(x_sub, task_id=tid.item(), device=device)
    return all_logits


# =============================================================================
# Training + Evaluation
# =============================================================================

def train_model(model, train_data, device, lr=5e-4, n_epochs=100, label=""):
    inputs, labels, task_ids = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = MTDataset(inputs, labels, task_ids)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b, tid_b in loader:
            x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)
            logits = mt_forward(model, x_b, tid_b, device)
            loss = F.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x_b.shape[0]
            correct += (logits.argmax(-1) == y_b.argmax(-1)).sum().item()
            total += x_b.shape[0]
        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")


@torch.no_grad()
def evaluate_model(model, test_data, device, task_id_mode='normal'):
    inputs, labels, task_ids = test_data
    model.eval()
    ds = MTDataset(inputs, labels, task_ids)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    task_correct = {i: 0 for i in range(4)}
    task_total = {i: 0 for i in range(4)}

    for x_b, y_b, tid_b in loader:
        x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)
        batch_size = x_b.shape[0]
        all_logits = torch.zeros(batch_size, 2, device=device)

        for tid in tid_b.unique():
            mask = tid_b == tid
            tid_int = tid.item()
            sl = TASK_SEQ_LENS[tid_int]
            x_sub = x_b[mask][:, :sl, :]

            if task_id_mode == 'normal':
                logits = model(x_sub, task_id=tid_int, device=device)
            elif task_id_mode == 'none':
                logits = model(x_sub, task_id=None, device=device)
            elif task_id_mode == 'wrong':
                wrong = (tid_int + random.randint(1, 3)) % 4
                logits = model(x_sub, task_id=wrong, device=device)
            all_logits[mask] = logits

        preds = all_logits.argmax(-1)
        true = y_b.argmax(-1)
        is_correct = (preds == true)
        for tid in range(4):
            mask = tid_b == tid
            task_correct[tid] += is_correct[mask].sum().item()
            task_total[tid] += mask.sum().item()

    per_task = {TASK_NAMES[i]: task_correct[i] / max(task_total[i], 1) for i in range(4)}
    overall = sum(task_correct.values()) / max(sum(task_total.values()), 1)
    return overall, per_task


# =============================================================================
# Plotting — thesis-quality figures
# =============================================================================

def plot_results(all_results, save_path='fig_three_way_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    n_models = len(all_results)

    # Panel A: Per-task accuracy
    ax = axes[0]
    x = np.arange(len(TASK_NAMES))
    bar_width = 0.8 / n_models
    for i, (label, results) in enumerate(all_results):
        accs = [results['normal'][1].get(t, 0) for t in TASK_NAMES]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=label,
                      color=MODEL_COLORS[i], edgecolor='white', linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel('Test accuracy (novel entities)', fontsize=11)
    ax.set_title('A. Multi-Task Performance (with task_id)', fontsize=12,
                 fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel B: Task-ID robustness
    ax = axes[1]
    modes = ['normal', 'none', 'wrong']
    mode_labels = ['With ID', 'No ID', 'Wrong ID']
    mode_colors = ['#4CAF50', '#FF9800', '#F44336']
    x = np.arange(n_models)
    bar_width = 0.25
    for m_idx, (mode, m_label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        accs = [results[mode][0] for _, results in all_results]
        offset = (m_idx - 1) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=m_label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel('Overall test accuracy', fontsize=11)
    ax.set_title('B. Robustness to Task-ID Removal', fontsize=12,
                 fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in all_results], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (label, results) in enumerate(all_results):
        norm = results['normal'][0]
        noid = results['none'][0]
        drop = norm - noid
        if abs(drop) > 0.005:
            ax.annotate(f'{drop:+.0%}', xy=(i, min(norm, noid) - 0.02),
                        fontsize=7, ha='center', color='#555', fontstyle='italic')

    plt.suptitle('GEE Architecture Comparison: Where Should Context Enter?',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    log(f"\n  Saved: {save_path}")
    plt.close()


# =============================================================================
# Main — Three-way comparison
# =============================================================================

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cpu')
    cfg = GEEConfig()

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=seed)

    log("=" * 70)
    log("  GEE Architecture Comparison — Three-Way Experiment")
    log("=" * 70)
    log(f"  Device:      {device}")
    log(f"  value_dim:   {cfg.value_dim}  key_dim: {cfg.key_dim}  hidden: {cfg.hidden_dim}")
    log(f"  context_dim: {cfg.context_dim}")
    log(f"  {factory.pool.summary()}")
    log()
    log("  Models:")
    log("    ESBN          — No context (Webb-faithful baseline)")
    log("    GEE-Direct    — Task_id in keys + RCM at output (PI's version)")
    log("    GEE-Separated — Context → h_0 only, pure ESBN loop")
    log()

    train_data = prepare_data(factory, n_per_task=2000, split='train')
    test_data = prepare_data(factory, n_per_task=500, split='test')
    log(f"  Train: {len(train_data[0])}  Test: {len(test_data[0])}")
    log()

    models_config = [
        ("ESBN",          "esbn"),
        ("GEE-Direct",    "direct"),
        ("GEE-Separated", "separated"),
    ]

    all_results = []

    for label, model_type in models_config:
        log(f"{'─' * 70}")
        log(f"  {label}")
        log(f"{'─' * 70}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if model_type == "esbn":
            model = MultiTaskESBN(cfg).to(device)
        elif model_type == "direct":
            model = TrueGEE(cfg).to(device)
        elif model_type == "separated":
            model = SeparatedGEE(cfg).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Params: {n_params:,}")

        t0 = time.time()
        train_model(model, train_data, device, lr=cfg.lr, n_epochs=100, label=label)
        log(f"  Training: {time.time()-t0:.0f}s")

        results = {}
        for mode, mode_label in [('normal', 'With task_id'),
                                  ('none',   'No task_id'),
                                  ('wrong',  'Wrong task_id')]:
            overall, pt = evaluate_model(model, test_data, device, task_id_mode=mode)
            results[mode] = (overall, pt)
            task_str = "  ".join(f"{TASK_SHORT[i]}={pt[t]:.0%}"
                                for i, t in enumerate(TASK_NAMES))
            log(f"  {mode_label:<18} overall={overall:.1%}  {task_str}")

        all_results.append((label, results))
        log()

    # Summary
    log("=" * 70)
    log("  TABLE 1: Multi-task test accuracy (with task_id)")
    log("=" * 70)
    header = f"  {'Task':<22}"
    for label, _ in all_results:
        header += f" {label:>15}"
    log(header)
    log(f"  {'─' * (22 + 16 * len(all_results))}")
    for task in TASK_NAMES:
        row = f"  {task:<22}"
        for _, results in all_results:
            acc = results['normal'][1].get(task, 0)
            row += f" {acc:>14.1%}"
        log(row)
    row = f"  {'OVERALL':<22}"
    for _, results in all_results:
        row += f" {results['normal'][0]:>14.1%}"
    log(row)
    log()

    log("=" * 70)
    log("  TABLE 2: Task-ID removal robustness")
    log("=" * 70)
    log(f"  {'Model':<18} {'Normal':>8} {'No ID':>8} {'Wrong':>8} {'Drop':>8}")
    log(f"  {'─' * 54}")
    for label, results in all_results:
        norm = results['normal'][0]
        noid = results['none'][0]
        wrong = results['wrong'][0]
        drop = norm - noid
        log(f"  {label:<18} {norm:>7.1%} {noid:>7.1%} {wrong:>7.1%} {drop:>+7.1%}")
    log("=" * 70)

    plot_results(all_results)
    log("\n  Done.")


if __name__ == '__main__':
    main()
