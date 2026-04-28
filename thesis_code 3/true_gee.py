"""
True GEE — PI's Version
=========================
Clean separation of concerns:
  - Task identity → write keys (direct linear injection, no RCM)
  - Situational context → output (RCM builds C_t from sequence)
  - Abstract structure → GRU (indirection preserved)

Architecture:
  K_w = K_base + W_task @ task_id          (task separates key subspaces)
  logits = Linear(cat(h_t, z_ret) + W_out @ C_t)  (context informs output)

  GRU never sees task_id or C_t — its keys are biased by a hard discrete
  signal (task_id), not a drifting recurrent state. This is a stronger
  inductive bias: the GRU learns truly task-independent structural rules,
  separated only by task identity at the key injection point.

Three-way comparison:
  1. Pure ESBN     — no context, no task_id in keys
  2. True GEE      — task_id in keys, C_t in output (PI's version)
  3. Original GEE  — RCM C_t in keys AND output (original version)

All use: attribute-aware encoder, TCN, Webb's memory mechanism,
         extra timestep, gating, learnable confidence.

Run: python -u true_gee.py
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
from gee_model import (GEEConfig, GEEModel, AttributeAwareEncoder, TCN,
                        RCMContextModule)


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Task setup
# =============================================================================

TASK_NAMES = ['same_different', 'rmts', 'distribution_of_3', 'identity_rules']
TASK_SHORT = ['S/D', 'RMTS', 'Dist3', 'IdRules']
TASK_SEQ_LENS = [2, 4, 4, 6]
MAX_SEQ_LEN = max(TASK_SEQ_LENS)


# =============================================================================
# True GEE Model (PI's version)
# =============================================================================

class TrueGEE(nn.Module):
    """
    GEE-Direct: PI's version — task identity in keys, RCM at output only.

    Built on Webb-faithful ESBN core (nn.GRU, scalar gate, Webb confidence).
    Two documented deviations from ESBN:
      1. Task_id bias on write keys: K_w = K_base + W_task @ task_onehot
      2. RCM context bias on output: logits = y_out(h) + ctx_to_out(C_t)

    The GRU's retrieval pathway is clean — RCM never touches keys or memory.
    """
    def __init__(self, cfg: GEEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder + context normalization (Webb-faithful)
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = TCN(cfg.value_dim)

        # GRU controller (Webb-faithful: nn.GRU, batch_first)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)  # scalar gate (Webb)
        self.y_out = nn.Linear(cfg.hidden_dim, cfg.output_dim)

        # Webb-faithful confidence
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # === GEE-Direct additions (deviations from ESBN) ===
        # 1. Task_id → write key bias
        self.task_to_key = nn.Linear(cfg.n_tasks, cfg.key_dim, bias=False)

        # 2. RCM for output context (receives raw input + task_id + prev prediction)
        rcm_input_dim = cfg.input_dim + cfg.n_tasks
        self.rcm = RCMContextModule(rcm_input_dim, cfg.output_dim, cfg.context_dim)
        self.ctx_to_out = nn.Linear(cfg.context_dim, cfg.output_dim, bias=False)

    def forward(self, x_seq, task_id=None, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        # Encode + context normalization (Webb-faithful)
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # Task one-hot + pre-compute key bias
        task_onehot = torch.zeros(batch, cfg.n_tasks, device=device)
        if task_id is not None:
            task_onehot[:, task_id] = 1.0
        task_key_bias = self.task_to_key(task_onehot)  # (batch, key_dim)

        # Initialize states
        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, cfg.output_dim, device=device)

        for t in range(seq_len + 1):
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
                x_raw = torch.zeros(batch, cfg.input_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)
                x_raw = x_seq[:, t, :]

            # RCM: update context (for output only — never touches keys/memory)
            rcm_in = torch.cat([x_raw, task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

            # GRU controller (Webb-faithful)
            gru_out, hidden = self.gru(key_r, hidden)

            # Write key with task_id bias (GEE-Direct deviation #1)
            key_w = F.relu(self.key_w_out(gru_out)) + task_key_bias.unsqueeze(1)

            # Scalar gate (Webb)
            g = torch.sigmoid(self.g_out(gru_out))

            # Output with RCM context bias (GEE-Direct deviation #2)
            y_pred_linear = self.y_out(gru_out).squeeze(1) + self.ctx_to_out(c_t)
            y_prev = torch.softmax(y_pred_linear.detach(), dim=-1)

            # Read from memory (Webb-faithful)
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

            # Write to memory (Webb-faithful)
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
    """
    task_id_mode: 'normal' (true task_id), 'none' (zeros), 'wrong' (random other)
    """
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
    log("  True GEE — Three-Way Comparison")
    log("=" * 70)
    log(f"  {factory.pool.summary()}")
    log()

    # Data
    train_data = prepare_data(factory, n_per_task=2000, split='train')
    test_data = prepare_data(factory, n_per_task=500, split='test')
    log(f"  Train: {len(train_data[0])}  Test: {len(test_data[0])}")
    log()

    # Three models
    models_config = [
        ("Pure ESBN", "esbn"),
        ("True GEE (PI)", "true_gee"),
        ("Original GEE", "original_gee"),
    ]

    all_results = []

    for label, model_type in models_config:
        log(f"{'─'*70}")
        log(f"  {label}")
        log(f"{'─'*70}")

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        if model_type == "esbn":
            cfg.alpha_key = 0.0
            cfg.alpha_out = 0.0
            model = GEEModel(cfg).to(device)
        elif model_type == "true_gee":
            model = TrueGEE(cfg).to(device)
        elif model_type == "original_gee":
            cfg.alpha_key = 0.5
            cfg.alpha_out = 0.5
            model = GEEModel(cfg).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Params: {n_params:,}")

        t0 = time.time()
        train_model(model, train_data, device, lr=cfg.lr, n_epochs=100, label=label)
        log(f"  Training: {time.time()-t0:.0f}s")

        # Evaluate under 3 conditions
        results = {}
        for mode, mode_label in [('normal', 'With task_id'),
                                   ('none', 'No task_id'),
                                   ('wrong', 'Wrong task_id')]:
            overall, pt = evaluate_model(model, test_data, device, task_id_mode=mode)
            results[mode] = (overall, pt)
            task_str = " ".join(f"{TASK_SHORT[i]}={pt[t]:.0%}" for i, t in enumerate(TASK_NAMES))
            log(f"  {mode_label:<18} overall={overall:.1%}  {task_str}")

        all_results.append((label, results))
        log()

    # ══════════════════════════════════════════════════════════════════
    # Summary Tables
    # ══════════════════════════════════════════════════════════════════

    log("=" * 70)
    log("  RESULTS: Three-Way Comparison")
    log("=" * 70)
    log()

    # Table 1: Multi-task accuracy (with task_id)
    log("  Table 1: Multi-task test accuracy (with task_id)")
    log(f"  {'Task':<22}", end="")
    for label, _ in all_results:
        log(f" {label:>16}", end="")
    log()
    log(f"  {'─'*70}")
    for task in TASK_NAMES:
        log(f"  {task:<22}", end="")
        for _, results in all_results:
            acc = results['normal'][1].get(task, 0)
            log(f" {acc:>15.1%}", end="")
        log()
    log(f"  {'OVERALL':<22}", end="")
    for _, results in all_results:
        log(f" {results['normal'][0]:>15.1%}", end="")
    log()
    log()

    # Table 2: Task-ID removal
    log("  Table 2: Effect of removing task_id")
    log(f"  {'Model':<20} {'Normal':>8} {'No ID':>8} {'Wrong':>8} {'Drop':>8}")
    log(f"  {'─'*56}")
    for label, results in all_results:
        norm = results['normal'][0]
        noid = results['none'][0]
        wrong = results['wrong'][0]
        drop = norm - noid
        log(f"  {label:<20} {norm:>7.1%} {noid:>7.1%} {wrong:>7.1%} {drop:>+7.1%}")

    log()
    log("  Interpretation:")
    log("    True GEE uses task_id for keys (direct) + C_t for output (learned)")
    log("    Original GEE uses C_t for both keys and output (learned)")
    log("    If True GEE >= Original GEE → PI's design is cleaner AND at least as good")
    log("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # Plot
    # ══════════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fig 1: Multi-task accuracy bars
    ax = axes[0]
    x = np.arange(len(TASK_NAMES))
    bar_width = 0.25
    colors = ['#90CAF9', '#FFE082', '#CE93D8']
    for i, (label, results) in enumerate(all_results):
        accs = [results['normal'][1].get(t, 0) for t in TASK_NAMES]
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=label, color=colors[i])
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel('Test accuracy (novel entities)')
    ax.set_title('Multi-Task Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(TASK_SHORT)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Fig 2: Task-ID removal
    ax = axes[1]
    modes = ['normal', 'none', 'wrong']
    mode_labels = ['With ID', 'No ID', 'Wrong ID']
    mode_colors = ['#4CAF50', '#FF9800', '#F44336']
    x = np.arange(len(all_results))
    for m_idx, (mode, m_label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        accs = [results[mode][0] for _, results in all_results]
        offset = (m_idx - 1) * 0.25
        ax.bar(x + offset, accs, 0.25, label=m_label, color=color, alpha=0.8)
    ax.set_ylabel('Overall test accuracy')
    ax.set_title('Effect of Removing Task ID')
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in all_results], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('True GEE vs Original GEE vs Pure ESBN', fontsize=13)
    plt.tight_layout()
    plt.savefig('fig_true_gee_comparison.png', dpi=150, bbox_inches='tight')
    log(f"\n  Saved: fig_true_gee_comparison.png")


if __name__ == '__main__':
    main()
