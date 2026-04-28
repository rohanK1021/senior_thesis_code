"""
GEE Three-Way Comparison on Webb-Faithful Tasks
=================================================
Definitive experiment: ESBN vs GEE-Direct vs GEE-Separated on Webb's actual
task structures (seq=2/6/9, mixed 2-way and 4-way output).

Models all share Webb-faithful ESBN core, differing only in context pathway.
GEE-Separated uses task_id dropout (p=0.5) for robust fallback.

Run: python -u gee_webb_comparison.py
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


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Task configuration
# =============================================================================

TASK_CONFIGS = [
    {'name': 'same_different',   'short': 'S/D',     'attr': 'same_different',          'seq_len': 2, 'output_dim': 2},
    {'name': 'webb_rmts',        'short': 'RMTS',    'attr': 'webb_rmts',               'seq_len': 6, 'output_dim': 2},
    {'name': 'webb_dist3',       'short': 'Dist3',   'attr': 'webb_distribution_of_3',  'seq_len': 9, 'output_dim': 4},
    {'name': 'webb_idrules',     'short': 'IdRules', 'attr': 'webb_identity_rules',     'seq_len': 9, 'output_dim': 4},
]
MAX_SEQ_LEN = 9
MAX_LABEL_DIM = 4


# =============================================================================
# Models — all share Webb-faithful ESBN core, dual output heads (2 + 4)
# =============================================================================

class WebbESBN(nn.Module):
    """Webb-faithful ESBN baseline. Ignores task_id."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.y_out_4 = nn.Linear(cfg.hidden_dim, 4)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq, task_id=None, output_dim=2, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)
        hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            z_t = torch.zeros(batch, 1, cfg.value_dim, device=device) if t == seq_len else z_seq[:, t, :].unsqueeze(1)
            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)

        y_out = self.y_out_4 if output_dim == 4 else self.y_out_2
        return y_out(gru_out).squeeze(1)


class WebbGEEDirect(nn.Module):
    """GEE-Direct: task_id in keys + RCM at output. Dual output heads.
    Now with task_id dropout (p=0.5) for fair comparison with GEE-Separated."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.y_out_4 = nn.Linear(cfg.hidden_dim, 4)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # GEE-Direct additions
        self.task_to_key = nn.Linear(cfg.n_tasks, cfg.key_dim, bias=False)
        rcm_input_dim = cfg.input_dim + cfg.n_tasks
        self.rcm = RCMContextModule(rcm_input_dim, cfg.output_dim, cfg.context_dim)
        self.ctx_to_out_2 = nn.Linear(cfg.context_dim, 2, bias=False)
        self.ctx_to_out_4 = nn.Linear(cfg.context_dim, 4, bias=False)
        self.task_id_dropout = 0.5

    def forward(self, x_seq, task_id=None, output_dim=2, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # Task_id dropout during training
        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, cfg.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        task_key_bias = self.task_to_key(task_onehot)

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

            rcm_in = torch.cat([x_raw, task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out)) + task_key_bias.unsqueeze(1)
            g = torch.sigmoid(self.g_out(gru_out))

            if output_dim == 4:
                y_pred_linear = self.y_out_4(gru_out).squeeze(1) + self.ctx_to_out_4(c_t)
            else:
                y_pred_linear = self.y_out_2(gru_out).squeeze(1) + self.ctx_to_out_2(c_t)
            y_prev = torch.softmax(y_pred_linear[:, :cfg.output_dim].detach(), dim=-1)

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)

        return y_pred_linear


class WebbGEESeparated(nn.Module):
    """GEE-Separated: context at h_0 only. Task_id dropout. Dual output heads."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.y_out_4 = nn.Linear(cfg.hidden_dim, 4)
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # GEE-Separated addition
        self.h0_proj = nn.Linear(cfg.n_tasks, cfg.hidden_dim)
        self.task_id_dropout = 0.5

    def forward(self, x_seq, task_id=None, output_dim=2, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # Task_id dropout during training
        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, cfg.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)

        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            z_t = torch.zeros(batch, 1, cfg.value_dim, device=device) if t == seq_len else z_seq[:, t, :].unsqueeze(1)
            gru_out, hidden = self.gru(key_r, hidden)
            key_w = F.relu(self.key_w_out(gru_out))
            g = torch.sigmoid(self.g_out(gru_out))

            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid((z_t * M_v).sum(dim=2) * self.confidence_gain + self.confidence_bias)
                key_r = g * (torch.cat([M_k, c_k.unsqueeze(2)], dim=2) * w_k.unsqueeze(2)).sum(1).unsqueeze(1)

            if t == 0:
                M_k = key_w; M_v = z_t
            else:
                M_k = torch.cat([M_k, key_w], dim=1); M_v = torch.cat([M_v, z_t], dim=1)

        y_out = self.y_out_4 if output_dim == 4 else self.y_out_2
        return y_out(gru_out).squeeze(1)


# =============================================================================
# Data preparation
# =============================================================================

def prepare_data(factory, n_per_task, split):
    all_inputs, all_labels, all_task_ids = [], [], []
    embed_dim = factory.embedder.embed_dim

    for tid, tc in enumerate(TASK_CONFIGS):
        task = getattr(factory, tc['attr'])
        inputs, labels, _ = task.generate(n_per_task, split=split)
        sl = tc['seq_len']
        od = tc['output_dim']
        for inp, lbl in zip(inputs, labels):
            padded = np.zeros((MAX_SEQ_LEN, embed_dim), dtype=np.float32)
            for t in range(sl):
                padded[t] = inp[t]
            all_inputs.append(padded)
            # Pad labels to MAX_LABEL_DIM
            padded_lbl = np.zeros(MAX_LABEL_DIM, dtype=np.float32)
            padded_lbl[:len(lbl)] = lbl
            all_labels.append(padded_lbl)
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


# =============================================================================
# Training + Evaluation with mixed output dims
# =============================================================================

def mt_forward_and_loss(model, x_batch, y_batch, task_ids, device):
    """Forward pass with mixed output dims. Returns total loss and per-sample correctness."""
    batch_size = x_batch.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    is_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
    n_groups = 0

    for tid in task_ids.unique():
        mask = task_ids == tid
        tid_int = tid.item()
        tc = TASK_CONFIGS[tid_int]
        sl = tc['seq_len']
        od = tc['output_dim']

        x_sub = x_batch[mask][:, :sl, :]
        y_sub = y_batch[mask][:, :od]

        logits = model(x_sub, task_id=tid_int, output_dim=od, device=device)
        total_loss = total_loss + F.cross_entropy(logits, y_sub) * mask.sum()
        is_correct[mask] = logits.argmax(-1) == y_sub.argmax(-1)
        n_groups += mask.sum()

    return total_loss / n_groups, is_correct


def train_model(model, train_data, device, lr=5e-4, n_epochs=100):
    inputs, labels, task_ids = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = MTDataset(inputs, labels, task_ids)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b, tid_b in loader:
            x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)
            loss, is_correct = mt_forward_and_loss(model, x_b, y_b, tid_b, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x_b.shape[0]
            correct += is_correct.sum().item()
            total += x_b.shape[0]
        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")


@torch.no_grad()
def evaluate_model(model, test_data, device, task_id_mode='normal'):
    inputs, labels, task_ids = test_data
    model.eval()
    ds = MTDataset(inputs, labels, task_ids)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    task_correct = {i: 0 for i in range(len(TASK_CONFIGS))}
    task_total = {i: 0 for i in range(len(TASK_CONFIGS))}

    for x_b, y_b, tid_b in loader:
        x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)

        for tid in tid_b.unique():
            mask = tid_b == tid
            tid_int = tid.item()
            tc = TASK_CONFIGS[tid_int]
            sl = tc['seq_len']
            od = tc['output_dim']
            x_sub = x_b[mask][:, :sl, :]
            y_sub = y_b[mask][:, :od]

            if task_id_mode == 'normal':
                logits = model(x_sub, task_id=tid_int, output_dim=od, device=device)
            elif task_id_mode == 'none':
                logits = model(x_sub, task_id=None, output_dim=od, device=device)
            elif task_id_mode == 'wrong':
                wrong = (tid_int + random.randint(1, 3)) % len(TASK_CONFIGS)
                logits = model(x_sub, task_id=wrong, output_dim=od, device=device)

            preds = logits.argmax(-1)
            true = y_sub.argmax(-1)
            task_correct[tid_int] += (preds == true).sum().item()
            task_total[tid_int] += mask.sum().item()

    per_task = {}
    for i, tc in enumerate(TASK_CONFIGS):
        per_task[tc['name']] = task_correct[i] / max(task_total[i], 1)
    overall = sum(task_correct.values()) / max(sum(task_total.values()), 1)
    return overall, per_task


# =============================================================================
# Professional thesis figures
# =============================================================================

MODEL_NAMES = ['ESBN', 'GEE-Direct', 'GEE-Separated']

def plot_results(all_results):
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 200,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    task_shorts = [tc['short'] for tc in TASK_CONFIGS]
    model_colors = ['#5B9BD5', '#9B59B6', '#2ECC71']
    n_models = len(all_results)

    # ── Panel A: Per-task accuracy ──
    ax = axes[0]
    x = np.arange(len(TASK_CONFIGS))
    bar_width = 0.7 / n_models
    for i, (label, results) in enumerate(all_results):
        accs = [results['normal'][1].get(tc['name'], 0) for tc in TASK_CONFIGS]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=label,
                      color=model_colors[i], edgecolor='white', linewidth=0.8)
        for bar, acc in zip(bars, accs):
            va = 'bottom'
            y_pos = bar.get_height() + 0.01
            if acc < 0.3:
                va = 'bottom'
                y_pos = bar.get_height() + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{acc:.0%}', ha='center', va=va, fontsize=8, fontweight='bold')

    ax.set_ylabel('Test accuracy (novel entities)')
    ax.set_title('A.  Multi-task performance', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(task_shorts)
    ax.set_ylim(0, 1.12)
    chance_2 = 0.5
    chance_4 = 0.25
    ax.axhline(chance_2, color='gray', ls=':', alpha=0.4, linewidth=0.8)
    ax.axhline(chance_4, color='gray', ls=':', alpha=0.4, linewidth=0.8)
    ax.text(3.5, chance_2 + 0.01, '50%', fontsize=7, color='gray', ha='right')
    ax.text(3.5, chance_4 + 0.01, '25%', fontsize=7, color='gray', ha='right')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: Task-ID robustness ──
    ax = axes[1]
    modes = ['normal', 'none', 'wrong']
    mode_labels = ['With task ID', 'No task ID', 'Wrong task ID']
    mode_colors = ['#27AE60', '#F39C12', '#E74C3C']
    x = np.arange(n_models)
    bar_width = 0.22

    for m_idx, (mode, m_label, color) in enumerate(zip(modes, mode_labels, mode_colors)):
        accs = [results[mode][0] for _, results in all_results]
        offset = (m_idx - 1) * bar_width
        bars = ax.bar(x + offset, accs, bar_width, label=m_label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.8)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Drop annotations
    for i, (label, results) in enumerate(all_results):
        norm = results['normal'][0]
        noid = results['none'][0]
        drop = norm - noid
        if abs(drop) > 0.005:
            mid_x = i - bar_width / 2
            ax.annotate('', xy=(mid_x, noid), xytext=(mid_x, norm),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
            ax.text(mid_x - 0.08, (norm + noid) / 2, f'{drop:+.0%}',
                    fontsize=8, color='#555', fontstyle='italic', ha='right', va='center')

    ax.set_ylabel('Overall test accuracy')
    ax.set_title('B.  Robustness to task-ID removal', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in all_results])
    ax.set_ylim(0, 1.12)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4, linewidth=0.8)
    ax.axhline(0.25, color='gray', ls=':', alpha=0.4, linewidth=0.8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('GEE Architecture Comparison on Webb-Faithful Tasks',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('fig_webb_three_way.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    log(f"\n  Saved: fig_webb_three_way.png")
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

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=seed)

    log("=" * 70)
    log("  GEE Three-Way Comparison — Webb-Faithful Tasks")
    log("=" * 70)
    log(f"  value_dim={cfg.value_dim}  key_dim={cfg.key_dim}  hidden={cfg.hidden_dim}")
    log(f"  {factory.pool.summary()}")
    log()
    for tc in TASK_CONFIGS:
        log(f"  {tc['short']:<8} seq={tc['seq_len']}  output_dim={tc['output_dim']}  ({tc['name']})")
    log()

    train_data = prepare_data(factory, n_per_task=2000, split='train')
    test_data = prepare_data(factory, n_per_task=500, split='test')
    log(f"  Train: {len(train_data[0])}  Test: {len(test_data[0])}")
    log()

    models_config = [
        ("ESBN",          lambda: WebbESBN(cfg)),
        ("GEE-Direct",    lambda: WebbGEEDirect(cfg)),
        ("GEE-Separated", lambda: WebbGEESeparated(cfg)),
    ]

    all_results = []

    for label, make_model in models_config:
        log(f"{'─' * 70}")
        log(f"  {label}")
        log(f"{'─' * 70}")

        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = make_model().to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Params: {n_params:,}")

        t0 = time.time()
        train_model(model, train_data, device, lr=cfg.lr, n_epochs=100)
        log(f"  Training: {time.time()-t0:.0f}s")

        results = {}
        for mode, mode_label in [('normal', 'With task_id'),
                                  ('none',   'No task_id'),
                                  ('wrong',  'Wrong task_id')]:
            overall, pt = evaluate_model(model, test_data, device, task_id_mode=mode)
            results[mode] = (overall, pt)
            task_str = "  ".join(f"{tc['short']}={pt[tc['name']]:.0%}" for tc in TASK_CONFIGS)
            log(f"  {mode_label:<18} overall={overall:.1%}  {task_str}")

        all_results.append((label, results))
        log()

    # Summary tables
    log("=" * 70)
    log("  TABLE 1: Multi-task accuracy (with task_id) — Webb tasks")
    log("=" * 70)
    header = f"  {'Task':<22}"
    for label, _ in all_results:
        header += f" {label:>15}"
    log(header)
    log(f"  {'─' * (22 + 16 * len(all_results))}")
    for tc in TASK_CONFIGS:
        row = f"  {tc['name']:<22}"
        for _, results in all_results:
            acc = results['normal'][1].get(tc['name'], 0)
            row += f" {acc:>14.1%}"
        log(row)
    row = f"  {'OVERALL':<22}"
    for _, results in all_results:
        row += f" {results['normal'][0]:>14.1%}"
    log(row)
    log()

    log("=" * 70)
    log("  TABLE 2: Task-ID removal robustness — Webb tasks")
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
