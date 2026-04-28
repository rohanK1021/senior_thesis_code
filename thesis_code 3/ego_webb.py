"""
EGO-Style Baseline — Tested on Webb Tasks, WCST, and Unified Tasks
====================================================================
Tests whether EGO's approach (GRU + context, NO indirection) can solve
abstract relational reasoning.

Key difference from ESBN/GEE: The GRU processes encoded input z_t directly.
There is no key-value external memory and no indirection.
ESBN's indirection hypothesis: GRU never seeing raw input is essential for
learning abstract relations over novel entities.

This baseline has EVERYTHING except indirection:
  - Same encoder (AttributeAwareEncoder)
  - Same ContextNorm
  - RCM for task context
  - h_0 from task_id (like GEE-Separated)
  - Task_id dropout (p=0.5)
  - Dual output heads

If it fails on Webb tasks but works on WCST, that shows indirection matters
specifically for abstract relational reasoning (not for rule-based switching).

Run: python -u ego_webb.py
"""

import sys
import os
import json
import time
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import DatasetFactory, EntityPool, ALL_ENTITIES
from gee_model import GEEConfig, RCMContextModule
from esbn_pure import ContextNorm, AttributeAwareEncoder
from gee_webb_comparison import (
    TASK_CONFIGS, MAX_SEQ_LEN, MAX_LABEL_DIM,
    prepare_data, MTDataset,
)
from gee_wcst import (
    N_RULES, RULE_ATTR_IDX, WCST_RULES,
    generate_wcst_data, run_evaluation,
    CRITERION_TRIALS, MAX_COMPLETIONS, MAX_TRIALS_PER_SESSION,
)


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


SEED = 42


# =============================================================================
# EGO Baseline Model — Webb Tasks (mixed seq_len / output_dim)
# =============================================================================

class EGOBaseline(nn.Module):
    """
    EGO-style baseline: GRU processes z_t directly (no indirection).

    Contrast with ESBN where GRU input = retrieved abstract key (key_r).
    Here GRU input = z_t (the encoded stimulus itself).

    Everything else matches GEE-Separated as closely as possible:
      - Same encoder + ContextNorm
      - h_0 from task_id
      - RCM context bias at output
      - Task_id dropout p=0.5
      - Dual output heads (2-way and 4-way)
    """

    def __init__(self, cfg: GEEConfig, n_tasks: int = 4):
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.task_id_dropout = 0.5

        # Encoder + ContextNorm — same as ESBN
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)

        # GRU: input is z_t (value_dim=128), NOT key_r (key_dim+1=257)
        # This is the fundamental difference from ESBN/GEE
        self.gru = nn.GRU(cfg.value_dim, cfg.hidden_dim, batch_first=True)

        # Dual output heads
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.y_out_4 = nn.Linear(cfg.hidden_dim, 4)

        # h_0 from task_id (like GEE-Separated)
        self.h0_proj = nn.Linear(n_tasks, cfg.hidden_dim)

        # RCM context module
        rcm_input_dim = cfg.input_dim + n_tasks
        self.rcm = RCMContextModule(rcm_input_dim, cfg.output_dim, cfg.context_dim)

        # Context → output bias (one per output head)
        self.ctx_to_out_2 = nn.Linear(cfg.context_dim, 2, bias=False)
        self.ctx_to_out_4 = nn.Linear(cfg.context_dim, 4, bias=False)

    def forward(self, x_seq, task_id=None, output_dim=2, device=None):
        """
        x_seq:    (batch, seq_len, input_dim)
        task_id:  int or None
        output_dim: 2 or 4
        Returns: logits (batch, output_dim)
        """
        if device is None:
            device = x_seq.device
        batch, seq_len, input_dim = x_seq.shape
        cfg = self.cfg

        # Encode all timesteps + ContextNorm
        z_seq = self.encoder(
            x_seq.reshape(-1, input_dim)
        ).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # Task_id dropout during training
        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, self.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0

        # h_0 from task_id (context sets starting state)
        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)  # (1, batch, hidden_dim)

        # Process sequence directly with GRU — NO indirection
        # GRU receives z_t at each step; no external memory involved
        gru_out, _ = self.gru(z_seq, hidden)  # (batch, seq_len, hidden_dim)

        # RCM: build context C_t step by step from raw input + task_id
        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, cfg.output_dim, device=device)

        for t in range(seq_len):
            rcm_in = torch.cat([x_seq[:, t, :], task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

        # Output: last GRU hidden state + RCM context bias
        y_out = self.y_out_4 if output_dim == 4 else self.y_out_2
        logits = y_out(gru_out[:, -1, :])  # last timestep

        ctx_out = self.ctx_to_out_4 if output_dim == 4 else self.ctx_to_out_2
        logits = logits + ctx_out(c_t)

        return logits


# =============================================================================
# EGO Baseline for WCST (seq_len=2, output_dim=2 always)
# =============================================================================

class EGOBaseline_WCST(nn.Module):
    """
    EGO-style baseline adapted for WCST.
    Identical architecture but uses N_RULES (4) instead of n_tasks.
    Single output head (2-way binary).
    """

    def __init__(self, cfg: GEEConfig):
        super().__init__()
        self.cfg = cfg
        self.task_id_dropout = 0.5

        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)

        # GRU: z_t directly, no memory
        self.gru = nn.GRU(cfg.value_dim, cfg.hidden_dim, batch_first=True)

        # Single output head (WCST is always binary)
        self.y_out = nn.Linear(cfg.hidden_dim, 2)

        # h_0 from task_id
        self.h0_proj = nn.Linear(N_RULES, cfg.hidden_dim)

        # RCM
        rcm_input_dim = cfg.input_dim + N_RULES
        self.rcm = RCMContextModule(rcm_input_dim, 2, cfg.context_dim)
        self.ctx_to_out = nn.Linear(cfg.context_dim, 2, bias=False)

    def forward(self, x_seq, task_id=None, device=None, noise_sigma=0.0):
        if device is None:
            device = x_seq.device
        batch, seq_len, input_dim = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(
            x_seq.reshape(-1, input_dim)
        ).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, N_RULES, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        if noise_sigma > 0:
            task_onehot = task_onehot + torch.randn_like(task_onehot) * noise_sigma

        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)

        gru_out, _ = self.gru(z_seq, hidden)

        # RCM
        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, 2, device=device)
        for t in range(seq_len):
            rcm_in = torch.cat([x_seq[:, t, :], task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

        logits = self.y_out(gru_out[:, -1, :]) + self.ctx_to_out(c_t)
        return logits


# =============================================================================
# EGO Baseline for Unified Tasks (fixed seq_len=6, all binary output)
# =============================================================================

class EGOBaseline_Unified(nn.Module):
    """EGO-style baseline for unified tasks (all seq_len=6, output_dim=2)."""

    def __init__(self, cfg: GEEConfig, n_tasks: int = 4):
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.task_id_dropout = 0.5

        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)

        self.gru = nn.GRU(cfg.value_dim, cfg.hidden_dim, batch_first=True)
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.h0_proj = nn.Linear(n_tasks, cfg.hidden_dim)

        rcm_input_dim = cfg.input_dim + n_tasks
        self.rcm = RCMContextModule(rcm_input_dim, cfg.output_dim, cfg.context_dim)
        self.ctx_to_out_2 = nn.Linear(cfg.context_dim, 2, bias=False)

    def forward(self, x_seq, task_id=None, output_dim=2, device=None):
        if device is None:
            device = x_seq.device
        batch, seq_len, input_dim = x_seq.shape
        cfg = self.cfg

        z_seq = self.encoder(
            x_seq.reshape(-1, input_dim)
        ).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.task_id_dropout:
            effective_task_id = None

        task_onehot = torch.zeros(batch, self.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0

        hidden = torch.tanh(self.h0_proj(task_onehot)).unsqueeze(0)
        gru_out, _ = self.gru(z_seq, hidden)

        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, cfg.output_dim, device=device)
        for t in range(seq_len):
            rcm_in = torch.cat([x_seq[:, t, :], task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

        logits = self.y_out_2(gru_out[:, -1, :]) + self.ctx_to_out_2(c_t)
        return logits


# =============================================================================
# Webb Task Training Infrastructure (mirrors gee_webb_comparison.py)
# =============================================================================

def mt_forward_and_loss(model, x_batch, y_batch, task_ids, device):
    """Forward pass with mixed output dims. Returns total loss and per-sample correctness."""
    batch_size = x_batch.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    is_correct = torch.zeros(batch_size, dtype=torch.bool, device=device)
    n_groups = torch.tensor(0, dtype=torch.long, device=device)

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
        n_groups = n_groups + mask.sum()

    return total_loss / n_groups.clamp(min=1), is_correct


def train_webb(model, train_data, device, lr=5e-4, n_epochs=100):
    """Train on Webb tasks (multi-task, mixed seq_len/output_dim)."""
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
def evaluate_webb(model, test_data, device, task_id_mode='normal'):
    """Evaluate on Webb tasks."""
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
# Unified Task Training Infrastructure
# =============================================================================

def train_unified(model, factory, device, n_epochs=50, batches_per_task=100,
                  batch_size=64, lr=5e-4):
    """Train on unified tasks (all seq_len=6, all binary output)."""
    from unified_tasks import N_TASKS
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

        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")


@torch.no_grad()
def evaluate_unified(model, factory, device, task_id_mode='normal', n_eval=500):
    """Evaluate on unified tasks."""
    from unified_tasks import N_TASKS, TASK_NAMES
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

        logits = model(x, task_id=tid, output_dim=2, device=device)
        preds = logits.argmax(-1)
        true = y.argmax(-1)
        task_correct[task_id] = (preds == true).sum().item()
        task_total[task_id] = n_eval

    per_task = {TASK_NAMES[i]: task_correct[i] / max(task_total[i], 1)
                for i in range(N_TASKS)}
    overall = sum(task_correct.values()) / max(sum(task_total.values()), 1)
    return overall, per_task


# =============================================================================
# Figure
# =============================================================================

def plot_results(webb_results, wcst_results, unified_results, out_path='fig_ego_baseline.png'):
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 200,
    })

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── Panel A: Webb task accuracy (with/without task_id) ──
    ax = axes[0]
    model_labels = ['ESBN', 'GEE-Direct', 'GEE-Sep.', 'EGO-Base']
    colors_with = ['#5B9BD5', '#9B59B6', '#2ECC71', '#E67E22']
    colors_none = ['#2E86C1', '#6C3483', '#1D8348', '#A04000']
    ref_with  = [0.984, 0.975, 0.981, webb_results.get('normal', 0)]
    ref_none  = [0.984, 0.610, 0.982, webb_results.get('none', 0)]

    x = np.arange(len(model_labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, ref_with, w, label='With task ID', color=colors_with, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w / 2, ref_none, w, label='No task ID',   color=colors_none, alpha=0.85, edgecolor='white')
    for b, v in zip(bars1, ref_with):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{v:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for b, v in zip(bars2, ref_none):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{v:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Overall accuracy (novel entities)')
    ax.set_title('A.  Webb Tasks — Multi-task', fontweight='bold', loc='left')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4, lw=0.8)
    ax.axhline(0.25, color='gray', ls=':', alpha=0.4, lw=0.8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: WCST accuracy ──
    ax = axes[1]
    wcst_model_labels = ['ESBN', 'GEE-Direct', 'GEE-Sep.', 'EGO-Base']
    wcst_ref_acc = [0.729, 0.949, 0.526, wcst_results.get('accuracy_mean', 0)]
    wcst_ref_persev = [24.3, 1.1, 41.5, wcst_results.get('persev_errors_mean', 0)]

    bar_colors = ['#5B9BD5', '#9B59B6', '#2ECC71', '#E67E22']
    xw = np.arange(len(wcst_model_labels))
    bars = ax.bar(xw, wcst_ref_acc, 0.6, color=bar_colors, alpha=0.85, edgecolor='white')
    for b, v in zip(bars, wcst_ref_acc):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{v:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = ax.twinx()
    ax2.plot(xw, wcst_ref_persev, 'D--', color='#C0392B', markersize=7, linewidth=1.5,
             label='Persev. errors')
    ax2.set_ylabel('Perseverative errors (mean)', color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    ax2.set_ylim(0, max(wcst_ref_persev) * 1.4 + 1)

    ax.set_xticks(xw)
    ax.set_xticklabels(wcst_model_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('WCST accuracy')
    ax.set_title('B.  WCST — Rule Switching', fontweight='bold', loc='left')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4, lw=0.8)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines['top'].set_visible(False)

    lines, lbls = ax2.get_legend_handles_labels()
    ax.legend(lines, lbls, loc='upper left', framealpha=0.9)

    # ── Panel C: Unified tasks ──
    ax = axes[2]
    uni_model_labels = ['ESBN', 'GEE', 'EGO-Base']
    uni_ref_with = [0.671, 0.912, unified_results.get('normal', 0)]
    uni_ref_none = [0.665, 0.627, unified_results.get('none', 0)]

    xu = np.arange(len(uni_model_labels))
    wu = 0.35
    uni_colors_with = ['#5B9BD5', '#9B59B6', '#E67E22']
    uni_colors_none = ['#2E86C1', '#6C3483', '#A04000']
    bars1 = ax.bar(xu - wu / 2, uni_ref_with, wu, label='With task ID',
                   color=uni_colors_with, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(xu + wu / 2, uni_ref_none, wu, label='No task ID',
                   color=uni_colors_none, alpha=0.85, edgecolor='white')
    for b, v in zip(bars1, uni_ref_with):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{v:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for b, v in zip(bars2, uni_ref_none):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'{v:.0%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(xu)
    ax.set_xticklabels(uni_model_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Overall accuracy (novel entities)')
    ax.set_title('C.  Unified Tasks (fixed seq=6)', fontweight='bold', loc='left')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4, lw=0.8)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.suptitle('EGO-Style Baseline: GRU + Context, No Indirection',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    log(f"\n  Saved: {out_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = GEEConfig()

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'ego_baseline_results.json')
    fig_path = os.path.join(os.path.dirname(__file__), 'fig_ego_baseline.png')

    log("=" * 70)
    log("  EGO-Style Baseline — GRU + Context, No Indirection")
    log("=" * 70)
    log(f"  Device:     {device}")
    log(f"  value_dim:  {cfg.value_dim}  hidden_dim: {cfg.hidden_dim}")
    log(f"  GRU input:  z_t (dim={cfg.value_dim}) — NOT key_r (ESBN) — NO indirection")
    log(f"  task_id_dropout: 0.5  (like GEE-Separated)")
    log()
    log("  Hypothesis: Without indirection, the model cannot learn abstract")
    log("  relations that generalize to novel entities (Webb tasks).")
    log("  But it may still handle rule-switching with explicit context (WCST).")
    log()

    n_ego = sum(p.numel() for p in EGOBaseline(cfg).parameters() if p.requires_grad)
    log(f"  EGO parameters: {n_ego:,}")
    log()

    all_results = {}

    # =========================================================================
    # Part 1: Webb Tasks (skip if results cached from previous run)
    # =========================================================================
    webb_cached = {
        'normal': 0.960, 'none': 0.936, 'wrong': 0.905,
        'per_task_normal': {'same_different': 1.0, 'webb_rmts': 0.994,
                            'webb_dist3': 0.898, 'webb_idrules': 0.944},
        'per_task_none':   {'same_different': 1.0, 'webb_rmts': 0.992,
                            'webb_dist3': 0.832, 'webb_idrules': 0.922},
    }

    skip_webb = '--skip-webb' in sys.argv
    if skip_webb:
        log("=" * 70)
        log("  PART 1: Webb Tasks — SKIPPED (using cached results)")
        log("=" * 70)
        log(f"  EGO With task_id       overall=96.0%")
        log(f"  EGO No task_id         overall=93.6%")
        log()
        all_results['webb'] = webb_cached
    else:
        log("=" * 70)
        log("  PART 1: Webb-Faithful Tasks (4 tasks, mixed seq_len / output_dim)")
        log("=" * 70)
        log()

        factory_webb = DatasetFactory(
            embed_dim=cfg.input_dim, n_train=40, n_test=16, pool_seed=SEED)
        log(f"  {factory_webb.pool.summary()}")
        for tc in TASK_CONFIGS:
            log(f"  {tc['short']:<8} seq={tc['seq_len']}  output_dim={tc['output_dim']}")
        log()

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        train_data_webb = prepare_data(factory_webb, n_per_task=2000, split='train')
        test_data_webb  = prepare_data(factory_webb, n_per_task=500,  split='test')
        log(f"  Train: {len(train_data_webb[0])}  Test: {len(test_data_webb[0])}")
        log()

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        ego_webb_model = EGOBaseline(cfg, n_tasks=4).to(device)

        log("  Training EGO-Baseline on Webb tasks (100 epochs)...")
        t0 = time.time()
        train_webb(ego_webb_model, train_data_webb, device, lr=cfg.lr, n_epochs=100)
        log(f"  Training time: {time.time()-t0:.0f}s")
        log()

        webb_results = {}
        for mode, mode_label in [('normal', 'With task_id'),
                                  ('none',   'No task_id'),
                                  ('wrong',  'Wrong task_id')]:
            overall, pt = evaluate_webb(ego_webb_model, test_data_webb, device, task_id_mode=mode)
            webb_results[mode] = overall
            webb_results[f'{mode}_per_task'] = pt
            task_str = "  ".join(f"{tc['short']}={pt[tc['name']]:.0%}" for tc in TASK_CONFIGS)
            log(f"  EGO {mode_label:<18} overall={overall:.1%}  {task_str}")
        log()

        # Reference numbers
        log("  Reference results (from CLAUDE.md):")
        log(f"  {'Model':<18} {'With ID':>8} {'No ID':>8} {'Drop':>8}")
        log(f"  {'─'*48}")
        ref_models = [
            ('ESBN',          0.984, 0.984),
            ('GEE-Direct',    0.975, 0.610),
            ('GEE-Separated', 0.981, 0.982),
            ('EGO-Baseline',  all_results['webb']['normal'], all_results['webb']['none']),
        ]
        for name, w, n in ref_models:
            drop = w - n
            log(f"  {name:<18} {w:>7.1%} {n:>7.1%} {drop:>+7.1%}")
        log()

        all_results['webb'] = {
            'normal': all_results['webb']['normal'],
            'none':   all_results['webb']['none'],
            'wrong':  all_results['webb']['wrong'],
            'per_task_normal': webb_results['normal_per_task'],
            'per_task_none':   webb_results['none_per_task'],
        }

    # =========================================================================
    # Part 2: WCST
    # =========================================================================
    log("=" * 70)
    log("  PART 2: WCST — Wisconsin Card Sorting Task")
    log("=" * 70)
    log()
    log("  All trials have the same structure (seq_len=2: ref + test entity).")
    log("  Only context (task_id = sorting rule) distinguishes the tasks.")
    log()

    # Build entity pools for WCST
    pool = EntityPool(n_train=40, n_test=16, seed=SEED)
    train_entities = pool.get('train')
    test_entities  = pool.get('test')
    log(f"  {pool.summary()}")
    log()

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    ego_wcst = EGOBaseline_WCST(cfg).to(device)

    log("  Training EGO-WCST (50 epochs, n_per_rule=1000)...")
    t0 = time.time()

    # Use train_wcst infrastructure from gee_wcst.py (via inline replication)
    rng = np.random.RandomState(SEED)
    inputs_w, labels_w, task_ids_w = generate_wcst_data(train_entities, 1000, rng)
    ds_wcst = torch.utils.data.TensorDataset(
        torch.tensor(inputs_w), torch.tensor(labels_w), torch.tensor(task_ids_w))
    loader_wcst = torch.utils.data.DataLoader(ds_wcst, batch_size=32, shuffle=True)
    optimizer_wcst = torch.optim.Adam(ego_wcst.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(1, 51):
        ego_wcst.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_b, y_b, tid_b in loader_wcst:
            x_b, y_b, tid_b = x_b.to(device), y_b.to(device), tid_b.to(device)
            batch_logits = torch.zeros(x_b.shape[0], 2, device=device)
            for tid in tid_b.unique():
                mask = tid_b == tid
                batch_logits[mask] = ego_wcst(x_b[mask], task_id=tid.item(), device=device)
            loss = F.cross_entropy(batch_logits, y_b)
            optimizer_wcst.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ego_wcst.parameters(), 1.0)
            optimizer_wcst.step()
            total_loss += loss.item() * x_b.shape[0]
            correct += (batch_logits.argmax(-1) == y_b.argmax(-1)).sum().item()
            total += x_b.shape[0]
        if epoch % 10 == 0 or epoch == 1:
            log(f"    ep{epoch:>3}: loss={total_loss/total:.4f} acc={correct/total:.1%}")

    log(f"  Training time: {time.time()-t0:.0f}s")
    log()

    log("  Evaluating WCST (50 sessions)...")
    wcst_eval_normal = run_evaluation(ego_wcst, test_entities, device, n_sessions=50, task_id_mode='normal')
    wcst_eval_none   = run_evaluation(ego_wcst, test_entities, device, n_sessions=50, task_id_mode='none')
    wcst_eval_wrong  = run_evaluation(ego_wcst, test_entities, device, n_sessions=50, task_id_mode='wrong')

    log()
    log("  WCST Results:")
    log(f"  {'Model':<18} {'Acc':>7} {'Persev':>8} {'Completions':>13}")
    log(f"  {'─'*50}")
    ref_wcst = [
        ('ESBN',       0.729, 24.3, 5.9),
        ('GEE-Direct', 0.949,  1.1, 6.0),
        ('GEE-Sep.',   0.526, 41.5, 2.9),
    ]
    for name, acc, persev, compl in ref_wcst:
        log(f"  {name:<18} {acc:>6.1%} {persev:>8.1f} {compl:>12.1f}")
    log(f"  {'EGO-Baseline':<18} "
        f"{wcst_eval_normal['accuracy_mean']:>6.1%} "
        f"{wcst_eval_normal['persev_errors_mean']:>8.1f} "
        f"{wcst_eval_normal['completions_mean']:>12.1f}")
    log()
    log(f"  EGO no task_id:    acc={wcst_eval_none['accuracy_mean']:.1%}  "
        f"persev={wcst_eval_none['persev_errors_mean']:.1f}")
    log(f"  EGO wrong task_id: acc={wcst_eval_wrong['accuracy_mean']:.1%}  "
        f"persev={wcst_eval_wrong['persev_errors_mean']:.1f}")
    log()

    all_results['wcst'] = {
        'normal': {
            'accuracy_mean':      wcst_eval_normal['accuracy_mean'],
            'accuracy_std':       wcst_eval_normal['accuracy_std'],
            'persev_errors_mean': wcst_eval_normal['persev_errors_mean'],
            'persev_errors_std':  wcst_eval_normal['persev_errors_std'],
            'completions_mean':   wcst_eval_normal['completions_mean'],
            'completions_std':    wcst_eval_normal['completions_std'],
        },
        'none': {
            'accuracy_mean':      wcst_eval_none['accuracy_mean'],
            'persev_errors_mean': wcst_eval_none['persev_errors_mean'],
            'completions_mean':   wcst_eval_none['completions_mean'],
        },
        'wrong': {
            'accuracy_mean':      wcst_eval_wrong['accuracy_mean'],
            'persev_errors_mean': wcst_eval_wrong['persev_errors_mean'],
            'completions_mean':   wcst_eval_wrong['completions_mean'],
        },
    }

    # =========================================================================
    # Part 3: Unified Tasks
    # =========================================================================
    log("=" * 70)
    log("  PART 3: Unified Tasks (fixed seq_len=6, all binary output)")
    log("=" * 70)
    log()
    log("  All tasks have the same structure — context is essential.")
    log("  ESBN baseline struggles (~67%) without context signals.")
    log()

    try:
        from unified_tasks import UnifiedTaskFactory, N_TASKS as UNI_N_TASKS

        pool_uni = EntityPool(n_train=40, n_test=16, seed=SEED)
        factory_uni = UnifiedTaskFactory(
            pool_uni.get('train'), pool_uni.get('test'), seed=SEED)

        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        ego_uni = EGOBaseline_Unified(cfg, n_tasks=UNI_N_TASKS).to(device)

        log("  Training EGO-Unified (50 epochs)...")
        t0 = time.time()
        train_unified(ego_uni, factory_uni, device,
                      n_epochs=50, batches_per_task=100, batch_size=64, lr=5e-4)
        log(f"  Training time: {time.time()-t0:.0f}s")
        log()

        uni_results = {}
        for mode, label in [('normal', 'With task_id'),
                             ('none',   'No task_id'),
                             ('wrong',  'Wrong task_id')]:
            overall, pt = evaluate_unified(ego_uni, factory_uni, device,
                                           task_id_mode=mode, n_eval=500)
            uni_results[mode] = overall
            uni_results[f'{mode}_per_task'] = pt
            log(f"  EGO {label:<18} overall={overall:.1%}")
        log()

        log("  Unified Tasks Reference (from CLAUDE.md):")
        log(f"  {'Model':<18} {'With ID':>8} {'No ID':>8} {'Drop':>8}")
        log(f"  {'─'*48}")
        ref_uni = [
            ('ESBN',       0.671, 0.665),
            ('GEE',        0.912, 0.627),
            ('EGO-Baseline', uni_results['normal'], uni_results['none']),
        ]
        for name, w, n in ref_uni:
            drop = w - n
            log(f"  {name:<18} {w:>7.1%} {n:>7.1%} {drop:>+7.1%}")
        log()

        all_results['unified'] = {
            'normal': uni_results['normal'],
            'none':   uni_results['none'],
            'wrong':  uni_results.get('wrong', 0),
            'per_task_normal': uni_results['normal_per_task'],
        }

    except ImportError as e:
        log(f"  WARNING: Could not import unified_tasks: {e}")
        log("  Skipping Part 3.")
        all_results['unified'] = {'normal': 0.0, 'none': 0.0, 'wrong': 0.0}
        uni_results = {'normal': 0.0, 'none': 0.0}

    # =========================================================================
    # Summary
    # =========================================================================
    log("=" * 70)
    log("  FINAL SUMMARY — EGO Baseline vs ESBN / GEE")
    log("=" * 70)
    log()
    log("  Webb Tasks (novel entities, mixed seq_len):")
    log(f"  {'Model':<18} {'With ID':>8} {'No ID':>8} {'Drop':>8}  Interpretation")
    log(f"  {'─'*72}")
    final_webb = [
        ('ESBN',          0.984, 0.984, 'Indirection works, no context needed'),
        ('GEE-Direct',    0.975, 0.610, 'Context creates catastrophic dependency'),
        ('GEE-Separated', 0.981, 0.982, 'Best of both worlds'),
        ('EGO-Baseline',  all_results['webb']['normal'], all_results['webb']['none'],
         'No indirection → fails to generalize?'),
    ]
    for name, w, n, interp in final_webb:
        drop = w - n
        log(f"  {name:<18} {w:>7.1%} {n:>7.1%} {drop:>+7.1%}  {interp}")
    log()

    log("  WCST (attribute-specific rule switching, all trials structurally identical):")
    log(f"  {'Model':<18} {'Acc':>7} {'Persev':>8} {'Completions':>13}  Interpretation")
    log(f"  {'─'*72}")
    final_wcst = [
        ('ESBN',       0.729, 24.3, 5.9, 'No context → high perseveration'),
        ('GEE-Direct', 0.949,  1.1, 6.0, 'Explicit context is essential here'),
        ('GEE-Sep.',   0.526, 41.5, 2.9, 'Dropout too aggressive for WCST'),
        ('EGO-Baseline',
         wcst_eval_normal['accuracy_mean'],
         wcst_eval_normal['persev_errors_mean'],
         wcst_eval_normal['completions_mean'],
         'Context helps without indirection?'),
    ]
    for name, acc, persev, compl, interp in final_wcst:
        log(f"  {name:<18} {acc:>6.1%} {persev:>8.1f} {compl:>12.1f}  {interp}")
    log()

    log("  Unified Tasks (fixed seq=6, context essential):")
    log(f"  {'Model':<18} {'With ID':>8} {'No ID':>8} {'Drop':>8}  Interpretation")
    log(f"  {'─'*72}")
    uni_normal = all_results['unified']['normal']
    uni_none   = all_results['unified']['none']
    final_uni = [
        ('ESBN',          0.671, 0.665, 'No context → structural ambiguity'),
        ('GEE',           0.912, 0.627, 'Context helps but creates dependency'),
        ('EGO-Baseline',  uni_normal, uni_none, 'No indirection — can context save it?'),
    ]
    for name, w, n, interp in final_uni:
        drop = w - n
        log(f"  {name:<18} {w:>7.1%} {n:>7.1%} {drop:>+7.1%}  {interp}")
    log()

    log("  KEY FINDING:")
    ego_webb_acc = all_results['webb']['normal']
    if ego_webb_acc < 0.75:
        log("  EGO-Baseline FAILS on Webb tasks (< 75%). This confirms that")
        log("  indirection (GRU never seeing raw input) is essential for")
        log("  learning abstract relations that generalize to novel entities.")
    elif ego_webb_acc < 0.90:
        log(f"  EGO-Baseline achieves {ego_webb_acc:.0%} on Webb tasks — well below ESBN (98.4%).")
        log("  Partial generalization but indirection still provides a large advantage.")
    else:
        log(f"  EGO-Baseline achieves {ego_webb_acc:.0%} on Webb tasks — surprisingly high.")
        log("  This challenges the indirection hypothesis for these task variants.")
    log()

    # Save results JSON
    serializable = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    serializable[k][k2] = {
                        str(kk): float(vv) for kk, vv in v2.items()}
                else:
                    serializable[k][k2] = float(v2) if isinstance(v2, (float, np.floating)) else v2
        else:
            serializable[k] = v

    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    log(f"  Results saved to: {results_path}")

    # Generate figure
    plot_results(
        webb_results={'normal': all_results['webb']['normal'], 'none': all_results['webb']['none']},
        wcst_results=wcst_eval_normal,
        unified_results={'normal': all_results['unified']['normal'],
                         'none':   all_results['unified']['none']},
        out_path=fig_path,
    )

    log("\n  Done.")


if __name__ == '__main__':
    main()
