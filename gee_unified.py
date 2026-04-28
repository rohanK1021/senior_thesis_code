"""
GEE Unified Model
==================
Single GEE architecture combining:
  - Webb-faithful ESBN core (indirection, scalar gate, Webb confidence)
  - h_0 initialization from task_id (from GEE-Separated)
  - RCM output bias (from GEE-Direct, but NOT in keys)
  - Task_id dropout during training

The ESBN core is copied verbatim from the verified implementation.
Keys and memory are NEVER touched by context — pure ESBN indirection.

Usage:
    cfg = GEEConfig()
    model = GEE(cfg, n_tasks=4)
    logits = model(x_seq, task_id=2, output_dim=2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from gee_model import GEEConfig, RCMContextModule
from esbn_pure import ContextNorm, AttributeAwareEncoder


class GEE(nn.Module):
    """
    GEE: GRU-ESBN-EGO unified architecture.

    Webb-faithful ESBN core with two context pathways:
      1. h_0 = tanh(Linear(task_onehot)) — task-dependent GRU starting state
      2. logits += ctx_to_out(C_T) — RCM context biases output at final step

    Context NEVER enters keys or memory. The indirection pathway is pure ESBN.

    Task_id dropout (p=dropout_p): during training, randomly drops task_id.
    When dropped: h_0 = tanh(bias) (learned default), RCM receives no task signal.
    Forces the model to maintain structural reasoning without context.

    Flags:
      use_h0:  enable h_0 initialization from task_id
      use_rcm: enable RCM output bias
      Both False → pure ESBN (baseline)
    """
    def __init__(self, cfg: GEEConfig, n_tasks: int = 4,
                 use_h0: bool = True, use_rcm: bool = True,
                 dropout_p: float = 0.3):
        super().__init__()
        self.cfg = cfg
        self.n_tasks = n_tasks
        self.use_h0 = use_h0
        self.use_rcm = use_rcm
        self.dropout_p = dropout_p

        # === Webb-faithful ESBN core (do not modify) ===
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.contextnorm = ContextNorm(cfg.value_dim)
        self.gru = nn.GRU(cfg.key_dim + 1, cfg.hidden_dim, batch_first=True)
        self.key_w_out = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.g_out = nn.Linear(cfg.hidden_dim, 1)  # scalar gate
        self.confidence_gain = nn.Parameter(torch.ones(1))
        self.confidence_bias = nn.Parameter(torch.zeros(1))

        # Dual output heads (for tasks with different output dims)
        self.y_out_2 = nn.Linear(cfg.hidden_dim, 2)
        self.y_out_4 = nn.Linear(cfg.hidden_dim, 4)

        # === GEE additions ===
        # h_0 initialization from task_id
        if use_h0:
            self.h0_proj = nn.Linear(n_tasks, cfg.hidden_dim)

        # RCM: integrates (raw_input + task_onehot + prev_prediction) → context
        if use_rcm:
            rcm_input_dim = cfg.input_dim + n_tasks
            self.rcm = RCMContextModule(rcm_input_dim, 2, cfg.context_dim)
            self.ctx_to_out_2 = nn.Linear(cfg.context_dim, 2, bias=False)
            self.ctx_to_out_4 = nn.Linear(cfg.context_dim, 4, bias=False)

    def forward(self, x_seq, task_id=None, output_dim=2, device=None,
                noise_sigma=0.0, h0_noise_sigma=0.0, rcm_noise_sigma=0.0,
                h0_gain=1.0, rcm_gain=1.0):
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        # ── Encode + context normalization (Webb-faithful) ──
        z_seq = self.encoder(
            x_seq.reshape(-1, cfg.input_dim)
        ).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.contextnorm(z_seq)

        # ── Task_id dropout ──
        effective_task_id = task_id
        if self.training and task_id is not None and random.random() < self.dropout_p:
            effective_task_id = None

        task_onehot = torch.zeros(batch, self.n_tasks, device=device)
        if effective_task_id is not None:
            task_onehot[:, effective_task_id] = 1.0
        if noise_sigma > 0:
            task_onehot = task_onehot + torch.randn_like(task_onehot) * noise_sigma

        # ── h_0 from task_id ──
        if self.use_h0:
            h0_pre = self.h0_proj(task_onehot)
            hidden = torch.tanh(h0_gain * h0_pre).unsqueeze(0)
            if h0_noise_sigma > 0:
                hidden = hidden + torch.randn_like(hidden) * h0_noise_sigma
        else:
            hidden = torch.zeros(1, batch, cfg.hidden_dim, device=device)

        # ── RCM state ──
        if self.use_rcm:
            c_t = torch.zeros(batch, cfg.context_dim, device=device)
            y_prev = torch.zeros(batch, 2, device=device)

        # ── ESBN core loop (Webb-faithful, do not modify) ──
        key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)

        for t in range(seq_len + 1):
            if t == seq_len:
                z_t = torch.zeros(batch, 1, cfg.value_dim, device=device)
                x_raw = torch.zeros(batch, cfg.input_dim, device=device)
            else:
                z_t = z_seq[:, t, :].unsqueeze(1)
                x_raw = x_seq[:, t, :]

            # RCM update (if enabled) — does NOT touch keys/memory
            if self.use_rcm:
                rcm_in = torch.cat([x_raw, task_onehot], dim=-1)
                c_t = self.rcm(rcm_in, y_prev, c_t)
                if rcm_gain != 1.0:
                    c_t = c_t * rcm_gain
                if rcm_noise_sigma > 0:
                    c_t = c_t + torch.randn_like(c_t) * rcm_noise_sigma

            # GRU controller
            gru_out, hidden = self.gru(key_r, hidden)

            # Write key — pure ESBN, no context modulation
            key_w = F.relu(self.key_w_out(gru_out))

            # Scalar gate
            g = torch.sigmoid(self.g_out(gru_out))

            # Output (computed at every step, but only final step matters)
            y_out = self.y_out_4 if output_dim == 4 else self.y_out_2
            y_pred_linear = y_out(gru_out).squeeze(1)
            if self.use_rcm:
                ctx_out = self.ctx_to_out_4 if output_dim == 4 else self.ctx_to_out_2
                y_pred_linear = y_pred_linear + ctx_out(c_t)
                y_prev = torch.softmax(y_pred_linear.detach(), dim=-1)

            # Read from memory (Webb-faithful)
            if t == 0:
                key_r = torch.zeros(batch, 1, cfg.key_dim + 1, device=device)
            else:
                w_k = F.softmax((z_t * M_v).sum(dim=2), dim=1)
                c_k = torch.sigmoid(
                    (z_t * M_v).sum(dim=2) * self.confidence_gain
                    + self.confidence_bias
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
