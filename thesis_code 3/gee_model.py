"""
GEE Model v3: GRU-ESBN-EGO Hybrid
===================================
Built on the corrected pure ESBN that matches Webb et al. (97.3% overall).

Two-α ablation design:
  α_key: K_w = K_base + α_key * ReLU(W_key_ctx @ C_t)
  α_out: logits = Linear(cat(h_t, z_ret) + α_out * project(C_t))

  α_key=0, α_out=0  →  Pure ESBN (exact same architecture)
  α_key>0, α_out=0  →  Context-modulated keys only
  α_key=0, α_out>0  →  Context-modulated output only
  α_key>0, α_out>0  →  Full GEE

All corrections from ESBN debugging:
  - Deep learned encoder (12-dim one-hot → value_dim)
  - TCN (Temporal Context Normalization)
  - Memory: z_t queries VALUES, retrieves KEYS (Webb's mechanism)
  - Extra timestep (seq_len+1), output at final step
  - Gating on retrieved key + learnable confidence
  - Output from GRU hidden state

Requirements: torch, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass


def log(msg="", end="\n"):
    print(msg, end=end, flush=True)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GEEConfig:
    # Dimensions
    input_dim:   int = 12    # raw one-hot (preserves attribute structure)
    value_dim:   int = 128   # encoder output / memory value (Webb: z_size=128)
    key_dim:     int = 256   # abstract key (Webb: key_size=256)
    hidden_dim:  int = 512   # GRU controller (Webb: hidden_size=512)
    context_dim: int = 64    # RCM context vector (GEE-specific)
    output_dim:  int = 2     # binary output

    # Two-α ablation
    alpha_key: float = 0.5   # context modulation on write keys
    alpha_out: float = 0.5   # context modulation on output

    # Multi-task
    n_tasks: int = 4

    # Training
    n_epochs:   int = 100
    lr:         float = 5e-4
    batch_size: int = 32
    seed:       int = 42


# =============================================================================
# Components
# =============================================================================

class AttributeAwareEncoder(nn.Module):
    """
    Attribute-aware encoder. Each attribute gets its own learned subspace.
    Two entities sharing an attribute have identical vectors in that subspace.
    Critical for DistributionOf3 (shared attribute detection).
    """
    ATTR_SLICES = [(0, 4), (4, 8), (8, 10), (10, 12)]

    def __init__(self, input_dim, value_dim):
        super().__init__()
        n_attrs = len(self.ATTR_SLICES)
        self.dim_per_attr = value_dim // n_attrs
        remainder = value_dim - self.dim_per_attr * n_attrs

        self.attr_encoders = nn.ModuleList()
        for i, (start, end) in enumerate(self.ATTR_SLICES):
            attr_size = end - start
            out_dim = self.dim_per_attr + (1 if i < remainder else 0)
            self.attr_encoders.append(nn.Sequential(
                nn.Linear(attr_size, out_dim * 2),
                nn.ReLU(),
                nn.Linear(out_dim * 2, out_dim),
            ))

        self.norm = nn.LayerNorm(value_dim)

    def forward(self, x):
        parts = []
        for i, (start, end) in enumerate(self.ATTR_SLICES):
            parts.append(self.attr_encoders[i](x[:, start:end]))
        return self.norm(torch.cat(parts, dim=-1))


class TCN(nn.Module):
    """Temporal Context Normalization — Webb's exact implementation.
    Full sequence mean/var. Intentionally non-causal."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z_seq):
        eps = 1e-8
        z_mu = z_seq.mean(1)
        z_sigma = (z_seq.var(1) + eps).sqrt()
        z_seq = (z_seq - z_mu.unsqueeze(1)) / z_sigma.unsqueeze(1)
        return z_seq * self.gamma + self.beta


class RCMContextModule(nn.Module):
    """
    Recurrent Context Module (from EGO).
    Integrates stimulus + task_id + previous reward into running context C_t.
    """
    def __init__(self, input_dim, reward_dim, context_dim):
        super().__init__()
        self.gru = nn.GRUCell(input_dim + reward_dim, context_dim)

    def forward(self, x_t, reward_prev, c_prev):
        inp = torch.cat([x_t, reward_prev], dim=-1)
        return torch.tanh(self.gru(inp, c_prev))


# =============================================================================
# Full GEE Model
# =============================================================================

class GEEModel(nn.Module):
    """
    GRU-ESBN-EGO Hybrid with two-α ablation.

    Per-sequence:
      1. Encode all timesteps (deep encoder) + TCN
      2. Run RCM to build context C_t at each step
      3. Run GRU controller for seq_len+1 steps:
         - GRU receives gated(k_retrieved, conf) — indirection
         - Write key: K_w = K_base + α_key * ReLU(W @ C_t)
         - Memory: z_t queries M_v, retrieves M_k and M_v
      4. Output: Linear(cat(h_t, z_ret) + α_out * project(C_t)) at final step
    """
    def __init__(self, cfg: GEEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder + TCN
        self.encoder = AttributeAwareEncoder(cfg.input_dim, cfg.value_dim)
        self.tcn = TCN(cfg.value_dim)

        # RCM: input = raw stimulus + task_id + prev reward
        rcm_input_dim = cfg.input_dim + cfg.n_tasks
        self.rcm = RCMContextModule(rcm_input_dim, cfg.output_dim, cfg.context_dim)

        # GRU controller: input = gated(retrieved_key + confidence)
        self.gru = nn.GRUCell(cfg.key_dim + 1, cfg.hidden_dim)
        self.key_write = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.key_query = nn.Linear(cfg.hidden_dim, cfg.key_dim)
        self.gate_head = nn.Linear(cfg.hidden_dim, 1)  # scalar gate (Webb)

        # Context → key modulation (α_key) — separate projections for write and query
        self.ctx_to_key = nn.Linear(cfg.context_dim, cfg.key_dim, bias=False)
        self.ctx_to_key_q = nn.Linear(cfg.context_dim, cfg.key_dim, bias=False)

        # Context → output modulation (α_out)
        # Projects into same dim as (h_t || z_ret) concatenation
        self.ctx_to_out = nn.Linear(cfg.context_dim, cfg.hidden_dim + cfg.value_dim, bias=False)

        # Output head: from GRU hidden + retrieved value (+ optional context)
        self.output_head = nn.Linear(cfg.hidden_dim + cfg.value_dim, cfg.output_dim)

        # Learnable confidence
        self.conf_gain = nn.Parameter(torch.ones(1))
        self.conf_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_seq, task_id=None, device=None):
        """
        x_seq:   (batch, seq_len, input_dim)
        task_id: int or None
        Returns: logits (batch, output_dim)
        """
        if device is None:
            device = x_seq.device
        batch, seq_len, _ = x_seq.shape
        cfg = self.cfg

        # Encode + TCN
        z_seq = self.encoder(x_seq.reshape(-1, cfg.input_dim)).reshape(batch, seq_len, cfg.value_dim)
        z_seq = self.tcn(z_seq)

        # Task one-hot
        task_onehot = torch.zeros(batch, cfg.n_tasks, device=device)
        if task_id is not None:
            task_onehot[:, task_id] = 1.0

        # Initialize states
        h = torch.zeros(batch, cfg.hidden_dim, device=device)
        c_t = torch.zeros(batch, cfg.context_dim, device=device)
        y_prev = torch.zeros(batch, cfg.output_dim, device=device)
        key_r = torch.zeros(batch, cfg.key_dim + 1, device=device)
        z_ret = torch.zeros(batch, cfg.value_dim, device=device)

        stored_keys = []
        stored_vals = []

        for t in range(seq_len + 1):
            if t < seq_len:
                z_t = z_seq[:, t, :]
                x_raw = x_seq[:, t, :]
            else:
                z_t = torch.zeros(batch, cfg.value_dim, device=device)
                x_raw = torch.zeros(batch, cfg.input_dim, device=device)

            # RCM: update context
            rcm_in = torch.cat([x_raw, task_onehot], dim=-1)
            c_t = self.rcm(rcm_in, y_prev, c_t)

            # GRU controller: key_r already gated from previous step
            h = self.gru(key_r, h)

            # Read FIRST: z_t queries value slot (M_v)
            if len(stored_keys) > 0:
                M_k = torch.stack(stored_keys, dim=1)
                M_v = torch.stack(stored_vals, dim=1)
                sim = (M_v * z_t.unsqueeze(1)).sum(dim=-1)
                w = F.softmax(sim, dim=-1)
                c = torch.sigmoid(sim * self.conf_gain + self.conf_bias)
                # Webb: concat keys + per-entry confidence, weighted sum, scalar gate
                M_kc = torch.cat([M_k, c.unsqueeze(-1)], dim=-1)
                retrieved = (w.unsqueeze(-1) * M_kc).sum(dim=1)
                g = torch.sigmoid(self.gate_head(h))
                key_r = g * retrieved
                z_ret = (w.unsqueeze(-1) * M_v).sum(dim=1)
            else:
                key_r = torch.zeros(batch, cfg.key_dim + 1, device=device)

            # Write AFTER read
            k_base = F.relu(self.key_write(h))
            k_w = k_base + cfg.alpha_key * F.relu(self.ctx_to_key(c_t))
            if t < seq_len:
                stored_keys.append(k_w)
                stored_vals.append(z_t)

            # Output: cat(h_t, z_ret) + α_out * project(C_t)
            out_repr = torch.cat([h, z_ret], dim=-1) + cfg.alpha_out * self.ctx_to_out(c_t)
            logits = self.output_head(out_repr)
            y_prev = torch.softmax(logits.detach(), dim=-1)

        return logits  # from final (extra) timestep


# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, loader, optimizer, device, task_id=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch, task_id=task_id, device=device)
        loss = F.cross_entropy(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x_batch.shape[0]
        preds = logits.argmax(dim=-1)
        labels = y_batch.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += x_batch.shape[0]
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, task_id=None):
    model.eval()
    correct, total = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch, task_id=task_id, device=device)
        preds = logits.argmax(dim=-1)
        labels = y_batch.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += x_batch.shape[0]
    return correct / max(total, 1)


def get_dataloader(inputs, targets, batch_size, shuffle=True):
    if isinstance(inputs, list):
        inputs = np.array(inputs, dtype=np.float32)
    if isinstance(targets, list):
        targets = np.array(targets, dtype=np.float32)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# =============================================================================
# Main — 2x2 ablation on single task (quick validation)
# =============================================================================

def main():
    from dataset import DatasetFactory
    import time

    cfg = GEEConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    factory = DatasetFactory(embed_dim=cfg.input_dim, n_train=40, n_test=16,
                             pool_seed=cfg.seed)

    log("=" * 65)
    log("  GEE v3 — 2x2 Ablation Demo (Same/Different)")
    log("=" * 65)
    log(f"  Device:      {device}")
    log(f"  input_dim:   {cfg.input_dim} (raw one-hot)")
    log(f"  value_dim:   {cfg.value_dim}  key_dim: {cfg.key_dim}")
    log(f"  context_dim: {cfg.context_dim}")
    log(f"  {factory.pool.summary()}")
    log()

    tr_inp, tr_lbl, _ = factory.same_different.generate(2000, 'train')
    te_inp, te_lbl, _ = factory.same_different.generate(500, 'test')
    train_loader = get_dataloader(tr_inp, tr_lbl, cfg.batch_size)
    test_loader = get_dataloader(te_inp, te_lbl, cfg.batch_size, shuffle=False)

    ablations = [
        ("Pure ESBN",      0.0, 0.0),
        ("Keys only",      0.5, 0.0),
        ("Output only",    0.0, 0.5),
        ("Full GEE",       0.5, 0.5),
    ]

    results = {}
    for label, ak, ao in ablations:
        log(f"{'─'*65}")
        log(f"  {label} (α_key={ak}, α_out={ao})")
        log(f"{'─'*65}")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        cfg.alpha_key = ak
        cfg.alpha_out = ao
        model = GEEModel(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"  Params: {n_params:,}")

        best_test = 0.0
        t0 = time.time()
        for epoch in range(1, 51):
            loss, train_acc = train_epoch(model, train_loader, optimizer, device, task_id=0)
            test_acc = evaluate(model, test_loader, device, task_id=0)
            if test_acc > best_test:
                best_test = test_acc
            if epoch % 10 == 0 or epoch == 1:
                log(f"  ep{epoch:>3}: loss={loss:.4f} train={train_acc:.1%} test={test_acc:.1%}")

        results[label] = best_test
        log(f"  Best: {best_test:.1%} ({time.time()-t0:.0f}s)")
        log()

    log("=" * 65)
    log("  2x2 Ablation Results (Same/Different)")
    log("=" * 65)
    log(f"  {'Variant':<20} {'α_key':>6} {'α_out':>6} {'Test':>8}")
    log(f"  {'─'*42}")
    for (label, ak, ao), acc in zip(ablations, results.values()):
        log(f"  {label:<20} {ak:>6.1f} {ao:>6.1f} {acc:>7.1%}")
    log("=" * 65)


if __name__ == '__main__':
    main()
